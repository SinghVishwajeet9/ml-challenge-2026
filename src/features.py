"""Stage `features`: one feature row per candidate pair (output of `block`).

Feature groups
  blocking   signals kept from the block stage (source flags, IDF scores, trigram /
             address cosines, house match, block score and rank)
  group      per S1: rank/gap/count; per S2/S3 record: how many S1s it is a candidate
             for and whether this S1 is its best (each S2/S3 matches at most one S1)
  name       RapidFuzz scores on the normalized forms, token Jaccards, joined-name /
             website / initials matches, legal-form agreement, out-of-vocabulary name
  address    token Jaccard, fuzzy score, number / house / state agreement and conflicts

For training subsets, non-matching pairs can be subsampled (`neg_frac`) *after* group
features are computed, so group features always see the full candidate set.
Output: work/features/<subset>.parquet (with `label` when ground truth is available).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from common import Paths, log, timed

CHUNK = 1_000_000
SEED = 11

NORM_COLS = ["rid", "entity_id", "source", "name_norm", "name_core", "name_concat",
             "name_initials", "name_skel", "name_legal", "name_domain", "name_translit",
             "addr_norm", "addr_core", "addr_state", "addr_house", "addr_numbers",
             "addr_unit", "addr_missing"]


# --------------------------------------------------------------------------- helpers

def _split(col: str) -> pl.Expr:
    return (pl.col(col).fill_null("").str.split(" ")
            .list.eval(pl.element().filter(pl.element() != "")).list.unique())


def _jacc(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    inter = a.list.set_intersection(b).list.len()
    union = a.list.set_union(b).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(None).cast(pl.Float32)


def _overlap_flags(a: str, b: str, prefix: str) -> list[pl.Expr]:
    both = (pl.col(a).list.len() > 0) & (pl.col(b).list.len() > 0)
    shared = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    return [
        pl.when(both).then(shared > 0).otherwise(None).alias(f"{prefix}_eq"),
        pl.when(both).then(shared == 0).otherwise(None).alias(f"{prefix}_conflict"),
        shared.cast(pl.Int16).alias(f"{prefix}_shared"),
    ]


def _fuzzy(df: pl.DataFrame, a: str, b: str, scorer, name: str) -> pl.Series:
    x = df[a].fill_null("").to_list()
    y = df[b].fill_null("").to_list()
    return pl.Series(name, process.cpdist(x, y, scorer=scorer, workers=-1)
                     .astype(np.float32) / 100.0)


# --------------------------------------------------------------------------- groups

def group_features(c: pl.DataFrame) -> pl.DataFrame:
    """Features comparing a candidate with the other candidates of its S1 and with the
    other S1s competing for the same S2/S3 record."""
    s1 = pl.col("s1_rid")
    o = pl.col("other_rid")
    return c.with_columns(
        pl.len().over(s1).cast(pl.Int16).alias("s1_n_cands"),
        (pl.col("block_score").max().over(s1) - pl.col("block_score")).alias("s1_gap_top"),
        pl.col("name_sim").rank("ordinal", descending=True).over(s1).cast(pl.Int16)
        .alias("s1_rank_name"),
        pl.col("addr_sim").rank("ordinal", descending=True).over(s1).cast(pl.Int16)
        .alias("s1_rank_addr"),
        pl.len().over(o).cast(pl.Int16).alias("o_n_s1"),
        pl.col("block_score").rank("ordinal", descending=True).over(o).cast(pl.Int16)
        .alias("o_rank"),
        (pl.col("block_score").max().over(o) - pl.col("block_score")).alias("o_gap_top"),
        (pl.col("block_score") - pl.col("block_score").filter(
            pl.col("block_score") < pl.col("block_score").max()).max().over(o))
        .alias("o_margin_second"),
    )


# --------------------------------------------------------------------------- pairs

def _oov(norm: pl.DataFrame) -> pl.DataFrame:
    """Share of an S2/S3 name's core tokens never seen in S1 names of the same country
    (replacement names like 'Dovacira' are out of vocabulary)."""
    s1_vocab = (norm.filter(pl.col("source") == 1)
                .select("country", _split("name_core").alias("t")).explode("t")
                .drop_nulls().unique())
    tok = (norm.filter(pl.col("source") != 1)
           .select("rid", "country", _split("name_core").alias("t")).explode("t").drop_nulls())
    known = tok.join(s1_vocab.with_columns(pl.lit(True).alias("k")), on=["country", "t"],
                     how="left")
    return known.group_by("rid").agg(
        (1 - pl.col("k").fill_null(False).cast(pl.Float32).mean()).alias("o_name_oov"))


def _unit_ids(col: str) -> pl.Expr:
    """Identifiers inside unit / floor text: 'Flat No F3' -> [f3], 'No.3Rd Floor' -> [3],
    'Unit No 3B2' -> [3b2]. Tokens with a digit, or single letters ('Unit D')."""
    return (pl.col(col).fill_null("").str.to_lowercase()
            .str.replace_all(r"(\d+)(?:st|nd|rd|th)\b", "${1}")
            .str.extract_all(r"\b[a-z]*\d+[a-z0-9]*\b|\b[a-z]\b")
            .list.eval(pl.when(pl.element().str.strip_chars_start("0") == "").then(pl.lit("0"))
                       .otherwise(pl.element().str.strip_chars_start("0"))).list.unique())


def _uniqueness(norm: pl.DataFrame) -> pl.DataFrame:
    """How many S1 and S2/S3 records of the same country share a record's core name."""
    cnt = norm.group_by("country", "source", "name_core").len("c")
    n_s1 = cnt.filter(pl.col("source") == 1).select("country", "name_core",
                                                    pl.col("c").alias("name_n_s1"))
    n_o = (cnt.filter(pl.col("source") != 1).group_by("country", "name_core")
           .agg(pl.col("c").sum().alias("name_n_o")))
    return (norm.select("rid", "country", "name_core")
            .join(n_s1, on=["country", "name_core"], how="left")
            .join(n_o, on=["country", "name_core"], how="left")
            .select("rid", pl.col("name_n_s1").fill_null(0).cast(pl.Int32),
                    pl.col("name_n_o").fill_null(0).cast(pl.Int32)))


def _idf_tables(norm: pl.DataFrame, col: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Token weights log(N / df) from the S2/S3 records of each country, and each
    record's total weight."""
    tok = (norm.select("rid", "country", "source", _split(col).alias("t")).explode("t")
           .drop_nulls().unique(["rid", "t"]))
    df = tok.filter(pl.col("source") != 1).group_by("country", "t").len("df")
    n = norm.filter(pl.col("source") != 1).group_by("country").len("n")
    w = (df.join(n, on="country")
         .select("country", "t", (pl.col("n") / pl.col("df")).log().cast(pl.Float32).alias("w")))
    tw = (tok.join(w, on=["country", "t"], how="left")
          .with_columns(pl.col("w").fill_null(pl.col("w").max())).select("rid", "t", "w"))
    return tw, tw.group_by("rid").agg(pl.col("w").sum().alias("W"))


class PairContext:
    """Per-record tables that pair features need, built once per country."""

    def __init__(self, norm: pl.DataFrame):
        self.side = (norm.select(NORM_COLS).join(_oov(norm), on="rid", how="left")
                     .join(_uniqueness(norm), on="rid", how="left"))
        self.name_w, self.name_tot = _idf_tables(norm, "name_core")
        self.addr_w, self.addr_tot = _idf_tables(norm, "addr_core")
        self.nums = norm.select("rid", pl.col("addr_numbers").list.eval(
            pl.element().cast(pl.Int64, strict=False)).alias("n"))


def _idf_pair(c: pl.DataFrame, tw: pl.DataFrame, tot: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Rarity-weighted token overlap: shared rare words count far more than common ones."""
    pairs = c.select("s1_rid", "other_rid")
    shared = (pairs.join(tw.rename({"rid": "s1_rid"}), on="s1_rid")
              .join(tw.select(pl.col("rid").alias("other_rid"), "t"), on=["other_rid", "t"])
              .group_by("s1_rid", "other_rid").agg(pl.col("w").sum().alias("ws")))
    d = (pairs.join(shared, on=["s1_rid", "other_rid"], how="left")
         .join(tot.rename({"rid": "s1_rid", "W": "w1"}), on="s1_rid", how="left")
         .join(tot.rename({"rid": "other_rid", "W": "w2"}), on="other_rid", how="left")
         .with_columns(pl.col("ws").fill_null(0)))
    return d.select(
        "s1_rid", "other_rid",
        (pl.col("ws") / (pl.col("w1") + pl.col("w2") - pl.col("ws"))).cast(pl.Float32)
        .alias(f"{prefix}_idf_jacc"),
        (pl.col("ws") / pl.col("w1")).cast(pl.Float32).alias(f"{prefix}_idf_cov1"),
        (pl.col("ws") / pl.col("w2")).cast(pl.Float32).alias(f"{prefix}_idf_cov2"),
        pl.col("ws").cast(pl.Float32).alias(f"{prefix}_idf_shared"))


def _num_pair(c: pl.DataFrame, nums: pl.DataFrame) -> pl.DataFrame:
    """Smallest difference between any two address numbers (384 vs 383 -> 1)."""
    return (c.select("s1_rid", "other_rid")
            .join(nums.rename({"rid": "s1_rid", "n": "n1"}), on="s1_rid")
            .join(nums.rename({"rid": "other_rid", "n": "n2"}), on="other_rid")
            .explode("n1").explode("n2").drop_nulls(["n1", "n2"])
            .group_by("s1_rid", "other_rid")
            .agg((pl.col("n1") - pl.col("n2")).abs().min().alias("num_min_absdiff")))


def pair_features(c: pl.DataFrame, ctx: PairContext) -> pl.DataFrame:
    a = ctx.side.rename(lambda x: "s1_rid" if x == "rid" else f"{x}_1")
    b = ctx.side.rename(lambda x: "other_rid" if x == "rid" else f"{x}_2")
    d = c.join(a, on="s1_rid").join(b, on="other_rid")
    keys = ["s1_rid", "other_rid"]
    d = (d.join(_idf_pair(c, ctx.name_w, ctx.name_tot, "name"), on=keys, how="left")
         .join(_idf_pair(c, ctx.addr_w, ctx.addr_tot, "addr"), on=keys, how="left")
         .join(_num_pair(c, ctx.nums), on=keys, how="left")
         .with_columns((pl.col("num_min_absdiff") <= 2).alias("num_near"))
         .rename({"name_n_s1_1": "s1_name_n_s1", "name_n_o_1": "s1_name_n_o",
                  "name_n_s1_2": "o_name_n_s1", "name_n_o_2": "o_name_n_o"}))

    d = d.with_columns(
        _split("name_core_1").alias("_nc1"), _split("name_core_2").alias("_nc2"),
        _split("name_skel_1").alias("_ns1"), _split("name_skel_2").alias("_ns2"),
        _split("name_legal_1").alias("_nl1"), _split("name_legal_2").alias("_nl2"),
        _split("addr_core_1").alias("_ac1"), _split("addr_core_2").alias("_ac2"),
        pl.col("addr_state_1").is_not_null().alias("_st1"),
        pl.col("addr_state_2").is_not_null().alias("_st2"),
        _unit_ids("addr_unit_1").alias("_u1"), _unit_ids("addr_unit_2").alias("_u2"),
    )
    d = d.with_columns(
        _jacc(pl.col("_nc1"), pl.col("_nc2")).alias("name_core_jacc"),
        _jacc(pl.col("_ns1"), pl.col("_ns2")).alias("name_skel_jacc"),
        _jacc(pl.col("_ac1"), pl.col("_ac2")).alias("addr_core_jacc"),
        # Legal forms: same set / clash (both present, disjoint).
        pl.when((pl.col("_nl1").list.len() > 0) & (pl.col("_nl2").list.len() > 0))
        .then(pl.col("_nl1").list.set_intersection(pl.col("_nl2")).list.len() == 0)
        .otherwise(None).alias("legal_conflict"),
        (pl.col("name_legal_1") == pl.col("name_legal_2")).alias("legal_eq"),
        # Joined-up names and websites.
        ((pl.col("name_concat_1") == pl.col("name_concat_2"))
         & (pl.col("name_concat_1") != "")).alias("concat_eq"),
        (pl.col("name_concat_1") == pl.col("name_domain_2")).fill_null(False).alias("domain_eq"),
        pl.col("name_domain_2").is_not_null().alias("o_is_domain"),
        ((pl.col("name_core_2") == pl.col("name_initials_1"))
         & (pl.col("name_initials_1").str.len_chars() >= 2)).alias("initials_eq"),
        pl.col("_nc1").list.len().cast(pl.Int8).alias("ntok_s1"),
        pl.col("_nc2").list.len().cast(pl.Int8).alias("ntok_o"),
        (pl.col("name_norm_1").str.len_chars().cast(pl.Int32)
         - pl.col("name_norm_2").str.len_chars().cast(pl.Int32))
        .abs().cast(pl.Int16).alias("name_len_diff"),
        # Address pieces.
        pl.when(pl.col("_st1") & pl.col("_st2"))
        .then(pl.col("addr_state_1") == pl.col("addr_state_2")).alias("state_eq"),
        *_overlap_flags("addr_house_1", "addr_house_2", "house"),
        *_overlap_flags("addr_numbers_1", "addr_numbers_2", "nums"),
        (pl.col("addr_unit_2").fill_null("") != "").alias("o_has_unit"),
        (pl.col("addr_unit_1").fill_null("") != "").alias("s1_has_unit"),
        *_overlap_flags("_u1", "_u2", "unit"),
        pl.col("name_translit_2").alias("o_translit"),
        pl.col("addr_missing_2").alias("o_addr_missing"),
        pl.col("source_2").cast(pl.Int8).alias("o_source"),
    )

    d = d.with_columns(
        _fuzzy(d, "name_norm_1", "name_norm_2", fuzz.ratio, "name_ratio"),
        _fuzzy(d, "name_norm_1", "name_norm_2", fuzz.token_set_ratio, "name_tset"),
        _fuzzy(d, "name_core_1", "name_core_2", fuzz.ratio, "core_ratio"),
        _fuzzy(d, "name_core_1", "name_core_2", fuzz.partial_ratio, "core_partial"),
        _fuzzy(d, "name_skel_1", "name_skel_2", fuzz.ratio, "skel_ratio"),
        _fuzzy(d, "name_skel_1", "name_skel_2", fuzz.token_set_ratio, "skel_tset"),
        pl.Series("concat_jw", process.cpdist(
            d["name_concat_1"].fill_null("").to_list(), d["name_concat_2"].fill_null("").to_list(),
            scorer=JaroWinkler.normalized_similarity, workers=-1).astype(np.float32)),
        _fuzzy(d, "name_concat_1", "name_concat_2", fuzz.partial_ratio, "concat_partial"),
        _fuzzy(d, "addr_norm_1", "addr_norm_2", fuzz.token_set_ratio, "addr_tset"),
        _fuzzy(d, "addr_norm_1", "addr_norm_2", fuzz.ratio, "addr_ratio"),
        _fuzzy(d, "addr_unit_1", "addr_unit_2", fuzz.token_set_ratio, "unit_tset"),
    )
    d = d.rename({"o_name_oov_2": "o_name_oov"})
    keep = [x for x in d.columns if not x.startswith("_") and not x.endswith(("_1", "_2"))]
    return d.select(keep)


# --------------------------------------------------------------------------- competition

COMPETE = ["name_tset", "name_idf_jacc", "addr_tset", "addr_idf_jacc", "name_skel_jacc"]


def competition_features(light: pl.DataFrame) -> pl.DataFrame:
    """For each similarity, how this S1 compares with the other S1s competing for the same
    S2/S3 record (and with the S1's other candidates). Needs the full candidate set."""
    # Group maxima first: a window expression nested inside another is extremely slow.
    light = light.with_columns(
        *[pl.col(c).max().over("s1_rid").alias(f"_s1max_{c}") for c in COMPETE],
        pl.col("name_tset").max().over("other_rid").alias("_omax_name_tset"))
    exprs = []
    for col in COMPETE:
        x = pl.col(col)
        second = x.filter(x < x.max()).max()
        exprs += [
            (x.max().over("other_rid") - x).cast(pl.Float32).alias(f"o_gap_{col}"),
            (x - second.over("other_rid")).cast(pl.Float32).alias(f"o_margin2_{col}"),
            x.rank("max", descending=True).over("other_rid").cast(pl.Int16)
            .alias(f"o_rank_{col}"),
            (x >= pl.col(f"_s1max_{col}") - 0.02).sum().over("s1_rid").cast(pl.Int16)
            .alias(f"s1_near_top_{col}"),
            x.rank("max", descending=True).over("s1_rid").cast(pl.Int16)
            .alias(f"s1_rank_{col}"),
        ]
    exprs += [
        (pl.col("name_idf_jacc") >= 0.99).sum().over("other_rid").cast(pl.Int16)
        .alias("o_n_s1_name_exact"),
        (pl.col("addr_idf_jacc") >= 0.9).sum().over("other_rid").cast(pl.Int16)
        .alias("o_n_s1_addr_exact"),
        (pl.col("name_idf_jacc") >= 0.99).sum().over("s1_rid").cast(pl.Int16)
        .alias("s1_n_o_name_exact"),
        pl.col("unit_eq").fill_null(False).sum().over("other_rid").cast(pl.Int16)
        .alias("o_n_s1_unit_eq"),
        (pl.col("unit_eq").fill_null(False)
         & (pl.col("name_tset") >= pl.col("_omax_name_tset") - 0.05))
        .sum().over("other_rid").cast(pl.Int16).alias("o_n_s1_unit_eq_top_name"),
    ]
    return light.select(exprs)


def build_features(cands: pl.DataFrame, norm: pl.DataFrame, tmp_dir):
    """Yield feature chunks for every candidate pair, in order.
    Pass 1 computes pair features chunk by chunk (spilled to tmp_dir) and keeps the few
    columns competition features need; pass 2 attaches competition features."""
    import shutil
    import uuid
    ctx = PairContext(norm)
    cands = group_features(cands)
    # Windows keeps a file locked while its data is in use, so temp files are not
    # deleted one by one: each call gets its own folder, and stale folders are removed.
    for old in tmp_dir.glob("*"):
        shutil.rmtree(old, ignore_errors=True)
    tmp_dir = tmp_dir / uuid.uuid4().hex
    tmp_dir.mkdir(parents=True, exist_ok=True)
    light, files = [], []
    for i in range(0, cands.height, CHUNK):
        f = pair_features(cands.slice(i, CHUNK), ctx)
        path = tmp_dir / f"part_{i // CHUNK:05d}.parquet"
        f.write_parquet(path)
        files.append((path, f.height))
        light.append(f.select("s1_rid", "other_rid", "unit_eq", *COMPETE))
    comp = competition_features(pl.concat(light))
    del light
    offset = 0
    for path, n in files:
        # memory_map=False: on Windows a mapped file cannot be deleted afterwards.
        chunk = pl.read_parquet(path, memory_map=False)
        yield pl.concat([chunk, comp.slice(offset, n)], how="horizontal")
        offset += n
    shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- stage

GT_FOR_SUBSET = {"dev": "dev_gt_pairs", "tune": "tune_gt_pairs", "val": "val_gt_pairs"}


def run(paths: Paths, subset: str, neg_frac: float | None) -> None:
    norm = pl.read_parquet(paths.norm / f"{subset}.parquet")
    gt_name = GT_FOR_SUBSET.get(subset)
    gt = None
    if gt_name or subset == "train":
        gt = pl.read_parquet(paths.slices / f"{gt_name}.parquet" if gt_name else paths.gt_pairs())
        ids = norm.select("rid", "entity_id")
        gt = (gt.join(ids.rename({"rid": "s1_rid", "entity_id": "s1_id"}), on="s1_id")
              .join(ids.rename({"rid": "other_rid", "entity_id": "match_id"}), on="match_id")
              .select("s1_rid", "other_rid", pl.lit(True).alias("label")))
    out_dir = paths.work / "features"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    parts = []
    for part in sorted((paths.work / "block" / subset).glob("*.parquet")):
        cands = pl.read_parquet(part)
        if gt is not None:
            cands = cands.join(gt, on=["s1_rid", "other_rid"], how="left").with_columns(
                pl.col("label").fill_null(False))
        country_norm = norm.filter(pl.col("country") == part.stem)
        with timed(f"features {subset}/{part.stem} ({cands.height} candidates)"):
            for f in build_features(cands, country_norm, paths.work / "tmp_features"):
                if gt is not None and neg_frac is not None and neg_frac < 1:
                    keep = f["label"].to_numpy() | (rng.random(f.height) < neg_frac)
                    f = f.filter(pl.Series(keep))
                parts.append(f)
    feats = pl.concat(parts)
    feats.write_parquet(out_dir / f"{subset}.parquet")
    if gt is not None:
        log.info("positives %d of %d rows", feats["label"].sum(), feats.height)
    log.info("wrote %s: %d rows x %d cols", out_dir / f"{subset}.parquet", *feats.shape)
