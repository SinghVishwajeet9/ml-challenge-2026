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


def pair_features(c: pl.DataFrame, side: pl.DataFrame) -> pl.DataFrame:
    a = side.rename(lambda x: "s1_rid" if x == "rid" else f"{x}_1")
    b = side.rename(lambda x: "other_rid" if x == "rid" else f"{x}_2")
    d = c.join(a, on="s1_rid").join(b, on="other_rid")

    d = d.with_columns(
        _split("name_core_1").alias("_nc1"), _split("name_core_2").alias("_nc2"),
        _split("name_skel_1").alias("_ns1"), _split("name_skel_2").alias("_ns2"),
        _split("name_legal_1").alias("_nl1"), _split("name_legal_2").alias("_nl2"),
        _split("addr_core_1").alias("_ac1"), _split("addr_core_2").alias("_ac2"),
        pl.col("addr_state_1").is_not_null().alias("_st1"),
        pl.col("addr_state_2").is_not_null().alias("_st2"),
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
    )
    d = d.rename({"o_name_oov_2": "o_name_oov"})
    keep = [x for x in d.columns if not x.startswith("_") and not x.endswith(("_1", "_2"))]
    return d.select(keep)


# --------------------------------------------------------------------------- stage

GT_FOR_SUBSET = {"dev": "dev_gt_pairs", "tune": "tune_gt_pairs", "val": "val_gt_pairs"}


def run(paths: Paths, subset: str, neg_frac: float | None) -> None:
    norm = pl.read_parquet(paths.norm / f"{subset}.parquet")
    cands = pl.read_parquet(paths.work / "block" / subset / "*.parquet")
    with timed("group features"):
        cands = group_features(cands)

    gt_name = GT_FOR_SUBSET.get(subset)
    if gt_name or subset == "train":
        gt = pl.read_parquet(paths.slices / f"{gt_name}.parquet" if gt_name else paths.gt_pairs())
        ids = norm.select("rid", "entity_id")
        gt = (gt.join(ids.rename({"rid": "s1_rid", "entity_id": "s1_id"}), on="s1_id")
              .join(ids.rename({"rid": "other_rid", "entity_id": "match_id"}), on="match_id")
              .select("s1_rid", "other_rid", pl.lit(True).alias("label")))
        cands = cands.join(gt, on=["s1_rid", "other_rid"], how="left").with_columns(
            pl.col("label").fill_null(False))
        log.info("candidates %d | positives %d (%.1f%%)", cands.height, cands["label"].sum(),
                 100 * cands["label"].mean())
        if neg_frac is not None and neg_frac < 1:
            neg = cands.filter(~pl.col("label")).sample(fraction=neg_frac, seed=SEED)
            cands = pl.concat([cands.filter(pl.col("label")), neg])
            log.info("subsampled negatives to %.0f%% -> %d rows", 100 * neg_frac, cands.height)

    side = norm.select(NORM_COLS).join(_oov(norm), on="rid", how="left")
    out_dir = paths.work / "features"
    out_dir.mkdir(exist_ok=True)
    parts = []
    with timed(f"pair features ({cands.height} rows)"):
        for i in range(0, cands.height, CHUNK):
            parts.append(pair_features(cands.slice(i, CHUNK), side))
            log.info("  %d / %d", min(i + CHUNK, cands.height), cands.height)
    feats = pl.concat(parts)
    feats.write_parquet(out_dir / f"{subset}.parquet")
    log.info("wrote %s: %d rows x %d cols", out_dir / f"{subset}.parquet", *feats.shape)
