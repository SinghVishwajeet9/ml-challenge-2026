"""Stage `sample`: a small, fixed sample of the dev slice for fast rule development.

Contents (all from the dev slice; the val slice is never touched):
  1. True pairs: ~6k S1 entities (50% US / 50% India) with *all* their matches.
  2. Boost: extra entities until each hard group has >= BOOST_MIN true pairs:
       nonlatin      match name contains non-ASCII (transliteration)
       addr_missing  match has no address
       name_jacc0    zero name-token overlap with the S1 name (address-only matches)
  3. Non-matching pairs from the base entities:
       neg_random     random S2/S3 record of the same country (dev = few states,
                      so this is effectively same-state)
       neg_lookalike  S2/S3 record sharing a (not too common) name token
  4. France: records from *test* (no labels) for eyeballing only.

Outputs in work/sample/: records.parquet, entities.parquet, pairs.parquet, france.parquet
"""
from __future__ import annotations

import polars as pl

from common import Paths, log, timed

SEED = 7
BASE_S1_PER_COUNTRY = 3000
BOOST_MIN = 1000
N_NEG_RANDOM = 10_000
N_NEG_LOOKALIKE = 10_000
LOOKALIKE_MAX_DF = 500  # skip name tokens shared by more records than this
FRANCE_PER_SOURCE = 1000
GROUPS = ["nonlatin", "addr_missing", "name_jacc0"]

# Tokens too generic to define a look-alike.
_GENERIC = {
    "the", "of", "and", "pvt", "private", "ltd", "limited", "llc", "llp", "inc", "co",
    "corp", "corporation", "company", "group", "india", "services", "service", "center",
}


def _name_tokens(col: str) -> pl.Expr:
    return (
        pl.col(col).str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.strip_chars()
        .str.split(" ")
        .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
        .list.unique()
    )


def _true_pairs(recs: pl.DataFrame, gt: pl.DataFrame) -> pl.DataFrame:
    """All dev ground-truth pairs with group flags."""
    s1 = recs.filter(pl.col("source") == 1).select(
        pl.col("entity_id").alias("s1_id"), pl.col("rid").alias("s1_rid"),
        pl.col("name").alias("s1_name"), "country")
    other = recs.filter(pl.col("source") != 1).select(
        pl.col("entity_id").alias("match_id"), pl.col("rid").alias("other_rid"),
        pl.col("name").alias("o_name"), pl.col("address").alias("o_addr"))
    p = gt.join(s1, on="s1_id").join(other, on="match_id")
    p = p.with_columns(_name_tokens("s1_name").alias("t1"), _name_tokens("o_name").alias("t2"))
    inter = pl.col("t1").list.set_intersection("t2").list.len()
    return p.select(
        "s1_id", "s1_rid", "other_rid", "country",
        pl.col("o_name").str.contains(r"[^\x00-\x7F]").alias("nonlatin"),
        pl.col("o_addr").is_null().alias("addr_missing"),
        (inter == 0).alias("name_jacc0"),
    )


def _pick_entities(s1: pl.DataFrame, tp: pl.DataFrame) -> pl.DataFrame:
    base = pl.concat([
        s1.filter(pl.col("country") == c).sample(
            min(BASE_S1_PER_COUNTRY, s1.filter(pl.col("country") == c).height), seed=SEED)
        for c in sorted(s1["country"].unique())
    ]).select("s1_id", pl.lit("base").alias("origin"))

    chosen = base
    ent_flags = tp.group_by("s1_id").agg(pl.col(g).sum() for g in GROUPS)
    for g in GROUPS:
        have = tp.join(chosen, on="s1_id", how="semi")[g].sum()
        if have >= BOOST_MIN:
            continue
        pool = (ent_flags.filter(pl.col(g) > 0)
                .join(chosen, on="s1_id", how="anti")
                .sample(fraction=1.0, shuffle=True, seed=SEED))
        need = pool.with_columns(pl.col(g).cum_sum().alias("cum")).filter(
            pl.col("cum").shift(1, fill_value=0) < BOOST_MIN - have)
        chosen = pl.concat([chosen, need.select("s1_id", pl.lit(f"boost:{g}").alias("origin"))])
        log.info("boost %-12s had %4d pairs, added %d entities", g, have, need.height)
    return chosen


def _negatives(recs: pl.DataFrame, base_s1: pl.DataFrame, gt: pl.DataFrame) -> pl.DataFrame:
    others = recs.filter(pl.col("source") != 1).select(
        pl.col("rid").alias("other_rid"), pl.col("entity_id").alias("match_id"),
        "country", "name")
    s1 = base_s1.select("s1_id", "s1_rid", "country", "name")

    # Random same-country pairs.
    rnd = []
    for c in sorted(s1["country"].unique()):
        a = s1.filter(pl.col("country") == c).sample(N_NEG_RANDOM, with_replacement=True, seed=SEED)
        b = others.filter(pl.col("country") == c).sample(N_NEG_RANDOM, with_replacement=True, seed=SEED)
        rnd.append(pl.concat([a.select("s1_id", "s1_rid", "country"),
                              b.select("other_rid", "match_id")], how="horizontal"))
    rnd = pl.concat(rnd).sample(N_NEG_RANDOM, seed=SEED).with_columns(
        pl.lit("neg_random").alias("kind"))

    # Look-alikes: share an informative name token.
    def tokens(df: pl.DataFrame, key: str) -> pl.DataFrame:
        return (df.select(key, "country", _name_tokens("name").alias("tok")).explode("tok")
                .filter(pl.col("tok").is_not_null() & (pl.col("tok").str.len_chars() >= 3)
                        & ~pl.col("tok").is_in(list(_GENERIC))))
    ot = tokens(others, "other_rid")
    df_ok = ot.group_by("country", "tok").len().filter(pl.col("len") <= LOOKALIKE_MAX_DF)
    look = (tokens(s1, "s1_rid").join(df_ok.select("country", "tok"), on=["country", "tok"])
            .join(ot, on=["country", "tok"])
            .unique(["s1_rid", "other_rid"])
            .join(s1.select("s1_rid", "s1_id"), on="s1_rid")
            .join(others.select("other_rid", "match_id"), on="other_rid"))
    look = look.join(gt, on=["s1_id", "match_id"], how="anti")
    look = look.sample(min(N_NEG_LOOKALIKE, look.height), seed=SEED).select(
        "s1_id", "s1_rid", "country", "other_rid", "match_id",
        pl.lit("neg_lookalike").alias("kind"))

    neg = pl.concat([rnd, look]).join(gt, on=["s1_id", "match_id"], how="anti")
    return neg.select("s1_id", "s1_rid", "other_rid", "country", "kind")


def run(paths: Paths) -> None:
    with timed("build sample"):
        recs = pl.read_parquet(paths.slices / "dev_records.parquet")
        gt = pl.read_parquet(paths.slices / "dev_gt_pairs.parquet")
        s1 = recs.filter(pl.col("source") == 1).select(
            pl.col("entity_id").alias("s1_id"), pl.col("rid").alias("s1_rid"), "country", "name")

        tp_all = _true_pairs(recs, gt)
        chosen = _pick_entities(s1, tp_all)
        entities = chosen.join(s1, on="s1_id").join(
            gt.group_by("s1_id").len("n_matches"), on="s1_id", how="left"
        ).with_columns(pl.col("n_matches").fill_null(0))

        tp = tp_all.join(chosen, on="s1_id").with_columns(pl.lit("true").alias("kind"))
        neg = _negatives(recs, entities.filter(pl.col("origin") == "base"), gt)
        pairs = pl.concat([
            tp.select("s1_id", "s1_rid", "other_rid", "country", "kind", "origin", *GROUPS),
            neg.join(chosen, on="s1_id").select(
                "s1_id", "s1_rid", "other_rid", "country", "kind", "origin",
                *[pl.lit(None, dtype=pl.Boolean).alias(g) for g in GROUPS]),
        ]).with_columns((pl.col("kind") == "true").alias("label"))

        rids = pl.concat([entities["s1_rid"], pairs["other_rid"]]).unique()
        records = recs.filter(pl.col("rid").is_in(rids.implode()))

        test = pl.read_parquet(paths.records("test"))
        france = pl.concat([
            test.filter((pl.col("country") == "France") & (pl.col("source") == s))
            .sample(FRANCE_PER_SOURCE, seed=SEED)
            for s in (1, 2, 3)
        ])

        records.write_parquet(paths.sample / "records.parquet")
        entities.write_parquet(paths.sample / "entities.parquet")
        pairs.write_parquet(paths.sample / "pairs.parquet")
        france.write_parquet(paths.sample / "france.parquet")

    log.info("entities by origin/country:\n%s",
             entities.group_by("origin", "country").len().sort("origin", "country"))
    log.info("pairs by kind:\n%s", pairs.group_by("kind").len().sort("kind"))
    t = pairs.filter(pl.col("label"))
    log.info("true pairs: %d (base %d) | %s", t.height, t.filter(pl.col("origin") == "base").height,
             " ".join(f"{g}={t[g].sum()}" for g in GROUPS))
    log.info("singleton S1 in sample: %d | records: %d | france: %d",
             (entities["n_matches"] == 0).sum(), records.height, france.height)
