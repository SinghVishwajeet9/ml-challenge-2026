"""Stage `slices`: region-based dev / tune / val slices of the training data.

Random S1 sampling thins out look-alike neighbours and inflates precision, so each slice
takes *whole regions* (US state / Indian state): every S1, S2, S3 record in the region,
distractors included.

Region of a record:
  - S1: state parsed from its (clean) address.
  - matched S2/S3: region of the S1 it belongs to (ground truth).
  - unmatched S2/S3: majority vote over address components, using a
    component -> region map learned from matched S2/S3 records. This handles codes
    ("MH"), native-script state names and city names without hand-written rules.

Region lists are written to configs/slices.json on first run and reused afterwards,
so slices are reproducible.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import polars as pl

from common import Paths, log, timed
from geo import INDIA_ALIASES, US_STATES

SEED = 42
# Fraction of each country's S1 entities per tier (disjoint regions).
TIER_FRACTIONS = {"val": 0.06, "dev": 0.05, "tune": 0.20}
TIER_OVERSHOOT = 1.3  # a region may push a tier to at most 130% of its target
MIN_COMPONENT_COUNT = 20
MIN_COMPONENT_PURITY = 0.9


def _components(df: pl.DataFrame) -> pl.DataFrame:
    """One row per (rid, address component), component lowercased and stripped."""
    return (
        df.select("rid", "country", pl.col("address").str.split(",").alias("comp"))
        .explode("comp", empty_as_null=True)
        .with_columns(pl.col("comp").str.strip_chars().alias("comp"))
        .filter(pl.col("comp").is_not_null() & (pl.col("comp") != ""))
    )


def _s1_regions(s1: pl.DataFrame) -> pl.DataFrame:
    comp = _components(s1)
    us = comp.filter(
        (pl.col("country") == "US") & pl.col("comp").is_in(list(US_STATES))
    ).with_columns(pl.col("comp").alias("state"))
    india = comp.filter(pl.col("country") == "India").with_columns(
        pl.col("comp").str.to_lowercase().replace_strict(INDIA_ALIASES, default=None).alias("state")
    ).filter(pl.col("state").is_not_null())
    # Last matching component wins (state usually comes last in S1 addresses).
    states = pl.concat([us, india]).group_by("rid", maintain_order=True).last()
    return s1.select("rid", "country").join(
        states.select("rid", "state"), on="rid", how="left"
    ).with_columns(
        pl.when(pl.col("state").is_not_null())
        .then(pl.col("country") + ":" + pl.col("state"))
        .alias("region")
    ).select("rid", "region")


def _assign_regions(records: pl.DataFrame, paths: Paths) -> pl.DataFrame:
    s1 = records.filter(pl.col("source") == 1)
    other = records.filter(pl.col("source") != 1)
    s1_reg = _s1_regions(s1).join(s1.select("rid", "entity_id"), on="rid")
    log.info("S1 region parsed for %.2f%%", 100 * s1_reg["region"].is_not_null().mean())

    gt = pl.read_parquet(paths.gt_pairs())
    owner = gt.join(
        s1_reg.select(pl.col("entity_id").alias("s1_id"), "region"), on="s1_id"
    ).select(pl.col("match_id").alias("entity_id"), "region")
    other = other.join(owner, on="entity_id", how="left")

    # Learn component -> region from matched S2/S3 records.
    matched = other.filter(pl.col("region").is_not_null())
    comp_map = (
        _components(matched)
        .join(matched.select("rid", "region"), on="rid")
        .with_columns(pl.col("comp").str.to_lowercase())
        .group_by("country", "comp", "region").len()
        .with_columns(
            (pl.col("len") / pl.col("len").sum().over("country", "comp")).alias("purity"),
            pl.col("len").sum().over("country", "comp").alias("total"),
        )
        .filter((pl.col("total") >= MIN_COMPONENT_COUNT)
                & (pl.col("purity") >= MIN_COMPONENT_PURITY))
        .select("country", "comp", pl.col("region").alias("vote"))
    )
    log.info("learned %d address components -> region", comp_map.height)

    unmatched = other.filter(pl.col("region").is_null()).drop("region")
    votes = (
        _components(unmatched)
        .with_columns(pl.col("comp").str.to_lowercase())
        .join(comp_map, on=["country", "comp"])
        .group_by("rid", "vote").len()
        .sort("len", descending=True)
        .group_by("rid").first()
        .select("rid", pl.col("vote").alias("region"))
    )
    unmatched = unmatched.join(votes, on="rid", how="left")
    log.info("unmatched S2/S3: %d records, region voted for %.2f%%",
             unmatched.height, 100 * unmatched["region"].is_not_null().mean())

    regions = pl.concat([
        s1_reg.select("rid", "region"),
        matched.select("rid", "region"),
        unmatched.select("rid", "region"),
    ])
    return records.join(regions, on="rid", how="left")


def _choose_tiers(records: pl.DataFrame) -> dict[str, list[str]]:
    sizes = (
        records.filter((pl.col("source") == 1) & pl.col("region").is_not_null())
        .group_by("country", "region").len().sort("region")
    )
    rng = random.Random(SEED)
    tiers: dict[str, list[str]] = {t: [] for t in TIER_FRACTIONS}
    for country in sorted(sizes["country"].unique()):
        rows = sizes.filter(pl.col("country") == country).rows(named=True)
        total = sum(r["len"] for r in rows)
        target = {t: f * total for t, f in TIER_FRACTIONS.items()}
        fill = {t: 0 for t in TIER_FRACTIONS}
        rng.shuffle(rows)
        for r in rows:
            open_tiers = [t for t in TIER_FRACTIONS
                          if fill[t] + r["len"] <= target[t] * TIER_OVERSHOOT]
            if not open_tiers:
                continue
            t = min(open_tiers, key=lambda t: fill[t] / target[t])
            if fill[t] >= target[t]:
                continue
            tiers[t].append(r["region"])
            fill[t] += r["len"]
        log.info("%s tiers (S1 share): %s", country,
                 {t: round(fill[t] / total, 3) for t in TIER_FRACTIONS})
    return {t: sorted(v) for t, v in tiers.items()}


def run(paths: Paths, config_path: Path) -> None:
    records = pl.read_parquet(paths.records("train"))
    with timed("assign regions"):
        records = _assign_regions(records, paths)

    if config_path.exists():
        tiers = json.loads(config_path.read_text())["tiers"]
        log.info("reusing region lists from %s", config_path)
    else:
        tiers = _choose_tiers(records)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(
            {"seed": SEED, "fractions": TIER_FRACTIONS, "tiers": tiers}, indent=2))
        log.info("wrote region lists to %s", config_path)

    gt = pl.read_parquet(paths.gt_pairs())
    for tier, regions in tiers.items():
        sl = records.filter(pl.col("region").is_in(regions))
        s1_ids = sl.filter(pl.col("source") == 1)["entity_id"]
        pairs = gt.filter(pl.col("s1_id").is_in(s1_ids.implode()))
        # Every true match of a chosen S1 must be present, even if its own region differs.
        missing = pairs.filter(~pl.col("match_id").is_in(sl["entity_id"].implode()))
        if missing.height:
            raise AssertionError(f"{tier}: {missing.height} matches missing from slice")
        sl.drop("region").write_parquet(paths.slices / f"{tier}_records.parquet")
        pairs.write_parquet(paths.slices / f"{tier}_gt_pairs.parquet")

        n_s1 = s1_ids.len()
        n_other = sl.height - n_s1
        n_distractor = n_other - pairs.height
        log.info("%-4s regions=%3d  S1=%8d  S2/S3=%9d  distractors=%.1f%%  singletons=%.1f%%",
                 tier, len(regions), n_s1, n_other, 100 * n_distractor / max(n_other, 1),
                 100 * (1 - pairs["s1_id"].n_unique() / max(n_s1, 1)))
