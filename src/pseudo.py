"""Stage `pseudo`: self-training rows from a subset's own confident first-pass scores.

Unlabeled regions (test, and val2 which stands in for an unseen region) have their own
patterns (e.g. France, test-only near-copies). Pairs the first-pass pipeline is sure about
become extra LightGBM training rows: final p >= POS_MIN -> match, p <= NEG_MAX (among
pairs LightGBM had not already ruled out, p_lgb >= NEG_MIN_LGB) -> non-match. No labels
are used. Measured on val2 (its real labels): LightGBM-only F0.5 0.9387 -> 0.9648.

Features are computed on the full candidate set (competition features need every
candidate of an S1 / record) and only the pseudo-labeled rows are kept.
Output: work/features/pseudo_<subset>.parquet
"""
from __future__ import annotations

import polars as pl

import features as F
from common import Paths, log, timed

POS_MIN = 0.98
NEG_MAX = 0.02
NEG_MIN_LGB = 0.001
# Share of S1 entities used per subset (bounds training memory on test).
S1_SHARE = {"test": 0.3, "val2": 1.0}
SEED = 7


def labels(paths: Paths, subset: str) -> pl.DataFrame:
    fp = pl.read_parquet(paths.work / "first_pass" / f"{subset}.parquet")
    lab = (fp.filter((pl.col("p") >= POS_MIN)
                     | ((pl.col("p") <= NEG_MAX) & (pl.col("p_lgb") >= NEG_MIN_LGB)))
           .select("s1_rid", "other_rid", (pl.col("p") >= POS_MIN).alias("label")))
    share = S1_SHARE.get(subset, 1.0)
    if share < 1:
        lab = lab.filter((pl.col("s1_rid").hash(SEED) % 1000) < int(share * 1000))
    return lab


def run(paths: Paths, subset: str, country: str | None = None) -> None:
    """country: only that country (own process on test, to bound memory); output
    work/features/pseudo_<subset>_<country>.parquet."""
    lab = labels(paths, subset)
    log.info("pseudo %s: %d rows (%d matches)", subset, lab.height, lab["label"].sum())
    parts = []
    files = sorted((paths.work / "block" / subset).glob("*.parquet"))
    if country:
        files = [f for f in files if f.stem == country]
    for part in files:
        cands = (pl.scan_parquet(part).join(lab.lazy(), on=["s1_rid", "other_rid"], how="left")
                 .collect())
        if cands["label"].is_null().all():
            continue
        with timed(f"pseudo features {subset}/{part.stem}"):
            for f in F.build_features(cands, F.load_norm(paths, subset, part.stem),
                                      paths.work / "tmp_features"):
                parts.append(f.filter(pl.col("label").is_not_null()))
    out = pl.concat(parts)
    (paths.work / "features").mkdir(exist_ok=True)
    name = f"pseudo_{subset}_{country}" if country else f"pseudo_{subset}"
    out.write_parquet(paths.work / "features" / f"{name}.parquet")
    log.info("pseudo %s: wrote %d rows (%d matches) with features", subset, out.height,
             out["label"].sum())
