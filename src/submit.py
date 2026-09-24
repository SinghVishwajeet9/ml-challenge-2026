"""Stage `submit`: write the two submission files for the test set and validate them.

output/matching_results.tsv   source1_entity_id <TAB> matched_entity_ids
output/candidate_pairs.tsv    source1_entity_id <TAB> candidate_entity_ids

Every test S1 entity gets exactly one row (empty list when there is nothing); IDs are
comma-separated, S2/S3 only, no duplicates; matches are a subset of candidates.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl

from common import Paths, log


def _id_lists(pairs: pl.DataFrame, norm: pl.DataFrame, s1: pl.DataFrame,
              col: str) -> pl.DataFrame:
    ids = norm.select(pl.col("rid").alias("other_rid"), pl.col("entity_id").alias("oid"))
    lists = (pairs.select("s1_rid", "other_rid").unique()
             .join(ids, on="other_rid")
             .group_by("s1_rid").agg(pl.col("oid").sort().str.join(",").alias(col)))
    return (s1.join(lists, on="s1_rid", how="left")
            .select(pl.col("entity_id").alias("source1_entity_id"),
                    pl.col(col).fill_null(""))
            .sort("source1_entity_id"))


def _write_tsv(df: pl.DataFrame, path: Path) -> None:
    # Plain tab-separated text, no quoting (IDs never contain tabs or quotes).
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(df.columns) + "\n")
        for a, b in df.iter_rows():
            f.write(f"{a}\t{b}\n")


def run(paths: Paths, out_dir: Path, validator: Path) -> None:
    norm = pl.read_parquet(paths.norm / "test.parquet", columns=["rid", "entity_id", "source"])
    s1 = norm.filter(pl.col("source") == 1).select(pl.col("rid").alias("s1_rid"), "entity_id")
    cands = pl.read_parquet(paths.work / "block" / "test" / "*.parquet",
                            columns=["s1_rid", "other_rid"])
    matches = pl.read_parquet(paths.work / "pred" / "test_matches.parquet")
    missing = matches.join(cands, on=["s1_rid", "other_rid"], how="anti").height
    if missing:
        raise AssertionError(f"{missing} matches are not in the candidate set")

    out_dir.mkdir(parents=True, exist_ok=True)
    m = _id_lists(matches, norm, s1, "matched_entity_ids")
    c = _id_lists(cands, norm, s1, "candidate_entity_ids")
    _write_tsv(m, out_dir / "matching_results.tsv")
    _write_tsv(c, out_dir / "candidate_pairs.tsv")
    log.info("wrote %s: %d rows, %d with matches, %d matched ids",
             out_dir / "matching_results.tsv", m.height,
             (m["matched_entity_ids"] != "").sum(), matches.height)
    log.info("wrote %s: %d rows, %d candidate ids", out_dir / "candidate_pairs.tsv",
             c.height, cands.height)

    cmd = [sys.executable, str(validator),
           "--matching", str(out_dir / "matching_results.tsv"),
           "--candidate", str(out_dir / "candidate_pairs.tsv"),
           "--test-dir", str(paths.data / "test")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    log.info("validator (exit %d):\n%s%s", res.returncode, res.stdout, res.stderr)
