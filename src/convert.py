"""Stage `convert`: raw TSVs -> Parquet (one file per split) + exploded ground truth."""
from __future__ import annotations

import polars as pl

from common import NULL_TOKENS, SOURCES, SPLITS, Paths, log, read_tsv, timed


def _load_source(paths: Paths, split: str, source: int) -> pl.DataFrame:
    df = read_tsv(paths.tsv(split, source))
    expected = f"S{source}-"
    bad = df.filter(~pl.col("entity_id").str.starts_with(expected)).height
    if bad:
        raise ValueError(f"{split} source{source}: {bad} ids without prefix {expected}")
    return df.select(
        pl.col("entity_id"),
        pl.lit(source, dtype=pl.Int8).alias("source"),
        pl.col("business_name").str.strip_chars().alias("name"),
        pl.col("business_address")
        .str.strip_chars()
        .replace(NULL_TOKENS, None)
        .alias("address"),
        pl.col("country").str.strip_chars(),
    )


def convert_records(paths: Paths) -> None:
    for split in SPLITS:
        with timed(f"convert {split} records"):
            df = pl.concat([_load_source(paths, split, s) for s in SOURCES])
            if df["entity_id"].n_unique() != df.height:
                raise ValueError(f"{split}: duplicate entity_ids")
            # Dense integer key used for joins downstream (cheaper than strings).
            df = df.with_row_index("rid").with_columns(pl.col("rid").cast(pl.Int32))
            df.write_parquet(paths.records(split), compression="zstd")
            summary = df.group_by("source", "country").len().sort("source", "country")
            log.info("%s: %d rows\n%s", split, df.height, summary)


def convert_ground_truth(paths: Paths) -> None:
    with timed("convert ground truth"):
        gt = read_tsv(paths.gt_tsv())
        gt = gt.rename({"source1_entity_id": "s1_id", "matched_entity_ids": "ids"})
        pairs = (
            gt.filter(pl.col("ids").is_not_null() & (pl.col("ids") != ""))
            .with_columns(pl.col("ids").str.split(","))
            .explode("ids")
            .rename({"ids": "match_id"})
            .with_columns(pl.col("match_id").str.strip_chars())
        )
        if pairs["match_id"].n_unique() != pairs.height:
            log.warning("some S2/S3 ids appear in more than one S1 match list")
        pairs.write_parquet(paths.gt_pairs(), compression="zstd")

        entities = gt.select("s1_id").join(
            pairs.group_by("s1_id").len("n_matches"), on="s1_id", how="left"
        ).with_columns(pl.col("n_matches").fill_null(0).cast(pl.Int16))
        entities.write_parquet(paths.gt_entities(), compression="zstd")
        log.info(
            "ground truth: %d S1 entities, %d pairs, %d singletons",
            entities.height, pairs.height, (entities["n_matches"] == 0).sum(),
        )


def run(paths: Paths) -> None:
    convert_records(paths)
    convert_ground_truth(paths)
