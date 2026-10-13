"""Shared paths, I/O helpers and logging."""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

import polars as pl

SOURCES = (1, 2, 3)
SPLITS = ("train", "test")

# Literal placeholder strings that mean "missing" in the raw data.
NULL_TOKENS = ["", "None", "none", "NULL", "<NULL>", "null", "nan", "NaN", "N/A"]

log = logging.getLogger("ber")

_SWITCHES: dict | None = None


def switch(name: str) -> bool:
    """Component switch from configs/switches.json (default on); env BER_<NAME>=0/1 overrides."""
    global _SWITCHES
    env = os.environ.get("BER_" + name.upper())
    if env is not None:
        return env == "1"
    if _SWITCHES is None:
        f = Path(__file__).resolve().parents[1] / "configs" / "switches.json"
        _SWITCHES = json.loads(f.read_text()) if f.exists() else {}
    return bool(_SWITCHES.get(name, True))


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    log.info("%s ...", label)
    yield
    log.info("%s done in %.1fs", label, time.perf_counter() - t0)


class Paths:
    """All filesystem locations, derived from the data dir and work dir."""

    def __init__(self, data_dir: str | Path, work_dir: str | Path):
        self.data = Path(data_dir)
        self.work = Path(work_dir)
        self.raw = self.work / "raw"
        self.slices = self.work / "slices"
        self.sample = self.work / "sample"
        self.norm = self.work / "norm"
        for d in (self.raw, self.slices, self.sample, self.norm):
            d.mkdir(parents=True, exist_ok=True)

    def tsv(self, split: str, source: int) -> Path:
        return self.data / split / f"{split}_source{source}.tsv"

    def gt_tsv(self) -> Path:
        return self.data / "train" / "train_ground_truth.tsv"

    def records(self, split: str) -> Path:
        """All three sources of one split, one parquet."""
        return self.raw / f"{split}_records.parquet"

    def gt_pairs(self) -> Path:
        """Exploded ground truth: one row per (s1_id, match_id)."""
        return self.raw / "train_gt_pairs.parquet"

    def gt_entities(self) -> Path:
        """One row per train S1 entity with its match count."""
        return self.raw / "train_gt_entities.parquet"


def read_tsv(path: Path) -> pl.DataFrame:
    # quote_char=None: names contain literal '"' characters that must not be parsed.
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
