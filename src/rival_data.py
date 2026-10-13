"""Input data for the rival-aware cross-encoder (score column "pf" in the stacker, tag f1).

The model reads a record next to its candidate S1 and the strongest competing S1 of the same record
("rival"). This module writes a self-contained folder <work>/rival/ with text pairs only, plus the
training / scoring scripts from src/rival_encoder/, so the GPU step needs nothing else:
  train.parquet   labelled uncertain pairs: tune (out-of-fold band), rest (crowded, Delhi held
                  out), pseudo-labelled test pairs (France in full, India / US sampled)
  score_<s>.parquet  uncertain pairs of val, val2, delhi, test to score (labels where known)
Columns: s1_rid, other_rid, text_s1, text_rec, text_rival, p_lgb, label, w, part.
text_rival is the strongest competing S1 of the same record (empty when there is none).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import model as M
import stack as S
import xenc as X
from common import Paths, log

K = ["s1_rid", "other_rid"]
BAND = X.BAND
PSEUDO_CAP = {"France": 250_000, "India": 120_000, "US": 120_000}
PSEUDO_W = 0.3


def _rivals(scored: pl.DataFrame) -> pl.DataFrame:
    """For every (s1, record) pair: the best-scoring other S1 of that record."""
    top2 = (scored.select(*K, "p").sort("p", descending=True)
            .group_by("other_rid", maintain_order=True).head(2)
            .with_columns(pl.int_range(pl.len()).over("other_rid").alias("r")))
    first = top2.filter(pl.col("r") == 0).select("other_rid", pl.col("s1_rid").alias("r1"))
    second = top2.filter(pl.col("r") == 1).select("other_rid", pl.col("s1_rid").alias("r2"))
    return first.join(second, on="other_rid", how="left")


def _with_block(paths: Paths, subset: str, scored: pl.DataFrame) -> pl.DataFrame:
    """tune / rest features keep only a sample of non-matches, so rivals missing from the
    scored pairs are filled from all blocking candidates (ordered after every scored pair)."""
    blk = (pl.scan_parquet(paths.work / "block" / subset / "*.parquet").select(*K, "block_score")
           .collect().join(scored.select(K), on=K, how="anti")
           .select(*K, (pl.col("block_score").cast(pl.Float64) - 1e9).alias("p")))
    return pl.concat([scored.select(*K, pl.col("p").cast(pl.Float64)), blk])


def _texts(paths: Paths, subset: str, pairs: pl.DataFrame, scored: pl.DataFrame) -> pl.DataFrame:
    if subset in ("tune", "rest"):
        scored = _with_block(paths, subset, scored)
    raw = X._raw_records(paths, subset)
    txt = raw.select("rid", X._text(pl.col("name"), pl.col("address"), True).alias("t"))
    riv = _rivals(scored)
    return (pairs.join(riv, on="other_rid", how="left")
            .with_columns(pl.when(pl.col("r1") == pl.col("s1_rid")).then(pl.col("r2"))
                          .otherwise(pl.col("r1")).alias("rival_rid"))
            .join(txt.rename({"rid": "s1_rid", "t": "text_s1"}), on="s1_rid")
            .join(txt.rename({"rid": "other_rid", "t": "text_rec"}), on="other_rid")
            .join(txt.rename({"rid": "rival_rid", "t": "text_rival"}), on="rival_rid", how="left")
            .with_columns(pl.col("text_rival").fill_null(""))
            .drop("r1", "r2", "rival_rid"))


def _lgb_scores(paths: Paths, feats_file: Path) -> pl.DataFrame:
    """sub7 LightGBM probabilities for a feature file (read in row slices)."""
    pol = json.loads((paths.work / "models" / "policy.json").read_text())
    booster = lgb.Booster(model_file=str(paths.work / "models" / "lgb.txt"))
    lf = pl.scan_parquet(feats_file)
    n = lf.select(pl.len()).collect().item()
    out = []
    for a in range(0, n, 500_000):
        d = lf.slice(a, 500_000).select(*K, "label", *pol["features"]).collect()
        out.append(d.select(*K, "label").with_columns(
            pl.Series("p", booster.predict(M._matrix(d, pol["features"])).astype(np.float32))))
    return pl.concat(out)


def _band(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter((pl.col("p") >= BAND[0]) & (pl.col("p") <= BAND[1]))


def _labels(paths: Paths, subset: str, pairs: pl.DataFrame) -> pl.DataFrame:
    gt, _ = S.gt_rids(paths, subset)
    gt = gt.select(pl.col("s1_id").alias("s1_rid"), pl.col("match_id").alias("other_rid"),
                   pl.lit(True).alias("label"))
    return pairs.join(gt, on=K, how="left").with_columns(pl.col("label").fill_null(False))


COLS = [*K, "text_s1", "text_rec", "text_rival", "p_lgb", "label", "w", "part"]


def _finish(df: pl.DataFrame, part: str) -> pl.DataFrame:
    return (df.rename({"p": "p_lgb"}).with_columns(pl.lit(part).alias("part"))
            .with_columns(pl.col("label").cast(pl.Boolean), pl.col("w").cast(pl.Float32),
                          pl.col("p_lgb").cast(pl.Float32))
            .select(COLS))


def build(paths: Paths) -> None:
    out = paths.work / "rival"
    out.mkdir(parents=True, exist_ok=True)
    train = []

    oof = pl.read_parquet(paths.work / "models" / "tune_oof.parquet")
    b = _band(oof).with_columns(pl.when(pl.col("label")).then(1.0).otherwise(1 / X.NEG_FRAC).alias("w"))
    train.append(_finish(_texts(paths, "tune", b, oof), "tune"))
    log.info("tune band %d", b.height)

    rest = _lgb_scores(paths, paths.work / "features" / "rest.parquet")
    b = _band(rest).with_columns(pl.when(pl.col("label")).then(1.0).otherwise(10.0).alias("w"))
    train.append(_finish(_texts(paths, "rest", b, rest), "rest"))
    log.info("rest band %d", b.height)
    rest = None

    test_scored = pl.read_parquet(paths.work / "pred" / "test_scored.parquet")
    for c, cap in PSEUDO_CAP.items():
        ps = pl.read_parquet(paths.work / "features" / f"pseudo_test_{c}.parquet", columns=[*K, "label"])
        ps = ps.join(test_scored, on=K, how="left").with_columns(pl.col("p").fill_null(0.0))
        pos, neg = ps.filter(pl.col("label")), ps.filter(~pl.col("label"))
        # keep the harder pseudo negatives first (highest LightGBM p)
        neg = neg.sort("p", descending=True).head(max(cap - min(pos.height, cap // 2), 0))
        ps = pl.concat([pos.sample(min(pos.height, cap // 2), seed=13), neg]).with_columns(
            pl.lit(PSEUDO_W).alias("w"))
        train.append(_finish(_texts(paths, "test", ps, test_scored), f"pseudo_{c}"))
        log.info("pseudo %s %d (pos %d)", c, ps.height, ps["label"].sum())
    pl.concat(train).sample(fraction=1.0, shuffle=True, seed=13).write_parquet(out / "train.parquet")

    for s in ("val", "val2", "delhi", "test"):
        sc = test_scored if s == "test" else S._scored(paths, s)
        b = pl.read_parquet(paths.work / "pred" / f"{s}_band_feats.parquet", columns=[*K, "p"])
        b = (_labels(paths, s, b) if s != "test" else b.with_columns(pl.lit(None, pl.Boolean).alias("label")))
        b = b.with_columns(pl.lit(1.0).alias("w"))
        _finish(_texts(paths, s, b, sc), s).write_parquet(out / f"score_{s}.parquet")
        log.info("score %s %d", s, b.height)

    copy_code(out)
    log.info("rival cross-encoder data written to %s", out)


def copy_code(out: Path) -> None:
    for f in (Path(__file__).parent / "rival_encoder").glob("*.py"):
        shutil.copy2(f, out / f.name)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../student_resource/dataset")
    ap.add_argument("--work-dir", default="work")
    a = ap.parse_args()
    build(Paths(a.data_dir, a.work_dir))
