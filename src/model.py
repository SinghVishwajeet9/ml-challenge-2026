"""Stages `train` and `predict`: LightGBM matcher + decision step.

train    fit on features/<train-subset> (negatives subsampled, re-weighted by 1/neg_frac
         so probabilities stay calibrated), score features/val, choose the decision
         policy on val, report F0.5.
predict  score features/<subset> and apply the chosen decision -> work/pred/<subset>.parquet

Decision
  1. exclusivity: each S2/S3 record may only go to the S1 where its probability is
     highest (ground truth: every S2/S3 record matches at most one S1).
  2. per S1, one of:
       threshold  keep candidates with p >= t
       expected   keep the top-k prefix maximising expected F0.5; k = 0 (singleton)
                  scores P(no match) = prod(1 - p)
"""
from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from common import Paths, log, switch, timed
import evaluate

ID_COLS = ["s1_rid", "other_rid", "label"]
BETA2 = 0.25
PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 127,
    "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1,
    "num_threads": int(os.environ.get("BER_THREADS", "0")),
}
# Optional overrides (hyperparameter grid): BER_LGB_PARAMS='{"num_leaves": 255}'
PARAMS.update(json.loads(os.environ.get("BER_LGB_PARAMS", "{}")))
REST_NEG_FRAC = 0.10
NUM_ROUNDS = 1500
WIDE_MIN = 0.001     # predict keeps features of pairs at or above this for the stacker
EARLY_STOP = 50
SEED = 7


def _matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select(pl.col(c).cast(pl.Float32) for c in cols).to_numpy()


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS]


# --------------------------------------------------------------------------- decision

def exclusivity(scored: pl.DataFrame) -> pl.DataFrame:
    """Keep, for each S2/S3 record, only its best S1."""
    return scored.filter(pl.col("p") == pl.col("p").max().over("other_rid"))


def decide_threshold(scored: pl.DataFrame, t: float) -> pl.DataFrame:
    return exclusivity(scored).filter(pl.col("p") >= t).select("s1_rid", "other_rid")


def decide_expected(scored: pl.DataFrame, min_p: float = 0.05) -> pl.DataFrame:
    """Per S1, the top-k prefix maximising  (1+b2) * sum_top_k(p) / (b2 * T + k),
    where T = sum of all candidate p of that S1 (expected number of true matches).
    Empty set scores P(no match) = prod(1 - p)."""
    # T and P(empty) use all candidates of the S1, before exclusivity.
    stats = scored.group_by("s1_rid").agg(
        pl.col("p").sum().alias("T"),
        (1 - pl.col("p")).log().sum().exp().alias("p_empty"))
    d = (exclusivity(scored).filter(pl.col("p") >= min_p)
         .join(stats, on="s1_rid")
         .sort(["s1_rid", "p"], descending=[False, True])
         .with_columns(pl.col("p").cum_sum().over("s1_rid").alias("cum_p"),
                       (pl.int_range(pl.len()).over("s1_rid") + 1).alias("k")))
    d = d.with_columns(((1 + BETA2) * pl.col("cum_p") / (BETA2 * pl.col("T") + pl.col("k")))
                       .alias("ef"))
    best = d.group_by("s1_rid").agg(pl.col("ef").max().alias("best_ef"),
                                    pl.col("k").filter(pl.col("ef") == pl.col("ef").max())
                                    .first().alias("best_k"),
                                    pl.col("p_empty").first())
    keep = best.filter(pl.col("best_ef") > pl.col("p_empty")).select("s1_rid", "best_k")
    return (d.join(keep, on="s1_rid").filter(pl.col("k") <= pl.col("best_k"))
            .select("s1_rid", "other_rid"))


def _score_decision(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series) -> float:
    return evaluate.f05(pred.rename({"s1_rid": "s1_id", "other_rid": "match_id"}),
                        gt, s1_ids)


def choose_policy(scored: pl.DataFrame, gt_rid: pl.DataFrame, s1_rids: pl.Series) -> dict:
    results = {}
    for t in np.arange(0.3, 0.981, 0.02):
        results[f"threshold@{t:.2f}"] = _score_decision(decide_threshold(scored, t), gt_rid,
                                                        s1_rids)
    results["expected"] = _score_decision(decide_expected(scored), gt_rid, s1_rids)
    for k, v in sorted(results.items(), key=lambda kv: -kv[1])[:6]:
        log.info("  %-16s F0.5 = %.4f", k, v)
    best = max(results, key=results.get)
    policy = ({"kind": "expected"} if best == "expected"
              else {"kind": "threshold", "t": float(best.split("@")[1])})
    policy["val_f05"] = results[best]
    return policy


def decide(scored: pl.DataFrame, policy: dict) -> pl.DataFrame:
    if policy["kind"] == "expected":
        return decide_expected(scored)
    return decide_threshold(scored, policy["t"])


# --------------------------------------------------------------------------- stages

def _gt_rids(paths: Paths, subset: str) -> tuple[pl.DataFrame, pl.Series]:
    norm = pl.read_parquet(paths.norm / f"{subset}.parquet", columns=["rid", "entity_id", "source"])
    gt = pl.read_parquet(paths.slices / f"{subset}_gt_pairs.parquet")
    gt = (gt.join(norm.select(pl.col("rid").alias("s1_rid"), pl.col("entity_id").alias("s1_id")),
                  on="s1_id")
          .join(norm.select(pl.col("rid").alias("other_rid"),
                            pl.col("entity_id").alias("match_id")), on="match_id")
          .select(pl.col("s1_rid").alias("s1_id"), pl.col("other_rid").alias("match_id")))
    return gt, norm.filter(pl.col("source") == 1)["rid"]


def train(paths: Paths, train_subset: str, neg_frac: float) -> None:
    models = paths.work / "models"
    models.mkdir(exist_ok=True)
    cols = feature_columns(pl.read_parquet(paths.work / "features" / f"{train_subset}.parquet",
                                           n_rows=1))
    # Sources: the training subset (sampled non-matches re-weighted by 1 / neg_frac), the
    # labelled rest subset if built (same sampling), pseudo-labelled rows (weight 1).
    # Each source becomes its own float32 array and is freed at once; LightGBM takes the
    # list, so the rows are never held twice (16 GB machines).
    srcs = [(paths.work / "features" / f"{train_subset}.parquet", 1.0 / neg_frac)]
    rest_f = paths.work / "features" / "rest.parquet"
    if switch("train_rest") and rest_f.exists():
        srcs.append((rest_f, 1.0 / REST_NEG_FRAC))
    if switch("pseudo_labels"):
        srcs += [(f, 1.0) for f in sorted((paths.work / "features").glob("pseudo_*.parquet"))]
    xs, ys, ws, xe, ye, we = [], [], [], [], [], []
    for f, neg_w in srcs:
        d = pl.read_parquet(f, columns=["s1_rid", "label", *cols])
        stop = ((d["s1_rid"].hash(SEED) % 10) == 0).to_numpy()
        y = d["label"].to_numpy()
        w = np.where(y, 1.0, neg_w)
        x = _matrix(d, cols)
        d = None
        xs.append(x[~stop]); ys.append(y[~stop]); ws.append(w[~stop])
        xe.append(x[stop]); ye.append(y[stop]); we.append(w[stop])
        x = None
        log.info("training rows from %s: %d (%d matches)", f.name, len(y), int(y.sum()))
    # Early stopping on 10% of the training S1 entities, so val is only used to choose
    # the decision policy and stays a clean estimate otherwise.
    log.info("train: %d rows, %d features | early-stop %d", sum(len(a) for a in ys), len(cols),
             sum(len(a) for a in ye))
    dtr = lgb.Dataset(xs, label=np.concatenate(ys), weight=np.concatenate(ws),
                      feature_name=cols, free_raw_data=True)
    des = lgb.Dataset(np.concatenate(xe), label=np.concatenate(ye), weight=np.concatenate(we),
                      reference=dtr)
    xe = None
    with timed("lightgbm"):
        booster = lgb.train(PARAMS, dtr, NUM_ROUNDS, valid_sets=[des],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                       lgb.log_evaluation(100)])
    booster.save_model(str(models / "lgb.txt"))
    dtr = des = xs = ys = ws = None      # free the training data before scoring val
    import gc
    gc.collect()
    imp = sorted(zip(cols, booster.feature_importance("gain")), key=lambda x: -x[1])
    log.info("top features: %s", ", ".join(f"{c} {g / sum(v for _, v in imp):.1%}"
                                           for c, g in imp[:15]))

    va = pl.read_parquet(paths.work / "features" / "val.parquet",
                         columns=["s1_rid", "other_rid", *cols])
    scored = va.select("s1_rid", "other_rid").with_columns(
        pl.Series("p", booster.predict(_matrix(va, cols)).astype(np.float32)))
    scored.write_parquet(models / "val_scored.parquet")
    gt, s1_rids = _gt_rids(paths, "val")
    log.info("val decision policies:")
    policy = choose_policy(scored, gt, s1_rids)
    policy["features"] = cols
    policy["best_iteration"] = booster.best_iteration
    (models / "policy.json").write_text(json.dumps(policy, indent=2))
    log.info("chosen policy %s | val F0.5 %.4f", {k: v for k, v in policy.items()
                                                 if k not in ("features",)}, policy["val_f05"])


def predict(paths: Paths, subset: str, country: str | None = None) -> None:
    """Score every candidate of `subset` and apply the decision policy.
    Streams one country at a time in feature chunks, so test (~69M candidate pairs)
    never holds all features in memory: only (s1_rid, other_rid, p) is kept."""
    import features as F

    models = paths.work / "models"
    policy = json.loads((models / "policy.json").read_text())
    booster = lgb.Booster(model_file=str(models / "lgb.txt"))
    import xenc
    # For the stacker: test and val2 also keep features of every pair with p >= WIDE_MIN,
    # written in parts to disk to protect memory.
    wide_dir = paths.work / "pred" / f"{subset}_wide"
    save_wide = subset in ("test", "val2", "delhi")
    tag = f"{country}_" if country else ""
    if save_wide:
        import shutil
        if country:
            wide_dir.mkdir(parents=True, exist_ok=True)
            for f in wide_dir.glob(f"{tag}part_*.parquet"):
                f.unlink()
        else:
            shutil.rmtree(wide_dir, ignore_errors=True)
            wide_dir.mkdir(parents=True)
    n_wide = 0
    scored, band = [], []
    parts = sorted((paths.work / "block" / subset).glob("*.parquet"))
    if country:
        parts = [f for f in parts if f.stem == country]
    for part in parts:
        with timed(f"score {subset}/{part.stem}"):
            for feats in F.build_features(pl.read_parquet(part),
                                          F.load_norm(paths, subset, part.stem),
                                          paths.work / "tmp_features"):
                p = booster.predict(_matrix(feats, policy["features"])).astype(np.float32)
                scored.append(feats.select("s1_rid", "other_rid").with_columns(pl.Series("p", p)))
                # Uncertain pairs keep their features for the cross-encoder.
                keep = (p >= xenc.BAND[0]) & (p <= xenc.BAND[1])
                band.append(feats.filter(pl.Series(keep))
                            .select("s1_rid", "other_rid", *policy["features"])
                            .with_columns(pl.Series("p", p[keep])))
                if save_wide:
                    w = p >= WIDE_MIN
                    (feats.filter(pl.Series(w)).select("s1_rid", "other_rid", *policy["features"])
                     .with_columns(pl.Series("p", p[w]))
                     .write_parquet(wide_dir / f"{tag}part_{n_wide:05d}.parquet"))
                    n_wide += 1
    scored = pl.concat(scored)
    out = paths.work / "pred"
    out.mkdir(exist_ok=True)
    if country:
        # One country per process (test); predict_merge combines the countries.
        pl.concat(band).write_parquet(out / f"{subset}_band_feats_{country}.parquet")
        scored.write_parquet(out / f"{subset}_scored_{country}.parquet")
        log.info("%s/%s: %d candidate pairs scored", subset, country, scored.height)
        return
    pl.concat(band).write_parquet(out / f"{subset}_band_feats.parquet")
    scored.write_parquet(out / f"{subset}_scored.parquet")
    matches = decide(scored, policy)
    matches.write_parquet(out / f"{subset}_matches.parquet")
    log.info("%s: %d candidate pairs scored, %d matches (policy %s)", subset, scored.height,
             matches.height, {k: v for k, v in policy.items() if k != "features"})


def predict_merge(paths: Paths, subset: str) -> None:
    """Combine per-country predict outputs into the usual files and decide."""
    out = paths.work / "pred"
    policy = json.loads((paths.work / "models" / "policy.json").read_text())
    scored = pl.concat([pl.read_parquet(f) for f in sorted(out.glob(f"{subset}_scored_*.parquet"))])
    pl.concat([pl.read_parquet(f) for f in sorted(out.glob(f"{subset}_band_feats_*.parquet"))]
              ).write_parquet(out / f"{subset}_band_feats.parquet")
    scored.write_parquet(out / f"{subset}_scored.parquet")
    matches = decide(scored, policy)
    matches.write_parquet(out / f"{subset}_matches.parquet")
    log.info("%s: %d candidate pairs scored, %d matches (merged)", subset, scored.height,
             matches.height)
