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

import lightgbm as lgb
import numpy as np
import polars as pl

from common import Paths, log, timed
import evaluate

ID_COLS = ["s1_rid", "other_rid", "label"]
BETA2 = 0.25
PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 127,
    "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "num_threads": 0,
}
NUM_ROUNDS = 1500
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
    tr = pl.read_parquet(paths.work / "features" / f"{train_subset}.parquet")
    va = pl.read_parquet(paths.work / "features" / "val.parquet")
    cols = feature_columns(tr)
    # Early stopping on 10% of the training S1 entities, so val is only used to choose
    # the decision policy and stays a clean estimate otherwise.
    stop = (pl.col("s1_rid").hash(SEED) % 10) == 0
    es, tr = tr.filter(stop), tr.filter(~stop)
    log.info("train %s: %d rows (%d positive), %d features | early-stop %d | val %d",
             train_subset, tr.height, tr["label"].sum(), len(cols), es.height, va.height)

    def weights(df: pl.DataFrame) -> np.ndarray:
        return np.where(df["label"].to_numpy(), 1.0, 1.0 / neg_frac)

    dtr = lgb.Dataset(_matrix(tr, cols), label=tr["label"].to_numpy(), weight=weights(tr),
                      feature_name=cols, free_raw_data=True)
    des = lgb.Dataset(_matrix(es, cols), label=es["label"].to_numpy(), weight=weights(es),
                      reference=dtr)
    with timed("lightgbm"):
        booster = lgb.train(PARAMS, dtr, NUM_ROUNDS, valid_sets=[des],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                                       lgb.log_evaluation(100)])
    booster.save_model(str(models / "lgb.txt"))
    imp = sorted(zip(cols, booster.feature_importance("gain")), key=lambda x: -x[1])
    log.info("top features: %s", ", ".join(f"{c} {g / sum(v for _, v in imp):.1%}"
                                           for c, g in imp[:15]))

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


def predict(paths: Paths, subset: str) -> None:
    models = paths.work / "models"
    policy = json.loads((models / "policy.json").read_text())
    booster = lgb.Booster(model_file=str(models / "lgb.txt"))
    feats = pl.read_parquet(paths.work / "features" / f"{subset}.parquet")
    scored = feats.select("s1_rid", "other_rid").with_columns(
        pl.Series("p", booster.predict(_matrix(feats, policy["features"])).astype(np.float32)))
    out = paths.work / "pred"
    out.mkdir(exist_ok=True)
    scored.write_parquet(out / f"{subset}_scored.parquet")
    matches = decide(scored, policy)
    matches.write_parquet(out / f"{subset}_matches.parquet")
    log.info("%s: %d candidate pairs scored, %d matches", subset, scored.height, matches.height)
