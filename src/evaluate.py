"""Local F0.5 scorer - macro-averaged per Source 1 entity, singletons included.

Per entity:
  - no true matches: 1.0 if prediction empty, else 0.0
  - otherwise F0.5 = 1.25*P*R / (0.25*P + R), 0.0 when there are no true positives
"""
from __future__ import annotations

import polars as pl

BETA2 = 0.25


def f05_per_entity(pred_pairs: pl.DataFrame, gt_pairs: pl.DataFrame,
                   s1_ids: pl.Series) -> pl.DataFrame:
    """
    pred_pairs, gt_pairs: columns (s1_id, match_id), one row per pair.
    s1_ids: every S1 entity being evaluated (entities absent from pred count as empty).
    Returns one row per S1 with n_pred, n_true, tp, f05.
    """
    ents = pl.DataFrame({"s1_id": s1_ids}).unique()
    pred = pred_pairs.join(ents, on="s1_id", how="semi").unique()
    gt = gt_pairs.join(ents, on="s1_id", how="semi")
    tp = pred.join(gt, on=["s1_id", "match_id"], how="inner").group_by("s1_id").len("tp")

    df = (
        ents.join(pred.group_by("s1_id").len("n_pred"), on="s1_id", how="left")
        .join(gt.group_by("s1_id").len("n_true"), on="s1_id", how="left")
        .join(tp, on="s1_id", how="left")
        .with_columns(pl.col("n_pred", "n_true", "tp").fill_null(0))
    )
    p = pl.col("tp") / pl.col("n_pred")
    r = pl.col("tp") / pl.col("n_true")
    f = (1 + BETA2) * p * r / (BETA2 * p + r)
    return df.with_columns(
        pl.when(pl.col("n_true") == 0)
        .then((pl.col("n_pred") == 0).cast(pl.Float64))
        .when(pl.col("tp") == 0)
        .then(0.0)
        .otherwise(f)
        .alias("f05")
    )


def f05(pred_pairs: pl.DataFrame, gt_pairs: pl.DataFrame, s1_ids: pl.Series) -> float:
    return float(f05_per_entity(pred_pairs, gt_pairs, s1_ids)["f05"].mean())


def recall_ceiling(cand_pairs: pl.DataFrame, gt_pairs: pl.DataFrame,
                   s1_ids: pl.Series) -> dict:
    """Blocking diagnostics: pair recall, and F0.5 if the model were perfect on candidates."""
    ents = pl.DataFrame({"s1_id": s1_ids}).unique()
    gt = gt_pairs.join(ents, on="s1_id", how="semi")
    cand = cand_pairs.join(ents, on="s1_id", how="semi").unique()
    hit = gt.join(cand, on=["s1_id", "match_id"], how="semi")
    return {
        "pair_recall": hit.height / max(gt.height, 1),
        "oracle_f05": f05(hit, gt, s1_ids),
        "cands_per_s1": cand.height / max(ents.height, 1),
    }


def _self_test() -> None:
    gt = pl.DataFrame({"s1_id": ["a", "a", "b"], "match_id": ["x", "y", "z"]})
    # a: example from the problem statement pattern (2 true, predict 2 right + 1 wrong)
    pred = pl.DataFrame({"s1_id": ["a", "a", "a", "c"], "match_id": ["x", "y", "w", "q"]})
    per = f05_per_entity(pred, gt, pl.Series(["a", "b", "c", "d"])).sort("s1_id")
    got = dict(zip(per["s1_id"], per["f05"].round(3)))
    assert got == {"a": 0.714, "b": 0.0, "c": 0.0, "d": 1.0}, got


if __name__ == "__main__":
    _self_test()
    print("evaluate self-test OK")
