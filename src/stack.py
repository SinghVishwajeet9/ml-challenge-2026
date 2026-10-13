"""Stacker and held-out reports.

The stacker is a small LightGBM that re-scores pairs using the blended LightGBM +
transformer probability plus competition features computed on that blended score (how an
S1 compares with the other S1s competing for the same S2/S3 record, and with its own other
candidates). It is trained on val; its estimate uses leave-one-state-out on val; val2 is a
fully independent check.

Stages (run_pipeline.py):
  holdout-report  LightGBM-only F0.5 per country and state for val, val2 and rest
  val2-check      val2 F0.5 of LightGBM alone and of transformer blends (fixed val settings)
  stack           fit + estimate on val, final stacker, val2 score -> work/stack/<tag>/
  stack-apply     apply the final stacker to test -> work/pred/test_matches.parquet
"""
from __future__ import annotations

import json

import lightgbm as lgb
import os

import numpy as np
import polars as pl

import evaluate as E
import model as M
from common import Paths, log, timed

ACTIVE_MIN = 0.001      # the stacker re-scores pairs with blended probability >= this
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4,
          "min_data_in_leaf": 200, "lambda_l2": 5.0, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
          "num_threads": M.PARAMS["num_threads"]}
ROUNDS = 500
THRESHOLDS = np.arange(0.40, 0.90, 0.02)
COMP = ["o_rank_pb", "o_margin_pb", "o_gap_pb", "o_n_pb05", "o_sum_pb",
        "s1_rank_pb", "s1_n_pb05", "s1_sum_pb", "s1_gap_pb"]


# --------------------------------------------------------------------------- helpers

def gt_rids(paths: Paths, subset: str) -> tuple[pl.DataFrame, pl.Series]:
    """Ground-truth pairs as (s1_id, match_id) record ids, and the subset's S1 ids."""
    norm = pl.read_parquet(paths.norm / f"{subset}.parquet",
                           columns=["rid", "entity_id", "source"])
    gt = (pl.read_parquet(paths.gt_pairs())
          .join(norm.select(pl.col("entity_id").alias("s1_id"), pl.col("rid").alias("s1_rid")),
                on="s1_id")
          .join(norm.select(pl.col("entity_id").alias("match_id"),
                            pl.col("rid").alias("other_rid")), on="match_id")
          .select(pl.col("s1_rid").alias("s1_id"), pl.col("other_rid").alias("match_id")))
    return gt, norm.filter(pl.col("source") == 1)["rid"]


def f05(pairs: pl.DataFrame, gt: pl.DataFrame, s1: pl.Series) -> float:
    return E.f05(pairs.select(pl.col("s1_rid").alias("s1_id"),
                              pl.col("other_rid").alias("match_id")), gt, s1)


def _scored(paths: Paths, subset: str) -> pl.DataFrame:
    if subset == "val":
        return pl.read_parquet(paths.work / "models" / "val_scored.parquet")
    return pl.read_parquet(paths.work / "pred" / f"{subset}_scored.parquet")


def _px(paths: Paths, subset: str, tag: str) -> pl.DataFrame:
    return pl.read_parquet(paths.work / "xenc" / tag / f"{subset}_scored.parquet")


def blended(paths: Paths, subset: str, tag: str) -> pl.DataFrame:
    """(s1_rid, other_rid, p, px, pb): LightGBM, transformer and their val-chosen blend.
    With EXTRA 'f1' also pf, the rival-aware cross-encoder score (null outside its band)."""
    w = json.loads((paths.work / "xenc" / tag / "blend.json").read_text())["w"]
    d = (_scored(paths, subset).join(_px(paths, subset, tag), on=["s1_rid", "other_rid"],
                                     how="left")
         .with_columns(pl.when(pl.col("px").is_not_null())
                       .then((1 - w) * pl.col("p") + w * pl.col("px"))
                       .otherwise(pl.col("p")).alias("pb")))
    if "f1" in EXTRA:
        pf = pl.read_parquet(paths.work / "xenc" / F1_TAG / f"{subset}_scored.parquet")
        d = d.join(pf.select("s1_rid", "other_rid", pl.col("px").alias("pf")),
                   on=["s1_rid", "other_rid"], how="left")
    return d


# --------------------------------------------------------------------------- extra inputs
# BER_STACK_EXTRA="f1,e1": f1 = local rival-aware cross-encoder, e1 = legal-form / exact
# spelling competition among the S1 candidates of a record (no-address twins).
EXTRA = set(filter(None, os.environ.get("BER_STACK_EXTRA", "").split(",")))
F1_TAG = "f1l"
F1_COLS = ["pf", "o_rank_pf", "o_margin_pf", "o_gap_pf"]
E1_COLS = ["legal_same", "raw_exact", "ci_exact", "tail_same", "o_n_legal_same",
           "o_only_legal_same", "o_n_raw_exact", "o_only_raw_exact", "o_n_ci_exact",
           "o_only_ci_exact", "o_n_tail_same", "o_only_tail_same", "o_n_core_same"]
E1_MIN = 0.02           # a candidate S1 counts as a competitor from this blended score


TRAP_COLS = ["s1_has_exact_house", "trap_rival", "s1_n_house_conflict_hi"]


def extra_cols() -> list[str]:
    return ((F1_COLS if "f1" in EXTRA else []) + (E1_COLS if "e1" in EXTRA else [])
            + (TRAP_COLS if "trap" in EXTRA else []))


def _names(paths: Paths, subset: str) -> pl.DataFrame:
    norm = pl.read_parquet(paths.norm / f"{subset}.parquet",
                           columns=["rid", "name_legal", "name_core"])
    if subset == "test":
        raw = pl.read_parquet(paths.records("test"), columns=["rid", "name"])
    else:
        raw = pl.read_parquet(paths.slices / f"{subset}_records.parquet", columns=["rid", "name"])
    nm = pl.col("name").fill_null("")
    return norm.join(raw, on="rid", how="left").select(
        "rid", "name_legal", "name_core", nm.alias("raw"),
        nm.str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars().alias("ci"),
        nm.str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()
        .str.split(" ").list.last().alias("tail"))


def extra_features(paths: Paths, subset: str, act: pl.DataFrame) -> pl.DataFrame:
    if "trap" in EXTRA and "house_eq" in act.columns:
        # the S1 already has a confident candidate with the same house number: rivals with a
        # conflicting house number are likely near-copy traps
        exact = ((pl.col("house_eq") == 1) & (pl.col("pb") >= 0.7))
        conf = (pl.col("house_conflict") == 1)
        act = act.with_columns(exact.any().over("s1_rid").cast(pl.Float32).alias("s1_has_exact_house"),
                               (conf & (conf & (pl.col("pb") >= 0.3)).sum().over("s1_rid").gt(0))
                               .cast(pl.Float32).alias("s1_n_house_conflict_hi"))
        act = act.with_columns((pl.col("s1_has_exact_house").cast(pl.Boolean) & conf)
                               .cast(pl.Float32).alias("trap_rival"))
    if "f1" in EXTRA:
        x = pl.col("pf")
        has = x.is_not_null()
        second = x.filter(x < x.max()).max()
        act = act.with_columns(
            pl.when(has).then(x.rank("ordinal", descending=True).over("other_rid"))
            .cast(pl.Float32).alias("o_rank_pf"),
            pl.when(has).then(x - second.over("other_rid")).alias("o_margin_pf"),
            pl.when(has).then(x.max().over("other_rid") - x).alias("o_gap_pf"))
    if "e1" in EXTRA:
        nm = _names(paths, subset)
        a = act.select("s1_rid", "other_rid", "pb").join(
            nm.rename({c: f"{c}_1" for c in nm.columns if c != "rid"}).rename({"rid": "s1_rid"}),
            on="s1_rid", how="left").join(
            nm.rename({c: f"{c}_2" for c in nm.columns if c != "rid"}).rename({"rid": "other_rid"}),
            on="other_rid", how="left")
        eq = lambda c: ((pl.col(f"{c}_1") == pl.col(f"{c}_2")) & (pl.col(f"{c}_1") != "")).fill_null(False)
        a = a.with_columns(eq("name_legal").alias("legal_same"), eq("raw").alias("raw_exact"),
                           eq("ci").alias("ci_exact"), eq("tail").alias("tail_same"),
                           eq("name_core").alias("core_same"))
        comp = pl.col("pb") >= E1_MIN
        cols = []
        for f in ("legal_same", "raw_exact", "ci_exact", "tail_same"):
            n = (pl.col(f) & comp).sum().over("other_rid").cast(pl.Float32)
            cols += [n.alias(f"o_n_{f}"), (pl.col(f) & (n == 1)).cast(pl.Float32).alias(f"o_only_{f}")]
        cols.append((pl.col("core_same") & comp).sum().over("other_rid").cast(pl.Float32)
                    .alias("o_n_core_same"))
        a = a.with_columns(cols).with_columns(
            pl.col("legal_same", "raw_exact", "ci_exact", "tail_same").cast(pl.Float32))
        act = act.join(a.select("s1_rid", "other_rid", *E1_COLS), on=["s1_rid", "other_rid"],
                       how="left")
    return act


def competition(d: pl.DataFrame) -> pl.DataFrame:
    x = pl.col("pb")
    second = x.filter(x < x.max()).max()
    return d.with_columns(
        x.rank("ordinal", descending=True).over("other_rid").cast(pl.Float32).alias("o_rank_pb"),
        (x - second.over("other_rid")).alias("o_margin_pb"),
        (x.max().over("other_rid") - x).alias("o_gap_pb"),
        (x >= 0.5).sum().over("other_rid").cast(pl.Float32).alias("o_n_pb05"),
        x.sum().over("other_rid").alias("o_sum_pb"),
        x.rank("ordinal", descending=True).over("s1_rid").cast(pl.Float32).alias("s1_rank_pb"),
        (x >= 0.5).sum().over("s1_rid").cast(pl.Float32).alias("s1_n_pb05"),
        x.sum().over("s1_rid").alias("s1_sum_pb"),
        (x.max().over("s1_rid") - x).alias("s1_gap_pb"))


def _features(paths: Paths, subset: str, cols: list[str]) -> pl.LazyFrame:
    if subset == "val":
        return pl.scan_parquet(paths.work / "features" / "val.parquet").select(
            "s1_rid", "other_rid", *cols)
    return pl.scan_parquet(paths.work / "pred" / f"{subset}_wide" / "*.parquet").select(
        "s1_rid", "other_rid", *cols)


def active_frame(paths: Paths, subset: str, tag: str, cols: list[str]) -> tuple[pl.DataFrame,
                                                                               pl.DataFrame]:
    """All pairs with their blended score, and the active pairs (pb >= ACTIVE_MIN) with
    blended score, competition and LightGBM features."""
    d = competition(blended(paths, subset, tag))
    act = d.filter(pl.col("pb") >= ACTIVE_MIN)
    act = act.join(_features(paths, subset, cols).collect(), on=["s1_rid", "other_rid"],
                   how="inner")
    return d.select("s1_rid", "other_rid", "pb"), extra_features(paths, subset, act)


def _final_scores(all_pb: pl.DataFrame, act: pl.DataFrame, p2: np.ndarray) -> pl.DataFrame:
    return (all_pb.join(act.select("s1_rid", "other_rid").with_columns(pl.Series("p2", p2)),
                        on=["s1_rid", "other_rid"], how="left")
            .select("s1_rid", "other_rid", pl.coalesce("p2", "pb").alias("p")))


def _best_threshold(scores: pl.DataFrame, gt, s1) -> tuple[float, float]:
    return max((f05(M.decide_threshold(scores, t), gt, s1), float(t)) for t in THRESHOLDS)


# --------------------------------------------------------------------------- reports

def holdout_report(paths: Paths) -> None:
    """LightGBM-only F0.5 per country and per state: is test India like Tamil Nadu?"""
    policy = json.loads((paths.work / "models" / "policy.json").read_text())
    out = {}
    for subset in ("val", "val2", "rest"):
        gt, s1 = gt_rids(paths, subset)
        pred = M.decide(_scored(paths, subset), policy)
        per = E.f05_per_entity(pred.select(pl.col("s1_rid").alias("s1_id"),
                                           pl.col("other_rid").alias("match_id")), gt, s1)
        norm = pl.read_parquet(paths.norm / f"{subset}.parquet",
                               columns=["rid", "country", "addr_state", "source"])
        per = per.join(norm.filter(pl.col("source") == 1).select(
            pl.col("rid").alias("s1_id"), "country", "addr_state"), on="s1_id")
        by_c = per.group_by("country").agg(pl.len().alias("S1"), pl.col("f05").mean().alias("f05"))
        by_s = (per.group_by("country", "addr_state")
                .agg(pl.len().alias("S1"), pl.col("f05").mean().alias("f05"))
                .filter(pl.col("S1") >= 5000).sort("S1", descending=True))
        out[subset] = {"overall": per["f05"].mean(), "by_country": by_c.rows(),
                       "by_state": by_s.rows()}
        log.info("[holdout] %s LightGBM-only F0.5 %.4f | by country %s", subset,
                 per["f05"].mean(), [(c, n, round(f, 4)) for c, n, f in by_c.rows()])
        for c, s, n, f in by_s.rows():
            log.info("[holdout]   %-6s %-18s S1 %7d  F0.5 %.4f", c, s, n, f)
    (paths.work / "holdout_report.json").write_text(json.dumps(out, indent=1, default=str))


def val2_check(paths: Paths, tags: list[str]) -> dict:
    """val2 F0.5 of LightGBM alone and of each tag's blend (weight/threshold from val)."""
    policy = json.loads((paths.work / "models" / "policy.json").read_text())
    gt, s1 = gt_rids(paths, "val2")
    res = {"lightgbm": f05(M.decide(_scored(paths, "val2"), policy), gt, s1)}
    for tag in tags:
        b = json.loads((paths.work / "xenc" / tag / "blend.json").read_text())
        d = blended(paths, "val2", tag).select("s1_rid", "other_rid", pl.col("pb").alias("p"))
        res[tag] = f05(M.decide_threshold(d, b["t"]), gt, s1)
    log.info("[val2] %s", {k: round(v, 4) for k, v in res.items()})
    (paths.work / "val2_check.json").write_text(json.dumps(res, indent=1))
    return res


# --------------------------------------------------------------------------- stacker

def _out(paths: Paths, tag: str):
    return paths.work / "stack" / (tag + "".join(f"_{e}" for e in sorted(EXTRA)))


def fit(paths: Paths, tag: str) -> None:
    out = _out(paths, tag)
    out.mkdir(parents=True, exist_ok=True)
    cols = json.loads((paths.work / "models" / "policy.json").read_text())["features"]
    feats = cols + ["p", "px", "pb"] + COMP + extra_cols()
    gt, s1 = gt_rids(paths, "val")

    with timed("[stack] build val frame"):
        all_pb, act = active_frame(paths, "val", tag, cols)
    states = (pl.read_parquet(paths.norm / "val.parquet", columns=["rid", "addr_state", "source"])
              .filter(pl.col("source") == 1)
              .select(pl.col("rid").alias("s1_rid"), pl.col("addr_state").alias("st")))
    act = act.join(states, on="s1_rid", how="left").with_columns(pl.col("st").fill_null("?"))
    labels = gt.rename({"s1_id": "s1_rid", "match_id": "other_rid"}).with_columns(
        pl.lit(True).alias("label"))
    act = act.join(labels, on=["s1_rid", "other_rid"], how="left").with_columns(
        pl.col("label").fill_null(False))
    X, y, st = M._matrix(act, feats), act["label"].to_numpy(), act["st"].to_numpy()

    with timed("[stack] leave-one-state-out estimate"):
        oof = np.zeros(len(y))
        for s in sorted(set(st)):
            tr = st != s
            oof[~tr] = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), ROUNDS).predict(X[~tr])
    f_stack, t_oof = _best_threshold(_final_scores(all_pb, act, oof), gt, s1)
    blend = json.loads((paths.work / "xenc" / tag / "blend.json").read_text())
    log.info("[stack] val: blend %.4f (t %.2f) | stacker leave-one-state-out %.4f (t %.2f)",
             blend["val_f05"], blend["t"], f_stack, t_oof)

    with timed("[stack] final fit on all of val"):
        booster = lgb.train(PARAMS, lgb.Dataset(X, y), ROUNDS)
    booster.save_model(str(out / "stack.txt"))

    # val2: fully independent check (never used above)
    f_val2_stack = f_val2_blend = None
    if (paths.work / "xenc" / tag / "val2_scored.parquet").exists():
        gt2, s12 = gt_rids(paths, "val2")
        all2, act2 = active_frame(paths, "val2", tag, cols)
        p2 = booster.predict(M._matrix(act2, feats))
        f_val2_stack = f05(M.decide_threshold(_final_scores(all2, act2, p2), t_oof), gt2, s12)
        f_val2_blend = f05(M.decide_threshold(all2.rename({"pb": "p"}), blend["t"]), gt2, s12)
    else:
        log.info("[stack] val2 inputs not found; val2 check skipped")
    # Delhi: crowded holdout, never trained on (needs pred/delhi_wide and transformer scores)
    f_delhi_stack = f_delhi_blend = None
    if (paths.work / "xenc" / tag / "delhi_scored.parquet").exists():
        gt3, s13 = gt_rids(paths, "delhi")
        all3, act3 = active_frame(paths, "delhi", tag, cols)
        p3 = booster.predict(M._matrix(act3, feats))
        f_delhi_stack = f05(M.decide_threshold(_final_scores(all3, act3, p3), t_oof), gt3, s13)
        f_delhi_blend = f05(M.decide_threshold(all3.rename({"pb": "p"}), blend["t"]), gt3, s13)
        log.info("[stack] delhi: blend %s | stacker %s", f_delhi_blend, f_delhi_stack)
    report = {"tag": tag, "extra": sorted(EXTRA), "val_blend": blend["val_f05"],
              "val_stack_loso": f_stack, "t_stack": t_oof, "t_blend": blend["t"],
              "w_blend": blend["w"], "val2_stack": f_val2_stack, "val2_blend": f_val2_blend,
              "delhi_stack": f_delhi_stack, "delhi_blend": f_delhi_blend, "features": feats}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    log.info("[stack] val2: blend %s | stacker %s", f_val2_blend, f_val2_stack)


def apply_test(paths: Paths, tag: str) -> None:
    rep = json.loads((_out(paths, tag) / "report.json").read_text())
    booster = lgb.Booster(model_file=str(_out(paths, tag) / "stack.txt"))
    cols = json.loads((paths.work / "models" / "policy.json").read_text())["features"]
    with timed("[stack] test competition features"):
        d = competition(blended(paths, "test", tag))
        all_pb = d.select("s1_rid", "other_rid", "pb")
        act = d.filter(pl.col("pb") >= ACTIVE_MIN).select(
            "s1_rid", "other_rid", "p", "px", "pb", *COMP, *(["pf"] if "f1" in EXTRA else []))
        del d
        act = extra_features(paths, "test", act)
    scored = []
    for part in sorted((paths.work / "pred" / "test_wide").glob("*.parquet")):
        f = pl.read_parquet(part, columns=["s1_rid", "other_rid", *cols]).join(
            act, on=["s1_rid", "other_rid"], how="inner")
        if f.height:
            scored.append(f.select("s1_rid", "other_rid").with_columns(
                pl.Series("p2", booster.predict(M._matrix(f, rep["features"])))))
    p2 = pl.concat(scored)
    final = (all_pb.join(p2, on=["s1_rid", "other_rid"], how="left")
             .select("s1_rid", "other_rid", pl.coalesce("p2", "pb").alias("p")))
    matches = M.decide_threshold(final, rep["t_stack"])
    matches.write_parquet(paths.work / "pred" / "test_matches.parquet")
    log.info("[stack] test: %d pairs re-scored, %d matches (threshold %.2f)", p2.height,
             matches.height, rep["t_stack"])
