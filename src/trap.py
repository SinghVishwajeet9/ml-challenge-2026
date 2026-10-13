"""Stage `trap`: prior-shift correction for near-copy records.

Test holds more near-copies of real companies than the training data: same (or almost the
same) name, same street, different house number. Most of the extra ones are non-matches.
The model learned from training how often such a pair is still a match; on test that rate
is lower. Per country and pattern cell, the number of non-matches per S1 on test is
measured against a labeled reference (val / val2), and the final probability of pairs in
the cell is shifted:

    logit(p') = logit(p) - ln(N_test / N_ref)

N = non-matches per S1 among pairs with LightGBM p >= WIDE_MIN (same population on both
sides). Test non-matches = test pairs per S1 minus the reference true matches per S1
(true matches per S1 are assumed unchanged). An EM prior estimate is reported as a
cross-check. Pairs outside the cells keep their scores.

Also measures the candidate cutoff curve (candidates per S1 vs share of true matches kept).

Outputs in work/trap/: <subset>_final.parquet (cached final scores), report.json,
test_matches.parquet.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import model as M
import stack as S
from common import Paths, log, timed

CELL_COLS = ["name_tset", "addr_tset", "nums_conflict", "nums_eq"]
# Reference subsets (subset, country) per test country. France has no labels: US reference.
REF = {"US": [("val", "US")], "India": [("val", "India"), ("val2", "India")],
       "France": [("val", "US")]}
CUTOFFS = [0.0001, 0.0003, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
DECIDE_MIN = 0.3


def cell_expr() -> pl.Expr:
    street = (pl.col("nums_conflict").cast(pl.Boolean) & ~pl.col("nums_eq").cast(pl.Boolean)
              & (pl.col("addr_tset") >= 0.6))
    return (pl.when(street & (pl.col("name_tset") >= 0.95)).then(pl.lit("same_name"))
            .when(street & (pl.col("name_tset") >= 0.8)).then(pl.lit("similar_name"))
            .otherwise(pl.lit(None, dtype=pl.String)).alias("cell"))


def _s1_country(paths: Paths, subset: str) -> pl.DataFrame:
    n = pl.read_parquet(paths.norm / f"{subset}.parquet", columns=["rid", "source", "country"])
    return n.filter(pl.col("source") == 1).select(pl.col("rid").alias("s1_rid"), "country")


def final_scores(paths: Paths, subset: str, tag: str, blend: Path | None = None,
                 name: str = "final") -> pl.DataFrame:
    """(s1_rid, other_rid, p_lgb, p): LightGBM and final stacker score, as in stack-apply.
    blend: optional parquet file (s1_rid, other_rid, p, px, pb) replacing the tag's own blend
    (read here so that no other copy stays in memory)."""
    out = paths.work / "trap" / f"{subset}_{name}{''.join(f'_{e}' for e in sorted(S.EXTRA))}.parquet"
    if out.exists():
        return pl.read_parquet(out)
    rep = json.loads((S._out(paths, tag) / "report.json").read_text())
    booster = lgb.Booster(model_file=str(S._out(paths, tag) / "stack.txt"))
    cols = json.loads((paths.work / "models" / "policy.json").read_text())["features"]
    with timed(f"[trap] {subset} competition features"):
        d = S.competition(S.blended(paths, subset, tag) if blend is None
                          else pl.read_parquet(blend))
        if "f1" in S.EXTRA and "pf" not in d.columns:
            pf = pl.read_parquet(paths.work / "xenc" / S.F1_TAG / f"{subset}_scored.parquet")
            d = d.join(pf.select("s1_rid", "other_rid", pl.col("px").alias("pf")),
                       on=["s1_rid", "other_rid"], how="left")
        all_pb = d.select("s1_rid", "other_rid", "p", "pb")
        act = d.filter(pl.col("pb") >= S.ACTIVE_MIN).select(
            "s1_rid", "other_rid", "p", "px", "pb", *S.COMP,
            *(["pf"] if "f1" in S.EXTRA else []))
        del d
        act = S.extra_features(paths, subset, act)
    scored = []
    for part in sorted((paths.work / "pred" / f"{subset}_wide").glob("*.parquet")):
        f = pl.read_parquet(part, columns=["s1_rid", "other_rid", *cols]).join(
            act, on=["s1_rid", "other_rid"], how="inner")
        if f.height:
            scored.append(f.select("s1_rid", "other_rid").with_columns(
                pl.Series("p2", booster.predict(M._matrix(f, rep["features"])))))
    p2 = pl.concat(scored)
    final = (all_pb.join(p2, on=["s1_rid", "other_rid"], how="left")
             .select("s1_rid", "other_rid", pl.col("p").alias("p_lgb"),
                     pl.coalesce("p2", "pb").alias("p")))
    out.parent.mkdir(parents=True, exist_ok=True)
    final.write_parquet(out)
    return final


def _population(paths: Paths, subset: str) -> pl.DataFrame:
    """Pairs with LightGBM p >= WIDE_MIN with cell, country and (if labeled) label."""
    if subset == "val":
        f = (pl.scan_parquet(paths.work / "features" / "val.parquet")
             .select("s1_rid", "other_rid", *CELL_COLS)
             .join(pl.scan_parquet(paths.work / "models" / "val_scored.parquet"),
                   on=["s1_rid", "other_rid"])
             .filter(pl.col("p") >= M.WIDE_MIN).collect())
    else:
        f = pl.scan_parquet(paths.work / "pred" / f"{subset}_wide" / "*.parquet").select(
            "s1_rid", "other_rid", *CELL_COLS).collect()
    f = f.with_columns(cell_expr()).select("s1_rid", "other_rid", "cell").join(
        _s1_country(paths, subset), on="s1_rid")
    if subset != "test":
        gt, _ = S.gt_rids(paths, subset)
        f = f.join(gt.rename({"s1_id": "s1_rid", "match_id": "other_rid"})
                   .with_columns(pl.lit(True).alias("label")),
                   on=["s1_rid", "other_rid"], how="left").with_columns(
            pl.col("label").fill_null(False))
    return f


def _density(pop: pl.DataFrame) -> dict:
    """Per cell: pairs, true matches and non-matches per S1 (S1s with any pair)."""
    n = pop["s1_rid"].n_unique()
    out = {"n_s1": n}
    for cell in ("same_name", "similar_name"):
        c = pop.filter(pl.col("cell") == cell)
        d = {"tot": c.height / n}
        if "label" in c.columns:
            d["pos"] = c["label"].sum() / n
            d["neg"] = d["tot"] - d["pos"]
        out[cell] = d
    return out


def _ref_density(pops: dict, refs: list[tuple[str, str]]) -> dict:
    parts = [pops[s].filter(pl.col("country") == c) for s, c in refs]
    return _density(pl.concat(parts))


def _shift(test_d: dict, ref_d: dict) -> dict:
    out = {}
    for cell in ("same_name", "similar_name"):
        t, r = test_d[cell], ref_d[cell]
        neg_test = max(t["tot"] - r["pos"], r["neg"])
        ratio = neg_test / r["neg"] if r["neg"] > 0 else 1.0
        out[cell] = {"test_tot": t["tot"], "ref_tot": r["tot"], "ref_pos": r["pos"],
                     "ref_neg": r["neg"], "test_neg": neg_test, "ratio": ratio,
                     "shift": math.log(max(ratio, 1.0))}
    return out


def _em_prior(p: np.ndarray, prior_ref: float, iters: int = 100) -> float:
    """Saerens EM: test prior of a cell given calibrated probabilities and the reference prior."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pi = prior_ref
    for _ in range(iters):
        a = p * pi / prior_ref
        b = (1 - p) * (1 - pi) / (1 - prior_ref)
        pi = float(np.mean(a / (a + b)))
    return pi


def correct(final: pl.DataFrame, cells: pl.DataFrame, shifts: dict) -> pl.DataFrame:
    """Apply the per (country, cell) logit shift to final p."""
    tab = pl.DataFrame([{"country": c, "cell": k, "shift": v["shift"]}
                        for c, sh in shifts.items() for k, v in sh.items()])
    d = (final.join(cells, on=["s1_rid", "other_rid"], how="left")
         .join(tab, on=["country", "cell"], how="left")
         .with_columns(pl.col("shift").fill_null(0.0)))
    p = pl.col("p").clip(1e-6, 1 - 1e-6)
    logit = (p / (1 - p)).log() - pl.col("shift")
    return d.with_columns((1 / (1 + (-logit).exp())).alias("p")).select(
        "s1_rid", "other_rid", "p_lgb", "p")


def _f05(pred: pl.DataFrame, gt, s1) -> float:
    return S.f05(pred, gt, s1)


def run(paths: Paths, tag: str) -> None:
    rep = json.loads((S._out(paths, tag) / "report.json").read_text())
    t = rep["t_stack"]
    report: dict = {"tag": tag, "threshold": t}

    with timed("[trap] populations"):
        pops = {s: _population(paths, s) for s in ("val", "val2", "test")}

    # ---- test shifts
    shifts, dens = {}, {}
    for country, refs in REF.items():
        td = _density(pops["test"].filter(pl.col("country") == country))
        rd = _ref_density(pops, refs)
        shifts[country] = _shift(td, rd)
        dens[country] = {"test": td, "ref": rd}
        for cell, v in shifts[country].items():
            log.info("[trap] %-6s %-12s test %.3f/S1 | ref %.3f (pos %.3f, neg %.3f) | "
                     "test neg %.3f | ratio %.2f | shift %.2f", country, cell, v["test_tot"],
                     v["ref_tot"], v["ref_pos"], v["ref_neg"], v["test_neg"], v["ratio"],
                     v["shift"])
    report["shifts"] = shifts

    # ---- test: correct and decide
    with timed("[trap] test final scores"):
        final = final_scores(paths, "test", tag)
    cells_t = pops["test"].select("s1_rid", "other_rid", "cell", "country")
    s1c = _s1_country(paths, "test")
    fin_cells = final.join(cells_t, on=["s1_rid", "other_rid"], how="inner")
    # Pairs used for the cutoff curve, then only pairs that can still become matches
    # (the correction only lowers p, and exclusivity keeps the best S1 per record, so
    # pairs below DECIDE_MIN never change a decision at threshold t).
    test_counts = {c: final.filter(pl.col("p_lgb") >= c).height for c in CUTOFFS}
    test_counts[None] = final.height
    final = final.filter(pl.col("p") >= DECIDE_MIN)
    base = M.decide_threshold(final, t)
    corr = correct(final.join(s1c, on="s1_rid"), cells_t.drop("country"), shifts)
    new = M.decide_threshold(corr, t)

    # EM cross-check on test cells (final p), reference prior from the reference cell
    em = {}
    for country, refs in REF.items():
        rd = dens[country]["ref"]
        for cell in ("same_name", "similar_name"):
            p = fin_cells.filter((pl.col("country") == country) & (pl.col("cell") == cell))["p"]
            prior_ref = rd[cell]["pos"] / rd[cell]["tot"] if rd[cell]["tot"] else 0.5
            prior_density = rd[cell]["pos"] / max(dens[country]["test"][cell]["tot"], 1e-9)
            em[f"{country}/{cell}"] = {"prior_ref": prior_ref,
                                       "prior_test_density": min(prior_density, 1.0),
                                       "prior_test_em": _em_prior(p.to_numpy(), prior_ref),
                                       "pairs": p.len()}
            log.info("[trap] EM %-6s %-12s prior ref %.3f | test by density %.3f | test by EM %.3f",
                     country, cell, prior_ref, em[f"{country}/{cell}"]["prior_test_density"],
                     em[f"{country}/{cell}"]["prior_test_em"])
    report["em"] = em
    del fin_cells, pops["test"]

    by_c = (base.join(s1c, on="s1_rid").group_by("country").len("before")
            .join(new.join(s1c, on="s1_rid").group_by("country").len("after"), on="country"))
    dropped = base.join(new, on=["s1_rid", "other_rid"], how="anti").height
    added = new.join(base, on=["s1_rid", "other_rid"], how="anti").height
    log.info("[trap] test matches %d -> %d (dropped %d, added %d) | by country %s",
             base.height, new.height, dropped, added, by_c.rows())
    report["test"] = {"before": base.height, "after": new.height, "dropped": dropped,
                      "added": added, "by_country": by_c.rows()}
    new.write_parquet(paths.work / "trap" / "test_matches.parquet")

    # ---- val2 safety checks
    with timed("[trap] val2 final scores"):
        f2 = final_scores(paths, "val2", tag)
    gt2, s12 = S.gt_rids(paths, "val2")
    c2 = pops["val2"].select("s1_rid", "other_rid", "cell")
    s1c2 = _s1_country(paths, "val2")
    f2c = f2.join(s1c2, on="s1_rid")
    base2 = _f05(M.decide_threshold(f2, t), gt2, s12)
    # (a) honest check: val2 shift measured against val India (Tamil Nadu) as reference
    sh_a = {"India": _shift(_density(pops["val2"].filter(pl.col("country") == "India")),
                            _ref_density(pops, [("val", "India")]))}
    fa = _f05(M.decide_threshold(correct(f2c, c2, sh_a), t), gt2, s12)
    # (b) downside: the test India shift applied to val2 (cost if the traps were not real)
    fb = _f05(M.decide_threshold(correct(f2c, c2, {"India": shifts["India"]}), t), gt2, s12)
    log.info("[trap] val2 F0.5: baseline %.5f | own shift vs Tamil Nadu %.5f (shifts %s) | "
             "test India shift %.5f", base2, fa,
             {k: round(v["shift"], 2) for k, v in sh_a["India"].items()}, fb)
    report["val2"] = {"baseline": base2, "own_shift": fa,
                      "own_shift_values": {k: v["shift"] for k, v in sh_a["India"].items()},
                      "test_india_shift": fb}

    # ---- candidate cutoff curve
    report["cutoff_curve"] = cutoff_curve(paths, final, test_counts, f2, gt2, s12, t)
    (paths.work / "trap" / "report.json").write_text(json.dumps(report, indent=1, default=str))


def cutoff_curve(paths: Paths, final_test: pl.DataFrame, test_counts: dict,
                 final_val2: pl.DataFrame, gt2, s12, t: float) -> list[dict]:
    """final_test: test pairs with final p >= DECIDE_MIN; test_counts: pairs per cutoff."""
    rows = []
    m2 = M.decide_threshold(final_val2, t)
    m2p = m2.join(final_val2, on=["s1_rid", "other_rid"])
    mt = M.decide_threshold(final_test, t).join(final_test, on=["s1_rid", "other_rid"])
    log.info("[trap] lowest LightGBM p among final matches: val2 %.5f | test %.5f",
             m2p["p_lgb"].min(), mt["p_lgb"].min())
    labeled = {}
    for subset in ("val", "val2", "rest"):
        if subset == "rest" and not (paths.work / "pred" / "rest_scored.parquet").exists():
            continue
        sc = S._scored(paths, subset)
        gt, s1 = S.gt_rids(paths, subset)
        gtp = gt.rename({"s1_id": "s1_rid", "match_id": "other_rid"})
        labeled[subset] = (sc.join(gtp.with_columns(pl.lit(True).alias("y")),
                                   on=["s1_rid", "other_rid"], how="left"),
                           gtp.height, s1.len())
    n_test = _s1_country(paths, "test").height
    for c in [None] + CUTOFFS:
        row = {"cutoff": c if c is not None else "all blocking (40)"}
        for subset, (sc, n_gt, n_s1) in labeled.items():
            k = sc if c is None else sc.filter(pl.col("p") >= c)
            row[f"{subset}_per_s1"] = k.height / n_s1
            row[f"{subset}_kept"] = k["y"].fill_null(False).sum() / n_gt
        k2 = m2p if c is None else m2p.filter(pl.col("p_lgb") >= c)
        row["val2_f05"] = _f05(k2.select("s1_rid", "other_rid"), gt2, s12)
        row["test_per_s1"] = test_counts[c] / n_test
        row["test_pairs"] = test_counts[c]
        row["test_matches_lost"] = 0 if c is None else mt.filter(pl.col("p_lgb") < c).height
        rows.append(row)
        log.info("[trap] cutoff %-18s | %s", row["cutoff"],
                 " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                            for k, v in row.items() if k != "cutoff"))
    return rows
