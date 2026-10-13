"""Final step: stacker, dense-retrieval rescue and per-S1 expected-F0.5 decision -> output files.

Runs after the pipeline stages in README.md (normalize, block, features, LightGBM, cross-encoders,
trap / refine) and after `biencoder.py retrieve` for val, val2, delhi and test.

1. Stacker: a LightGBM over the model scores (LightGBM p, small cross-encoder px, blend pb,
   rival-aware cross-encoder pf), score-competition features and collective S2/S3 features (how many
   of the S1's other confident candidates share the record's name skeleton, address core or house
   number). Trained on the three labelled holdouts (val, val2, delhi); settings chosen with
   leave-one-region-out over their 11 regions.
2. Dense-retrieval rescue: lexical blocking misses 1.0-1.5% of true pairs on the holdouts. Each
   S2/S3 record's nearest S1 from the fine-tuned bi-encoder (biencoder.py) recovers about 75% of
   those misses. Pairs only the bi-encoder proposed ("rescue pairs") are scored by a separate
   LightGBM (rescue model) trained on the holdouts, whose inputs use out-of-fold main scores.
3. Decision on test: stacker scores, trap correction, contested no-address rule, merge with rescue
   scores, per-S1 expected-F0.5 decision (ef_decision.py; one S1 per record), same-script links,
   source cap.
4. Output: matching_results.tsv; candidate_pairs.tsv = the lexical candidate file (pairs with
   LightGBM p >= 0.01, written by the submit stage) plus every rescue pair the rescue model scored.

Leave-one-region-out F0.5: val 0.99129 -> 0.99357, val2 0.99197 -> 0.99394, delhi 0.99036 -> 0.99302
(without -> with dense-retrieval rescue).

    python src/final.py --work-dir work --out output --candidates output/candidate_pairs.tsv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import ef_decision
import model as M
import refine as R
import stack as S
import submit as SU
import trap as T
from common import Paths, log, setup_logging, timed

SETS = ("val", "val2", "delhi")
K_RESCUE = 1            # a record's top-K bi-encoder S1s are rescue candidates
RESCUE_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
                 "min_data_in_leaf": 50, "lambda_l2": 5.0, "feature_fraction": 0.8,
                 "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
RESCUE_ROUNDS = 400
RESCUE = ["cos", "rank", "cos_top", "cos_gap", "top_margin", "s1_indeg", "s1_rank", "rec_main_p",
          "s1_main_n", "s1_main_psum", "name_norm_tset", "name_norm_ratio", "name_skel_tset",
          "name_skel_ratio", "addr_norm_tset", "addr_norm_ratio", "skel_eq", "house_shared",
          "nums_shared", "o_nnums", "state_eq", "o_addr_missing", "o_domain", "o_src"]

TAG = "e5s_big"
STACK_PARAMS = {**S.PARAMS, "num_leaves": 31, "max_depth": 6, "min_data_in_leaf": 100}
STACK_ROUNDS = 800
# Collective evidence (--sibling): how many of the S1's other confident candidates (blend
# pb >= SIB_ANCHOR) share this record's name skeleton, address core or house number.
# Leave-one-region-out: val 0.99118 -> 0.99129, val2 0.99192 -> 0.99197, delhi 0.99028 -> 0.99036.
SIB = ["sib_n", "sib_skel", "sib_addr", "sib_house", "sib_any"]
SIB_ANCHOR = 0.9


def sib_features(paths: Paths, subset: str, act: pl.DataFrame) -> pl.DataFrame:
    n = pl.read_parquet(paths.norm / f"{subset}.parquet",
                        columns=["rid", "name_skel", "addr_core", "addr_house"])
    n = n.select(pl.col("rid").alias("other_rid"), pl.col("name_skel").fill_null("").alias("k"),
                 pl.when(pl.col("addr_core").fill_null("").str.len_chars() >= 6)
                 .then(pl.col("addr_core")).alias("a"),
                 pl.col("addr_house").list.first().alias("h"))
    d = act.select("s1_rid", "other_rid", "pb").join(n, on="other_rid", how="left")
    anchor = pl.col("pb") >= SIB_ANCHOR
    anc = d.filter(anchor)
    out = d.select("s1_rid", "other_rid").join(anc.group_by("s1_rid").len("sib_n"), on="s1_rid",
                                               how="left")
    for col, name in (("k", "sib_skel"), ("a", "sib_addr"), ("h", "sib_house")):
        c = anc.filter(pl.col(col).is_not_null() & (pl.col(col) != "")).group_by("s1_rid", col).len(name)
        x = d.select("s1_rid", "other_rid", col, "pb").join(c, on=["s1_rid", col], how="left")
        # the pair itself is not counted as its own evidence
        x = x.with_columns((pl.col(name).fill_null(0) - anchor.cast(pl.UInt32)
                            * pl.col(col).is_not_null().cast(pl.UInt32)
                            * pl.col(name).is_not_null().cast(pl.UInt32)).clip(0, 50).alias(name))
        out = out.join(x.select("s1_rid", "other_rid", name), on=["s1_rid", "other_rid"], how="left")
    self_anchor = d.select("s1_rid", "other_rid", anchor.cast(pl.Int32).alias("_sa"))
    out = (out.join(self_anchor, on=["s1_rid", "other_rid"])
           .with_columns((pl.col("sib_n").fill_null(0) - pl.col("_sa")).clip(0, 50).alias("sib_n"))
           .drop("_sa")
           .with_columns((pl.col("sib_skel") + pl.col("sib_addr") + pl.col("sib_house")).alias("sib_any")))
    return act.join(out.with_columns([pl.col(c).cast(pl.Float32) for c in SIB]),
                    on=["s1_rid", "other_rid"], how="left")


def stack_scores(paths: Paths, subset: str, booster, cols: list[str], feats: list[str],
                 blend: Path, sibling: bool = False) -> pl.DataFrame:
    """trap.final_scores with an explicit booster and blend file, without the cache."""
    d = S.competition(pl.read_parquet(blend))
    if "pf" not in d.columns:
        pf = pl.read_parquet(paths.work / "xenc" / S.F1_TAG / f"{subset}_scored.parquet")
        d = d.join(pf.select("s1_rid", "other_rid", pl.col("px").alias("pf")),
                   on=["s1_rid", "other_rid"], how="left")
    all_pb = d.select("s1_rid", "other_rid", "p", "pb")
    act = d.filter(pl.col("pb") >= S.ACTIVE_MIN).select("s1_rid", "other_rid", "p", "px", "pb",
                                                          *S.COMP, "pf")
    del d
    act = S.extra_features(paths, subset, act)
    if sibling:
        act = sib_features(paths, subset, act)
    scored = []
    for part in sorted((paths.work / "pred" / f"{subset}_wide").glob("*.parquet")):
        f = pl.read_parquet(part, columns=["s1_rid", "other_rid", *cols]).join(
            act, on=["s1_rid", "other_rid"], how="inner")
        if f.height:
            scored.append(f.select("s1_rid", "other_rid").with_columns(
                pl.Series("p2", booster.predict(M._matrix(f, feats)))))
    p2 = pl.concat(scored)
    return (all_pb.join(p2, on=["s1_rid", "other_rid"], how="left")
            .select("s1_rid", "other_rid", pl.col("p").alias("p_lgb"),
                    pl.coalesce("p2", "pb").alias("p")))


# ----------------------------------------------------------------------------- main scores
def _region(paths: Paths, sub: str, act: pl.DataFrame) -> pl.DataFrame:
    st = (pl.read_parquet(paths.norm / f"{sub}.parquet", columns=["rid", "addr_state", "source"])
          .filter(pl.col("source") == 1).select(pl.col("rid").alias("s1_rid"), pl.col("addr_state").alias("st")))
    act = act.join(st, on="s1_rid", how="left")
    if sub == "delhi":
        return act.with_columns(pl.lit("delhi").alias("reg"))
    return act.with_columns((pl.lit(sub + ":") + pl.col("st").fill_null("?")).alias("reg"))


def holdout_oof(paths: Paths, cols: list[str], feats: list[str], cache: Path) -> dict[str, pl.DataFrame]:
    """Out-of-fold main final scores (s1_rid, other_rid, p) for val, val2, delhi."""
    files = {s: cache / f"oof_{s}.parquet" for s in SETS}
    if all(f.exists() for f in files.values()):
        return {s: pl.read_parquet(f) for s, f in files.items()}
    D, xs, ys, rg = {}, [], [], []
    for sub in SETS:
        all_pb, act = S.active_frame(paths, sub, TAG, cols)
        act = sib_features(paths, sub, act)
        gt, _ = S.gt_rids(paths, sub)
        lab = gt.rename({"s1_id": "s1_rid", "match_id": "other_rid"}).with_columns(pl.lit(True).alias("label"))
        act = _region(paths, sub, act.join(lab, on=["s1_rid", "other_rid"], how="left")
                      .with_columns(pl.col("label").fill_null(False)))
        D[sub] = (all_pb, act.select("s1_rid", "other_rid"))
        xs.append(M._matrix(act, feats)); ys.append(act["label"].to_numpy()); rg.append(act["reg"].to_numpy())
    X, y, rg = np.concatenate(xs), np.concatenate(ys), np.concatenate(rg)
    oof = np.zeros(len(y))
    with timed("[final] leave-one-region-out main scores"):
        for r in set(rg):
            te = rg == r
            oof[te] = lgb.train(STACK_PARAMS, lgb.Dataset(X[~te], y[~te]), STACK_ROUNDS).predict(X[te])
    out, o = {}, 0
    for sub in SETS:
        all_pb, keys = D[sub]
        out[sub] = S._final_scores(all_pb, keys, oof[o:o + keys.height])
        o += keys.height
        out[sub].write_parquet(files[sub])
    return out


# ----------------------------------------------------------------------------- rescue pairs
def _pair_strings(paths: Paths, sub: str, pr: pl.DataFrame) -> pl.DataFrame:
    cols = ["rid", "name_norm", "name_skel", "addr_norm", "addr_house", "addr_numbers", "addr_state",
            "addr_missing", "name_domain", "source"]
    n = pl.read_parquet(paths.norm / f"{sub}.parquet", columns=cols)
    need = pl.concat([pr.select(pl.col("s1_rid").alias("rid")), pr.select(pl.col("other_rid").alias("rid"))]).unique()
    n = n.join(need, on="rid", how="semi")
    a = n.select([pl.col(c).alias("s_" + c) for c in cols]).rename({"s_rid": "s1_rid"})
    b = n.select([pl.col(c).alias("o_" + c) for c in cols]).rename({"o_rid": "other_rid"})
    x = pr.join(a, on="s1_rid", how="left").join(b, on="other_rid", how="left")
    f = {}
    for col in ("name_norm", "name_skel", "addr_norm"):
        s1, o = x["s_" + col].fill_null("").to_list(), x["o_" + col].fill_null("").to_list()
        f[col + "_tset"] = cpdist(s1, o, scorer=fuzz.token_set_ratio, workers=8)
        f[col + "_ratio"] = cpdist(s1, o, scorer=fuzz.ratio, workers=8)
    x = x.with_columns([pl.Series(k, v.astype(np.float32)) for k, v in f.items()])
    x = x.with_columns(
        (pl.col("s_name_skel") == pl.col("o_name_skel")).fill_null(False).cast(pl.Int8).alias("skel_eq"),
        pl.col("s_addr_house").list.set_intersection(pl.col("o_addr_house")).list.len().fill_null(0).alias("house_shared"),
        pl.col("s_addr_numbers").list.set_intersection(pl.col("o_addr_numbers")).list.len().fill_null(0).alias("nums_shared"),
        pl.col("o_addr_numbers").list.len().fill_null(0).alias("o_nnums"),
        (pl.col("s_addr_state") == pl.col("o_addr_state")).cast(pl.Int8).fill_null(-1).alias("state_eq"),
        pl.col("o_addr_missing").cast(pl.Int8).alias("o_addr_missing"),
        pl.col("o_name_domain").is_not_null().cast(pl.Int8).alias("o_domain"),
        pl.col("o_source").alias("o_src"))
    return x.select("s1_rid", "other_rid", *f.keys(), "skel_eq", "house_shared", "nums_shared", "o_nnums",
                    "state_eq", "o_addr_missing", "o_domain", "o_src")


def rescue_pairs(paths: Paths, sub: str, main: pl.DataFrame) -> pl.DataFrame:
    """Bi-encoder top-K pairs that lexical blocking did not propose, with rescue features.
    main: (s1_rid, other_rid, p) final scores of the main pipeline for this subset."""
    knn = pl.read_parquet(paths.work / "bienc" / f"{sub}_knn.parquet")
    knn = knn.with_columns(pl.col("cos").max().over("other_rid").alias("cos_top"),
                           pl.col("cos").sort(descending=True).get(1).over("other_rid").alias("cos_2nd"))
    knn = knn.with_columns((pl.col("cos") - pl.col("cos_top")).alias("cos_gap"),
                           (pl.col("cos_top") - pl.col("cos_2nd")).alias("top_margin"),
                           pl.len().over("s1_rid").alias("s1_indeg"),
                           pl.col("cos").rank(descending=True).over("s1_rid").alias("s1_rank")).drop("cos_2nd")
    knn = knn.filter(pl.col("rank") < K_RESCUE)
    for f in sorted((paths.work / "block" / sub).glob("*.parquet")):
        knn = knn.join(pl.read_parquet(f, columns=["s1_rid", "other_rid"]), on=["s1_rid", "other_rid"], how="anti")
    mrec = main.group_by("other_rid").agg(pl.col("p").max().alias("rec_main_p"))
    ms1 = (main.filter(pl.col("p") >= 0.5).group_by("s1_rid")
           .agg(pl.len().alias("s1_main_n"), pl.col("p").sum().alias("s1_main_psum")))
    r = (knn.join(mrec, on="other_rid", how="left").join(ms1, on="s1_rid", how="left")
         .with_columns(pl.col("rec_main_p").fill_null(0.0), pl.col("s1_main_n").fill_null(0),
                       pl.col("s1_main_psum").fill_null(0.0)))
    return r.join(_pair_strings(paths, sub, r.select("s1_rid", "other_rid")), on=["s1_rid", "other_rid"])


def train_rescue(paths: Paths, oof: dict[str, pl.DataFrame]) -> lgb.Booster:
    xs, ys = [], []
    for sub in SETS:
        r = rescue_pairs(paths, sub, oof[sub])
        gt, _ = S.gt_rids(paths, sub)
        lab = gt.rename({"s1_id": "s1_rid", "match_id": "other_rid"}).with_columns(pl.lit(True).alias("y"))
        r = r.join(lab, on=["s1_rid", "other_rid"], how="left").with_columns(pl.col("y").fill_null(False))
        log.info("[final] %s rescue pairs %d, true %d", sub, r.height, r["y"].sum())
        xs.append(r.select(RESCUE).to_numpy().astype(np.float32)); ys.append(r["y"].to_numpy())
    return lgb.train(RESCUE_PARAMS, lgb.Dataset(np.concatenate(xs), np.concatenate(ys)), RESCUE_ROUNDS)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--data-dir", default="../student_resource/dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", required=True, help="lexical candidate_pairs.tsv written by the submit stage")
    a = ap.parse_args()
    setup_logging()
    paths = Paths(a.data_dir, a.work_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = paths.work / "final"
    cache.mkdir(exist_ok=True)
    S.EXTRA = {"f1"}
    cols = json.loads((paths.work / "models" / "policy.json").read_text())["features"]
    feats = cols + ["p", "px", "pb"] + S.COMP + S.extra_cols() + SIB

    # 1-2. rescue model from the holdouts (main scores out of fold)
    oof = holdout_oof(paths, cols, feats, cache)
    with timed("[final] rescue model"):
        rescue = train_rescue(paths, oof)
    rescue.save_model(str(out / "rescue_model.txt"))
    del oof

    # 3a. final stacker (9c recipe) and test main scores
    xs, ys = [], []
    for sub in SETS:
        _, act = S.active_frame(paths, sub, TAG, cols)
        act = sib_features(paths, sub, act)
        gt, _ = S.gt_rids(paths, sub)
        lab = gt.rename({"s1_id": "s1_rid", "match_id": "other_rid"}).with_columns(pl.lit(True).alias("label"))
        act = act.join(lab, on=["s1_rid", "other_rid"], how="left").with_columns(pl.col("label").fill_null(False))
        xs.append(M._matrix(act, feats)); ys.append(act["label"].to_numpy())
    with timed("[final] stacker on val + val2 + delhi"):
        booster = lgb.train(STACK_PARAMS, lgb.Dataset(np.concatenate(xs), np.concatenate(ys)), STACK_ROUNDS)
    del xs, ys
    booster.save_model(str(out / "stacker.txt"))
    with timed("[final] test stacker scores"):
        ft = stack_scores(paths, "test", booster, cols, feats,
                             paths.work / "trap" / "test_switched_blend.parquet", True)

    # 3b. test rescue pairs, scored
    with timed("[final] test rescue pairs"):
        rt = rescue_pairs(paths, "test", ft.select("s1_rid", "other_rid", "p"))
        rt = rt.select("s1_rid", "other_rid").with_columns(
            pl.Series("p", rescue.predict(rt.select(RESCUE).to_numpy().astype(np.float32))))
    rt.write_parquet(cache / "test_rescue.parquet")
    log.info("[final] test rescue pairs %d (p >= 0.5: %d)", rt.height, (rt["p"] >= 0.5).sum())

    # 3c. 9c's decision chain with the rescue scores merged in
    ft = ft.filter(pl.col("p") >= T.DECIDE_MIN)
    shifts = json.loads((paths.work / "trap" / "report.json").read_text())["shifts"]
    cells = T._population(paths, "test").select("s1_rid", "other_rid", "cell")
    ft = T.correct(ft.join(T._s1_country(paths, "test"), on="s1_rid"), cells, shifts)
    ft = R.contested_noaddr(paths, "test", ft)
    sc = pl.concat([ft.select("s1_rid", "other_rid", "p"),
                    rt.filter(pl.col("p") >= T.DECIDE_MIN).cast({"s1_rid": pl.Int32, "other_rid": pl.Int32,
                                                                  "p": ft.schema["p"]})])
    mt = ef_decision.decide(sc)
    add = R.same_script_links(paths, "test", mt)
    final = R.cap_sources(paths, "test", pl.concat([mt, add.cast(mt.schema)]).unique(), sc)
    log.info("[final] test matches: %d (links +%d, from rescue %d)", final.height, add.height,
             final.join(rt, on=["s1_rid", "other_rid"], how="semi").height)

    # 4. files: candidates = 9c candidate file + every scored rescue pair
    norm = pl.read_parquet(paths.norm / "test.parquet", columns=["rid", "entity_id", "source"])
    s1 = norm.filter(pl.col("source") == 1).select(pl.col("rid").alias("s1_rid"), "entity_id")
    ids = norm.select("rid", "entity_id")
    cand = (pl.read_csv(a.candidates, separator="\t", quote_char=None, infer_schema=False)
            .with_columns(pl.col("candidate_entity_ids").fill_null("").str.split(","))
            .explode("candidate_entity_ids").filter(pl.col("candidate_entity_ids").fill_null("") != "")
            .join(ids.rename({"rid": "s1_rid", "entity_id": "source1_entity_id"}), on="source1_entity_id")
            .join(ids.rename({"rid": "other_rid", "entity_id": "candidate_entity_ids"}), on="candidate_entity_ids")
            .select("s1_rid", "other_rid"))
    cand = pl.concat([cand.cast({"s1_rid": pl.Int32, "other_rid": pl.Int32}),
                      rt.select("s1_rid", "other_rid").cast({"s1_rid": pl.Int32, "other_rid": pl.Int32})]).unique()
    n0 = final.height
    final = final.join(cand.cast(final.schema), on=["s1_rid", "other_rid"], how="semi")
    log.info("[final] %d matches outside the candidate file dropped; candidates %d", n0 - final.height, cand.height)
    SU._write_tsv(SU._id_lists(final, norm, s1, "matched_entity_ids"), out / "matching_results.tsv")
    SU._write_tsv(SU._id_lists(cand, norm, s1, "candidate_entity_ids"), out / "candidate_pairs.tsv")
    final.write_parquet(out / "matches_rid.parquet")
    log.info("[final] wrote %s", out)


if __name__ == "__main__":
    main()
