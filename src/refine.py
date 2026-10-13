"""Stage `refine`: script-aware transformer and same-script sibling links, on top of the stacker.

1. Script-aware transformer. The main transformer (`e5s_big`) is better than the earlier one
   (`e5s_2ep`) everywhere except on names written in Bengali script (val2 West Bengal 0.924
   vs 0.959). Pairs whose S2/S3 name contains Bengali script use the earlier transformer's
   score; everything else is unchanged. Then the usual stacker and threshold.
2. Trap correction (trap.py): the per country / cell shifts measured by the `trap` stage.
3. Same-script sibling links. If a matched S2/S3 record's name is in an Indian script, an
   unmatched record with the identical native name AND identical address numbers joins the
   same S1 (only when a single S1 owns that native name). Identical native names alone are
   shared by many companies; with the address numbers too, the link is right about 99% of
   the time on val2 and rest.

Outputs in work/trap/: <subset>_refined.parquet (final scores), refine_report.json,
test_matches_refined.parquet.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

import model as M
import stack as S
import trap as T
import xenc
from common import Paths, log, switch, timed

BENGALI = r"[ঀ-৿]"
INDIAN_SCRIPTS = r"[ऀ-෿]"
K = ["s1_rid", "other_rid"]


def switched_blend(paths: Paths, subset: str, tag: str, alt_tag: str) -> Path:
    """Blend of LightGBM and the main transformer, with Bengali-script records scored by alt_tag.
    Written to work/trap/<subset>_switched_blend.parquet; returns the path."""
    w = json.loads((paths.work / "xenc" / tag / "blend.json").read_text())["w"]
    raw = xenc._raw_records(paths, subset)
    beng = raw.filter(pl.col("name").str.contains(BENGALI)).select(
        pl.col("rid").alias("other_rid"), pl.lit(True).alias("bn"))
    alt = S._px(paths, subset, alt_tag).rename({"px": "px_alt"})
    px = (S._px(paths, subset, tag).join(alt, on=K, how="left")
          .join(beng, on="other_rid", how="left")
          .with_columns(pl.when(pl.col("bn").fill_null(False) & pl.col("px_alt").is_not_null())
                        .then(pl.col("px_alt")).otherwise(pl.col("px")).alias("px"))
          .select(*K, "px"))
    log.info("[refine] %s: %d Bengali-script records; %d transformer scores switched", subset,
             beng.height, px.join(beng, on="other_rid").height)
    out = paths.work / "trap" / f"{subset}_switched_blend.parquet"
    (S._scored(paths, subset).join(px, on=K, how="left")
     .with_columns(pl.when(pl.col("px").is_not_null())
                   .then((1 - w) * pl.col("p") + w * pl.col("px"))
                   .otherwise(pl.col("p")).alias("pb"))
     .write_parquet(out))
    return out


def same_script_links(paths: Paths, subset: str, matches: pl.DataFrame) -> pl.DataFrame:
    """New (s1_rid, other_rid) pairs from identical native name + identical address numbers."""
    raw = xenc._raw_records(paths, subset).select("rid", "name")
    n = pl.read_parquet(paths.norm / f"{subset}.parquet",
                        columns=["rid", "source", "country", "addr_numbers"])
    nat = (n.filter(pl.col("source") != 1).join(raw, on="rid")
           .filter(pl.col("name").str.contains(INDIAN_SCRIPTS))
           .select("rid", "country",
                   pl.col("name").str.replace_all(r"\s+", " ").str.strip_chars().alias("key"),
                   pl.col("addr_numbers").list.sort().list.join(",").fill_null("").alias("nums")))
    owners = matches.join(nat.rename({"rid": "other_rid"}), on="other_rid").select(
        "s1_rid", "country", "key", "nums")
    single = owners.group_by("country", "key").agg(pl.col("s1_rid").n_unique().alias("n_owner"))
    owners = owners.join(single, on=["country", "key"]).filter(pl.col("n_owner") == 1)
    unmatched = nat.join(matches.select(pl.col("other_rid").alias("rid")).unique(), on="rid",
                         how="anti").filter(pl.col("nums") != "")
    return (owners.join(unmatched, on=["country", "key", "nums"])
            .select("s1_rid", pl.col("rid").alias("other_rid")).unique())


# Largest number of matches from one source for one S1 in the training ground truth.
SOURCE_CAP = {2: 5, 3: 6}


def cap_sources(paths: Paths, subset: str, matches: pl.DataFrame,
                scores: pl.DataFrame) -> pl.DataFrame:
    """Keep at most SOURCE_CAP[source] matches per S1 and source (highest scores)."""
    src = pl.read_parquet(paths.norm / f"{subset}.parquet", columns=["rid", "source"]).rename(
        {"rid": "other_rid"})
    d = (matches.join(scores.select(*K, "p"), on=K, how="left").join(src, on="other_rid")
         .sort(["s1_rid", "source", "p"], descending=[False, False, True], nulls_last=True)
         .with_columns(pl.int_range(pl.len()).over("s1_rid", "source").alias("_r")))
    cap = pl.col("source").replace_strict(SOURCE_CAP, default=99)
    return d.filter(pl.col("_r") < cap).select(K)


CONTESTED_T = 0.90


def contested_noaddr(paths: Paths, subset: str, scores: pl.DataFrame) -> pl.DataFrame:
    """A no-address S2/S3 record wanted by 2+ S1 (p >= 0.3) needs p >= CONTESTED_T.
    Measured: val +0.00006, val2 -0.00001, crowded Maharashtra +0.00116."""
    na = (pl.read_parquet(paths.norm / f"{subset}.parquet", columns=["rid", "source", "addr_missing"])
          .filter((pl.col("source") != 1) & pl.col("addr_missing")).select(pl.col("rid").alias("other_rid")))
    comp = scores.filter(pl.col("p") >= 0.3).group_by("other_rid").len("_n").filter(pl.col("_n") >= 2)
    hit = na.join(comp, on="other_rid").select("other_rid", pl.lit(True).alias("_c"))
    return (scores.join(hit, on="other_rid", how="left")
            .with_columns(pl.when(pl.col("_c").fill_null(False) & (pl.col("p") < CONTESTED_T))
                          .then(pl.lit(0.0)).otherwise(pl.col("p")).alias("p")).drop("_c"))


def _final(paths: Paths, subset: str, tag: str, alt_tag: str) -> pl.DataFrame:
    if switch("bengali_switch"):
        return T.final_scores(paths, subset, tag, switched_blend(paths, subset, tag, alt_tag),
                              name="refined")
    return T.final_scores(paths, subset, tag)


def run(paths: Paths, tag: str, alt_tag: str = "e5s_2ep", val2_only: bool = False) -> None:
    t = json.loads((S._out(paths, tag) / "report.json").read_text())["t_stack"]
    report: dict = {"tag": tag, "alt_tag": alt_tag, "threshold": t}

    # ---- val2: measure each step (independent holdout)
    gt2, s12 = S.gt_rids(paths, "val2")
    base2 = M.decide_threshold(T.final_scores(paths, "val2", tag), t)
    with timed("[refine] val2 final scores"):
        f2 = _final(paths, "val2", tag, alt_tag)
    f2d = f2.filter(pl.col("p") >= T.DECIDE_MIN)
    if switch("contested_noaddr"):
        f2d = contested_noaddr(paths, "val2", f2d)
    m2 = M.decide_threshold(f2d, t)
    add2 = (same_script_links(paths, "val2", m2) if switch("script_links")
            else pl.DataFrame(schema=m2.schema))
    fin2 = pl.concat([m2, add2]).unique()
    if switch("source_cap"):
        fin2 = cap_sources(paths, "val2", fin2, f2)
    report["val2"] = {"baseline": S.f05(base2, gt2, s12), "switch": S.f05(m2, gt2, s12),
                      "final": S.f05(fin2, gt2, s12), "links_added": add2.height}
    log.info("[refine] val2 F0.5: stacker %.5f | + Bengali switch %.5f | final (links, cap) "
             "%.5f (%d links)", report["val2"]["baseline"], report["val2"]["switch"],
             report["val2"]["final"], add2.height)
    if val2_only:
        (paths.work / "trap").mkdir(exist_ok=True)
        (paths.work / "trap" / "refine_report_val2.json").write_text(json.dumps(report, indent=1))
        return

    # ---- test
    with timed("[refine] test final scores"):
        ft = _final(paths, "test", tag, alt_tag)
    ft = ft.filter(pl.col("p") >= T.DECIDE_MIN)
    s1c = T._s1_country(paths, "test")
    if switch("trap_correction"):
        shifts = json.loads((paths.work / "trap" / "report.json").read_text())["shifts"]
        cells = T._population(paths, "test").select("s1_rid", "other_rid", "cell")
        ft = T.correct(ft.join(s1c, on="s1_rid"), cells, shifts)
    if switch("contested_noaddr"):
        ft = contested_noaddr(paths, "test", ft)
    mt = M.decide_threshold(ft, t)
    add = (same_script_links(paths, "test", mt) if switch("script_links")
           else pl.DataFrame(schema=mt.schema))
    final = pl.concat([mt, add]).unique()
    if switch("source_cap"):
        final = cap_sources(paths, "test", final, ft)
    report["test"] = {"after_threshold": mt.height, "links_added": add.height,
                      "final": final.height}
    log.info("[refine] test matches: after threshold %d | links +%d | final %d", mt.height,
             add.height, final.height)
    final.write_parquet(paths.work / "trap" / "test_matches_refined.parquet")
    (paths.work / "trap" / "refine_report.json").write_text(json.dumps(report, indent=1))
