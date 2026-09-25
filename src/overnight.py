"""Unattended run of the submission 4 plan (docs/plan_submission_4.md), safe mode.

Every step runs after the previous one succeeded; thread caps and a memory guard protect
the machine; checkpoints stop the run; an automatic decision decides whether submission 4
files are built. Nothing is uploaded or pushed. Log: work/overnight.log

python src/overnight.py [--from-step N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "work"
LOG = WORK / "overnight.log"
PY = sys.executable
TAG = "e5s_big"
BASELINE_TAG = "e5s_2ep"          # submission 3 transformer
BEST_SO_FAR = 0.9857              # best val F0.5 so far (4-epoch transformer blend)
SUBMIT_AT = BEST_SO_FAR + 0.001   # 0.9867
FLOOR = 0.9850                    # submission 3 val score
MIN_FREE_GB = 3.0
ENV = {**os.environ, "POLARS_MAX_THREADS": "10", "OMP_NUM_THREADS": "10", "BER_THREADS": "10",
       "BER_WORKERS": "10", "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1",
       "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}


def log(msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class Stop(Exception):
    pass


def wait_for_memory() -> None:
    for _ in range(40):                       # up to 20 minutes
        free = psutil.virtual_memory().available / 1e9
        if free >= MIN_FREE_GB:
            return
        log(f"memory guard: {free:.1f} GB free, waiting")
        time.sleep(30)
    log("memory guard: still low after 20 min, continuing")


def run(stage: str, *args: str) -> None:
    wait_for_memory()
    cmd = [PY, str(REPO / "src" / "run_pipeline.py"), "--stage", stage, *args]
    log("RUN " + " ".join(cmd[2:]))
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=REPO, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f"  exit {rc} after {(time.time() - t0) / 60:.1f} min")
    if rc != 0:
        raise Stop(f"{stage} {' '.join(args)} failed (exit {rc})")


def check(ok: bool, what: str) -> None:
    log(("  CHECK OK   " if ok else "  CHECK FAIL ") + what)
    if not ok:
        raise Stop("checkpoint failed: " + what)


# --------------------------------------------------------------------------- checks

def check_holdouts() -> None:
    import polars as pl
    ids = {s: set(pl.read_parquet(WORK / "slices" / f"{s}_records.parquet", columns=["rid"])["rid"])
           for s in ("tune", "val")}
    rest = pl.read_parquet(WORK / "norm" / "rest.parquet", columns=["rid", "source"])
    val2 = pl.read_parquet(WORK / "norm" / "val2.parquet", columns=["rid", "source"])
    r, v = set(rest["rid"]), set(val2["rid"])
    n_rest, n_val2 = rest.filter(pl.col("source") == 1).height, val2.filter(pl.col("source") == 1).height
    log(f"  rest S1 {n_rest}, val2 S1 {n_val2}")
    check(1_200_000 <= n_rest <= 1_600_000, "rest S1 count plausible")
    check(80_000 <= n_val2 <= 150_000, "val2 S1 count plausible")
    check(not (r & v) and not (r & ids["tune"]) and not (r & ids["val"])
          and not (v & ids["tune"]) and not (v & ids["val"]), "no overlap among tune/val/val2/rest")


def check_rest_band() -> None:
    import polars as pl
    import stack as S
    band = pl.read_parquet(WORK / "pred" / "rest_band_feats.parquet", columns=["s1_rid", "other_rid"])
    gt, _ = S.gt_rids(_paths(), "rest")
    pos = band.join(gt.rename({"s1_id": "s1_rid", "match_id": "other_rid"}),
                    on=["s1_rid", "other_rid"], how="semi").height / max(band.height, 1)
    log(f"  rest uncertain pairs {band.height}, true-match share {pos:.1%}")
    check(300_000 <= band.height <= 1_200_000, "rest uncertain pair count plausible")
    check(0.20 <= pos <= 0.60, "rest true-match share plausible")


def _paths():
    sys.path.insert(0, str(REPO / "src"))
    from common import Paths
    return Paths(REPO.parent / "student_resource" / "dataset", WORK)


def last_auc() -> float | None:
    aucs = re.findall(rf"\[{TAG}\] epoch \d+ .*?val band AUC ([0-9.]+)", LOG.read_text(encoding="utf-8"))
    return float(aucs[-1]) if aucs else None


def check_submission(out_dir: Path) -> None:
    import polars as pl
    data_dir = REPO.parent / "student_resource" / "dataset" / "test"
    ids = {s: set(pl.read_csv(data_dir / f"test_source{s}.tsv", separator="\t", quote_char=None,
                              infer_schema=False, columns=["entity_id"])["entity_id"])
           for s in (1, 2, 3)}
    raw = (out_dir / "matching_results.tsv").read_bytes()
    check(raw.count(b"\r") == 0 and all(b < 128 for b in raw), "ASCII only, no carriage returns")
    lines = raw.decode("ascii").split("\n")
    check(lines[0] == "source1_entity_id\tmatched_entity_ids", "header exact")
    body = lines[1:-1]
    pat = re.compile(r"^S1-\d+\t((S[23]-\d+)(,S[23]-\d+)*)?$")
    check(all(pat.match(x) for x in body), "every row well formed")
    s1 = [x.split("\t")[0] for x in body]
    matched = [m for x in body for m in x.split("\t")[1].split(",") if m]
    check(set(s1) == ids[1] and len(s1) == len(ids[1]), "exactly one row per test S1")
    check(len(matched) == len(set(matched)), "no S2/S3 id used twice")
    check(set(matched) <= (ids[2] | ids[3]), "all matched ids are test S2/S3 ids")
    log(f"  rows {len(body)}, matched ids {len(matched)}")


def build_package(out_dir: Path, decision: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "output" / "matching_results.tsv", out_dir / "matching_results.tsv")
    files = sorted((REPO / "src").glob("*.py")) + [REPO / "configs" / "slices.json",
                                                   REPO / "README.md", REPO / "requirements.txt"]
    with zipfile.ZipFile(out_dir / "business_entity_resolution_code.zip", "w",
                         zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, "business_entity_resolution/" + f.relative_to(REPO).as_posix())
    lines = ["# Submission 4 (automatic run summary)", "",
             f"Built: {dt.datetime.now():%Y-%m-%d %H:%M}", ""]
    lines += [f"- {k}: {v}" for k, v in decision.items()]
    (out_dir / "RUN_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"  package written to {out_dir}")


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-step", type=int, default=1)
    start = ap.parse_args().from_step
    log("=" * 70)
    log(f"overnight run start (from step {start})")
    try:
        if start <= 1:
            log("STEP 1: build rest and val2")
            run("normalize", "--subset", "rest")
            check_holdouts()
        if start <= 2:
            log("STEP 2: blocking for rest and val2")
            run("block", "--subset", "rest")
            run("block", "--subset", "val2")
        if start <= 3:
            log("STEP 3: score rest and val2 with the current LightGBM")
            run("predict", "--subset", "rest")
            check_rest_band()
            run("predict", "--subset", "val2")
            log("STEP 3b: per-state report (informational)")
            run("holdout-report")
        if start <= 4:
            log("STEP 4: train the transformer on tune + rest")
            run("xenc-train", "--xenc-tag", TAG, "--xenc-epochs", "3", "--xenc-use-rest",
                "--xenc-addr-placeholder")
            auc = last_auc()
            log(f"  final val band AUC {auc} (best so far 0.9821)")
        if start <= 5:
            log("STEP 5: rescore test, saving the wider feature set")
            run("predict", "--subset", "test")
            check(any((WORK / "pred" / "test_wide").glob("*.parquet")), "test wide features saved")
        if start <= 6:
            log("STEP 6: transformer blend on val, scores on val2")
            run("xenc-blend", "--xenc-tag", TAG)
            run("xenc-score", "--subset", "val2", "--xenc-tag", BASELINE_TAG)
            run("xenc-score", "--subset", "val2", "--xenc-tag", TAG)
            run("val2-check", "--xenc-tag", TAG)
        if start <= 7:
            log("STEP 7: stacker")
            run("stack", "--xenc-tag", TAG)

        log("STEP 8: automatic decision")
        rep = json.loads((WORK / "stack" / TAG / "report.json").read_text())
        v2 = json.loads((WORK / "val2_check.json").read_text())
        use_stack = rep["val_stack_loso"] > rep["val_blend"]
        method = "stacker" if use_stack else "blend"
        val_f = rep["val_stack_loso"] if use_stack else rep["val_blend"]
        val2_f = rep["val2_stack"] if use_stack else rep["val2_blend"]
        val2_base = v2[BASELINE_TAG]
        decision = {"method": method, "val_f05": round(val_f, 5), "val2_f05": round(val2_f, 5),
                    "val2_submission3_method": round(val2_base, 5),
                    "val_blend": round(rep["val_blend"], 5),
                    "val_stack_loso": round(rep["val_stack_loso"], 5),
                    "val2_lightgbm_only": round(v2["lightgbm"], 5)}
        log(f"  {decision}")
        if val2_f < val2_base:
            decision["result"] = "STOP: val2 guard failed (worse than submission 3 method on val2)"
        elif val_f < FLOOR:
            decision["result"] = "STOP: val below submission 3 (0.9850)"
        elif val_f < SUBMIT_AT:
            decision["result"] = "BUILD, small gain (val between 0.9850 and 0.9867): decide together"
        else:
            decision["result"] = "BUILD: val at or above 0.9867"
        log("  DECISION: " + decision["result"])
        (WORK / "decision.json").write_text(json.dumps(decision, indent=1))
        if decision["result"].startswith("STOP"):
            log("run finished without building submission 4 (submission 3 stays best)")
            return

        log("STEP 9: transformer on test")
        run("xenc-score", "--subset", "test", "--xenc-tag", TAG)
        log("STEP 10: final scores on test")
        if use_stack:
            run("stack-apply", "--xenc-tag", TAG)
        else:
            run("xenc-apply", "--subset", "test", "--xenc-tag", TAG)
        log("STEP 11: files and official validator")
        run("submit")
        text = LOG.read_text(encoding="utf-8")
        check("PASS" in text[text.rfind("STEP 11"):], "validator printed PASS")
        log("STEP 12: checks and filing")
        check_submission(REPO / "output")
        out_dir = REPO.parent / "submission" / "submission_4"
        build_package(out_dir, decision)
        check_submission(out_dir)
        log("ALL DONE: submission 4 files ready in " + str(out_dir) + " (not uploaded, not pushed)")
    except Stop as e:
        log("STOPPED: " + str(e))
    except Exception as e:  # noqa: BLE001 - log anything unexpected and stop cleanly
        log(f"STOPPED (unexpected error): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
