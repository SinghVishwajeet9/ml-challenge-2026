"""Entry point: python src/run_pipeline.py --stage <name> [--data-dir ...] [--work-dir ...]

Stages (each reads the previous stage's Parquet output, so they can be rerun alone):
  convert  raw TSVs -> work/raw/*.parquet
  slices   region-based dev / tune / val slices of train -> work/slices/
  sample   small fixed sample of dev (+ test France) for rule development -> work/sample/
  normalize  name/address representations for --subset -> work/norm/<subset>.parquet
  block    candidate pairs for --subset (needs normalize) -> work/block/<subset>.parquet
  features pair + group features for --subset (needs block) -> work/features/<subset>.parquet
  train    LightGBM on features/<--train-subset>, decision policy chosen on val -> work/models/
  predict  score + decide for --subset -> work/pred/<subset>_matches.parquet
  submit   test TSVs -> <--out-dir>/matching_results.tsv, candidate_pairs.tsv (+ validator)
  xenc-oof / xenc-train / xenc-blend / xenc-score / xenc-apply
           hybrid cross-encoder for uncertain pairs (see xenc.py)
  holdout-report / val2-check / stack / stack-apply   stacker and held-out checks (stack.py)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import Paths, setup_logging

REPO = Path(__file__).resolve().parents[1]
STAGES = ["convert", "slices", "sample", "normalize", "block", "features", "train", "predict",
          "submit", "xenc-oof", "xenc-train", "xenc-blend", "xenc-score", "xenc-apply",
          "holdout-report", "val2-check", "stack", "stack-apply"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES + ["all"])
    ap.add_argument("--data-dir", default=str(REPO.parent / "student_resource" / "dataset"),
                    help="folder containing train/ and test/")
    ap.add_argument("--work-dir", default=str(REPO / "work"),
                    help="where intermediate Parquet files go")
    ap.add_argument("--subset", default="sample",
                    choices=["sample", "dev", "tune", "val", "train", "test", "rest", "val2"],
                    help="which records the normalize / block / features stages process")
    ap.add_argument("--neg-frac", type=float, default=None,
                    help="features: keep this fraction of non-matching pairs (training only)")
    ap.add_argument("--train-subset", default="tune", help="train: features to fit on")
    ap.add_argument("--out-dir", default=str(REPO / "output"), help="submit: output folder")
    ap.add_argument("--xenc-tag", default="e5s_2ep", help="xenc: variant folder name")
    ap.add_argument("--xenc-model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--xenc-epochs", type=int, default=2)
    ap.add_argument("--xenc-lora", action="store_true")
    ap.add_argument("--xenc-train-frac", type=float, default=1.0)
    ap.add_argument("--xenc-use-rest", action="store_true")
    ap.add_argument("--xenc-addr-placeholder", action="store_true")
    ap.add_argument("--validator", default=str(REPO.parent / "student_resource" / "utils"
                                               / "validate_submission.py"))
    args = ap.parse_args()

    setup_logging()
    paths = Paths(args.data_dir, args.work_dir)
    for stage in STAGES if args.stage == "all" else [args.stage]:
        if stage == "convert":
            import convert
            convert.run(paths)
        elif stage == "slices":
            import slices
            slices.run(paths, REPO / "configs" / "slices.json")
        elif stage == "sample":
            import sample
            sample.run(paths)
        elif stage == "normalize":
            import normalize
            normalize.run(paths, args.subset)
        elif stage == "block":
            import blocking
            blocking.run(paths, args.subset)
        elif stage == "features":
            import features
            features.run(paths, args.subset, args.neg_frac)
        elif stage == "train":
            import model
            model.train(paths, args.train_subset, args.neg_frac or 1.0)
        elif stage == "predict":
            import model
            model.predict(paths, args.subset)
        elif stage == "submit":
            import submit
            submit.run(paths, Path(args.out_dir), Path(args.validator))
        elif stage == "xenc-oof":
            import xenc
            xenc.oof_predictions(paths, args.train_subset, args.neg_frac or 0.15)
        elif stage == "xenc-train":
            import xenc
            xenc.train(paths, args.xenc_tag, xenc.XConfig(
                model=args.xenc_model, epochs=args.xenc_epochs, lora=args.xenc_lora,
                train_frac=args.xenc_train_frac, use_rest=args.xenc_use_rest,
                addr_placeholder=args.xenc_addr_placeholder))
        elif stage == "xenc-blend":
            import xenc
            xenc.blend(paths, args.xenc_tag)
        elif stage == "xenc-score":
            import xenc
            xenc.score(paths, args.subset, args.xenc_tag)
        elif stage == "xenc-apply":
            import xenc
            xenc.apply(paths, args.subset, args.xenc_tag)
        elif stage == "holdout-report":
            import stack
            stack.holdout_report(paths)
        elif stage == "val2-check":
            import stack
            stack.val2_check(paths, ["e5s_2ep", args.xenc_tag])
        elif stage == "stack":
            import stack
            stack.fit(paths, args.xenc_tag)
        elif stage == "stack-apply":
            import stack
            stack.apply_test(paths, args.xenc_tag)


if __name__ == "__main__":
    main()
