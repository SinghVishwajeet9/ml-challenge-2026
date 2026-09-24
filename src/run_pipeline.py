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
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import Paths, setup_logging

REPO = Path(__file__).resolve().parents[1]
STAGES = ["convert", "slices", "sample", "normalize", "block", "features", "train", "predict",
          "submit"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES + ["all"])
    ap.add_argument("--data-dir", default=str(REPO.parent / "student_resource" / "dataset"),
                    help="folder containing train/ and test/")
    ap.add_argument("--work-dir", default=str(REPO / "work"),
                    help="where intermediate Parquet files go")
    ap.add_argument("--subset", default="sample",
                    choices=["sample", "dev", "tune", "val", "train", "test"],
                    help="which records the normalize / block / features stages process")
    ap.add_argument("--neg-frac", type=float, default=None,
                    help="features: keep this fraction of non-matching pairs (training only)")
    ap.add_argument("--train-subset", default="tune", help="train: features to fit on")
    ap.add_argument("--out-dir", default=str(REPO / "output"), help="submit: output folder")
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


if __name__ == "__main__":
    main()
