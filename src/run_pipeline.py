"""Entry point: python src/run_pipeline.py --stage <name> [--data-dir ...] [--work-dir ...]

Stages (each reads the previous stage's Parquet output, so they can be rerun alone):
  convert  raw TSVs -> work/raw/*.parquet
  slices   region-based dev / tune / val slices of train -> work/slices/
  sample   small fixed sample of dev (+ test France) for rule development -> work/sample/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import Paths, setup_logging

REPO = Path(__file__).resolve().parents[1]
STAGES = ["convert", "slices", "sample"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES + ["all"])
    ap.add_argument("--data-dir", default=str(REPO.parent / "student_resource" / "dataset"),
                    help="folder containing train/ and test/")
    ap.add_argument("--work-dir", default=str(REPO / "work"),
                    help="where intermediate Parquet files go")
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


if __name__ == "__main__":
    main()
