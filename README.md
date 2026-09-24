# Business Entity Resolution

Matches Source 2 / Source 3 business records to each Source 1 entity.
Pipeline: convert → normalize → block → features → LightGBM → decision → output TSVs.
Design and roadmap: [docs/PLAN.md](docs/PLAN.md). Data findings:
[docs/PROFILING_FINDINGS.md](docs/PROFILING_FINDINGS.md).

## Setup
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # Linux / Colab / AWS
```
The dataset is not in the repo. By default it is read from
`../student_resource/dataset` (containing `train/` and `test/`); override with
`--data-dir`. Intermediate files go to `work/` (override with `--work-dir`).

## Run
```bash
python src/run_pipeline.py --stage convert   # TSV -> Parquet (~35 s)
python src/run_pipeline.py --stage slices    # region-based dev/tune/val slices (~45 s)
python src/run_pipeline.py --stage all
```
Each stage reads the previous stage's Parquet output and can be rerun on its own.

## Stages so far
| Stage | Output | Notes |
|---|---|---|
| `convert` | `work/raw/{train,test}_records.parquet`, `train_gt_pairs.parquet` | Reads TSVs with quoting disabled (names contain `"`); maps `None`/`<NULL>` addresses to null |
| `slices` | `work/slices/{dev,tune,val}_*.parquet` | Whole-region slices keep the real distractor (≈26%) and singleton (≈5.6%) rates. Region lists pinned in `configs/slices.json` |

`src/evaluate.py` holds the local F0.5 scorer (`python src/evaluate.py` runs its self-test).

## Validation protocol
- `dev`: development of normalization / blocking / features.
- `tune`: training the matcher.
- `val`: held out, used only for thresholds and final estimates.
