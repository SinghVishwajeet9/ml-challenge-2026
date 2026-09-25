# Business Entity Resolution

Matches Source 2 / Source 3 business records to each Source 1 entity.

Pipeline: convert, normalize, block (candidate generation), features, LightGBM matcher,
decision step, output TSVs. Fully offline: no external data, APIs or lookups.

## Setup
Python 3.12 or 3.13.
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
.venv/bin/python -m pip install -r requirements.txt       # Linux / macOS
```
The dataset is not included. By default it is read from `../student_resource/dataset`
(the folder containing `train/` and `test/`); override with `--data-dir`.
Intermediate files go to `work/` (override with `--work-dir`).
A machine with 16 GB RAM is enough; every stage processes data in chunks.

## Reproduce the submission
Run from the repository root, in this order:
```bash
python src/run_pipeline.py --stage convert                      # TSV -> Parquet
python src/run_pipeline.py --stage slices                       # region-based tune / val slices
python src/run_pipeline.py --stage normalize --subset train
python src/run_pipeline.py --stage normalize --subset tune
python src/run_pipeline.py --stage normalize --subset val
python src/run_pipeline.py --stage block --subset tune
python src/run_pipeline.py --stage block --subset val
python src/run_pipeline.py --stage features --subset tune --neg-frac 0.15
python src/run_pipeline.py --stage features --subset val
python src/run_pipeline.py --stage train --train-subset tune --neg-frac 0.15
python src/run_pipeline.py --stage normalize --subset test
python src/run_pipeline.py --stage block --subset test
python src/run_pipeline.py --stage predict --subset test
python src/run_pipeline.py --stage submit                       # writes output/*.tsv and validates
```
Outputs: `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
Each stage reads the previous stage's Parquet output and can be rerun on its own.

## Stages
| Stage | What it does |
|---|---|
| `convert` | Reads the TSVs with quoting disabled (names contain `"`), maps placeholder addresses (`None`, `<NULL>`) to null, writes Parquet |
| `slices` | Splits train into whole-region slices (tune for training, val held out), keeping the real distractor and singleton rates. Region lists pinned in `configs/slices.json` |
| `normalize` | Names: transliteration of Indic scripts to Latin, legal-form canonicalisation, core / consonant-skeleton / joined-up / initials / website forms. Addresses: state, house numbers, unit, landmark, per-country abbreviations |
| `block` | Candidate generation within each state: IDF inverted indexes on name and address tokens and token pairs, joined-name / website key, character-trigram TF-IDF; pooled and re-ranked, top 40 per Source 1 entity |
| `features` | 53 pair and group features: fuzzy string scores, token overlaps, legal forms, house / number / state agreement, competition between Source 1 entities for the same record |
| `train` | LightGBM binary classifier; decision threshold chosen on the held-out val slice |
| `predict` | Scores all candidates, keeps each Source 2 / Source 3 record for its best Source 1 entity only, applies the threshold |
| `submit` | Writes both TSVs (one row per test Source 1 entity) and runs the official validator |

`src/evaluate.py` implements the macro F0.5 metric used for local validation.

## Licences
All dependencies are MIT, BSD, Apache-2.0 or ISC licensed. The matching model is a
LightGBM gradient-boosted tree model trained only on the provided training data.
