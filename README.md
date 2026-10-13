# Business Entity Resolution

Matches Source 2 / Source 3 business records (India, US, France) to each Source 1 entity.
Metric: macro F0.5 per Source 1 entity. Fully offline: only the provided data, no external
data, APIs or lookups. All models are open (MIT / Apache-2.0).

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| convert, slices | `convert.py`, `slices.py` | TSV to Parquet; region-based tune / val / val2 / delhi slices of train (configs/slices.json) |
| normalize | `normalize.py`, `geo.py`, `native_dict.py` | clean names / addresses, legal forms, house numbers, units, states; native-script words to English (configs/native_dict.json, learned without labels by stage `native-dict`) |
| block | `blocking.py` | candidate pairs per state: name, address, name+address and joint TF-IDF nearest-neighbour indexes, no-address quota, code tokens, second pass from confident first-pass matches (`work/first_pass/<subset>.parquet`) |
| features | `features.py` | 106 pairwise and competition features |
| pseudo | `pseudo.py` | confident test / val2 pairs as pseudo-labels (p >= 0.98 match, p <= 0.02 non-match) |
| train, predict | `model.py` | LightGBM matcher; scores every candidate |
| xenc-* | `xenc.py` | multilingual-e5-small cross-encoder with numeric features on uncertain pairs (LightGBM p in [0.01, 0.99]) |
| rival cross-encoder | `rival_data.py`, `rival_encoder/train_rival.py`, `rival_encoder/score_rival.py` | rival-aware multilingual-e5-base cross-encoder (tag f1, stacker column pf): record vs "candidate S1 + strongest rival S1" |
| stack | `stack.py` | LightGBM stacker over model scores and score competition, leave-one-state-out on val; checks on val2 and delhi |
| trap, refine | `trap.py`, `refine.py` | final scores, Bengali-script transformer switch, same-script links, source cap, contested no-address rule |
| submit | `submit.py` | writes `output/matching_results.tsv`, `output/candidate_pairs.tsv` and runs the validator |
| bi-encoder | `biencoder.py` | fine-tunes multilingual-e5-small contrastively (hard negatives from blocking), embeds every record, retrieves each record's nearest S1 (dense candidates) |
| final | `final.py`, `ef_decision.py` | stacker on val + val2 + delhi, rescue model for dense-only candidates, per-S1 expected-F0.5 decision, refine, output files |

Switches (on / off per idea) are in `configs/switches.json`.

## Setup

Python 3.13 (3.12 also works). A 16 GB RAM machine is enough; large stages run per country and
in chunks. A CUDA GPU (12 GB) is needed for the cross-encoders.
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
.venv/bin/python -m pip install -r requirements.txt       # Linux / macOS
```
The dataset is read from `../student_resource/dataset` (override with `--data-dir`);
intermediate files go to `--work-dir` (default `work/`).

## Reproduce the submission

Run from the repository root (`P = python src/run_pipeline.py --work-dir work`).

```bash
# 1. data and normalisation
$P --stage convert
$P --stage slices
$P --stage native-dict
for s in tune val val2 delhi test; do $P --stage normalize --subset $s; done

# 2. first pass (writes the seeds for the second blocking pass): run steps 3-5 once with
#    "second_pass": false in configs/switches.json, then copy work/pred/<subset>_scored.parquet
#    (work/models/val_scored.parquet for val) to work/first_pass/<subset>.parquet

# 3. blocking and features
for s in tune val val2 delhi; do $P --stage block --subset $s; done
for c in France India US; do $P --stage block --subset test --country $c; done
$P --stage features --subset tune --neg-frac 0.15
$P --stage features --subset val

# 4. pseudo-labels, LightGBM, scoring
$P --stage pseudo --subset val2
for c in France India US; do $P --stage pseudo --subset test --country $c; done
$P --stage train --train-subset tune --neg-frac 0.15
for s in val val2 delhi; do $P --stage predict --subset $s; done
for c in France India US merge; do $P --stage predict --subset test --country $c; done

# 5. small cross-encoders (e5s_big: 3 epochs, e5s_2ep: 2 epochs, used for Bengali script)
$P --stage xenc-oof --train-subset tune --neg-frac 0.15
$P --stage xenc-train --xenc-tag e5s_big --xenc-epochs 3
$P --stage xenc-train --xenc-tag e5s_2ep --xenc-epochs 2
for t in e5s_big e5s_2ep; do
  for s in val val2 delhi test; do $P --stage xenc-score --subset $s --xenc-tag $t; done
  $P --stage xenc-blend --xenc-tag $t
done

# 6. rival-aware cross-encoder (GPU)
python src/rival_data.py --work-dir work  # work/rival/: text pairs + the src/rival_encoder scripts
cd work/rival
python train_rival.py --model intfloat/multilingual-e5-base --mode select --tag f1 --hours 1.5 --batch 32
for s in val val2 delhi test; do python score_rival.py --tag f1 --mode select --set $s; done
cd ../..
mkdir -p work/xenc/f1l && cp work/xenc/e5s_big/blend.json work/xenc/f1l/
for s in val val2 delhi test; do cp work/rival/results/f1_${s}_0.parquet work/xenc/f1l/${s}_scored.parquet; done

# 7. stacker with F1, trap + refine, files
export BER_STACK_EXTRA=f1
$P --stage stack --xenc-tag e5s_big
$P --stage refine --subset val2 --xenc-tag e5s_big
$P --stage trap --xenc-tag e5s_big
$P --stage refine --xenc-tag e5s_big
$P --stage submit --matches work/trap/test_matches_refined.parquet --cand-min-p 0.01

# 8. bi-encoder: dense candidates (GPU, about 1 h training + 1 h embedding)
python src/biencoder.py train --work-dir work --batch 128 --hours 1.0
for s in val val2 delhi test; do
  python src/biencoder.py embed    --work-dir work --subset $s
  python src/biencoder.py retrieve --work-dir work --subset $s --k 10
done

# 9. final: stacker on val + val2 + delhi, rescue model, expected-F0.5 decision, files
python src/final.py --work-dir work --out output --candidates output/candidate_pairs.tsv
```
Outputs: `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
Each stage reads the previous stage's Parquet output and can be rerun on its own.

## Results

F0.5 on held-out regions (leave-one-region-out for the stacker and the rescue model):

| | val | val2 | delhi (crowded holdout) | leaderboard |
|---|---|---|---|---|
| submission 7a | 0.99016 | 0.99136 | 0.98976 | 0.983905 |
| submission 8 (with F1) | 0.99031 | 0.99163 | 0.99006 | 0.984428 |
| submission 9c (stacker on 3 holdouts, expected-F0.5 decision) | 0.99129 | 0.99197 | 0.99036 | 0.984795 |
| **final: + bi-encoder dense retrieval and rescue** | **0.99357** | **0.99394** | **0.99302** | not uploaded (upload limit reached) |

Candidate recall on the holdouts rises from 98.5-99.0% (lexical blocking) to 99.6-99.7% with
each record's nearest S1 from the bi-encoder (about 0.3 extra candidates per S1).
