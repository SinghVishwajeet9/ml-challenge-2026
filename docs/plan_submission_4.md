# Plan: Submission 4 (final version, overnight safe mode)

Status: NOT started. Waiting for go-ahead.

## 1. Goal and realistic expectation

- Leaderboard now: **0.975** (submission 3). Top of leaderboard: about 0.986.
- This plan: transformer trained on about 4x more data + a stacker that feeds the
  transformer's judgement into the one-owner competition, checked on two held-out sets
  (val, and a new val2 of three other Indian states).
- Realistic expectation: validation about 0.988 to 0.990, leaderboard about **0.979 to 0.981**.
  Validation gains have carried over to the leaderboard roughly one to one, not multiplied.
- Reaching 0.985 on the leaderboard will likely need the follow-up levers in section 11.

## 2. Evidence behind the plan (all measured on validation)

| Fact | Numbers |
|---|---|
| Transformer is the strongest component | Val 0.9774 to 0.9850, leaderboard 0.962 to 0.975 |
| Transformer is data-limited, not overfitting | Train log-loss 0.166 vs val 0.194; improves to epoch 3, overfits at epoch 4 |
| It learned from little data | 220,000 uncertain pairs from the tune slice (23% of training S1) |
| Stacker adds a real but modest gain | Leave-one-state-out: blend 0.9857 vs stacker 0.9866 (**+0.0009**); positive in every real held-out state (DC, KS, ND, OR, UT, Tamil Nadu). An earlier random split showed +0.0015, partly from state-specific patterns |
| Stacker settings barely matter | 63 leaves 0.9866 vs shallow 15 leaves 0.9865 |
| LightGBM generalizes well | Train log-loss 0.0048, out-of-fold 0.0087, val 0.0074; its confident mistakes are only about 3% of all errors |
| Other ideas were small or zero | 4 epochs +0.0007; MuRIL blend +0.0003; dual no-address threshold -0.0001; widening band at most +0.0005 |

## 3. Data split

| Part | Share of training S1 | Use in this plan |
|---|---|---|
| tune | 23% (about 516,000) | transformer training (LightGBM unchanged, trained on tune only) |
| **rest** | **65% (about 1,430,000)** | **transformer training (new) + per-state check** |
| **val2 (new)** | **5.2% (about 115,000): West Bengal, Kerala, Bihar** | **second held-out check; never trained or tuned on** |
| val | 7% (about 145,000; Tamil Nadu + 5 US states) | never used to train LightGBM or the transformer; used to measure, choose settings, and train the stacker (estimated with leave-one-state-out) |

- Not 100%: about 88% trains the transformer; val and val2 (about 12%) stay held out.
- Why val2: val has only one Indian state (Tamil Nadu), while India is 47% of test. Val2
  adds three different regions and scripts (Bengali, Malayalam, Hindi). Val accuracy is
  already fine (about +/-0.0003 with 145,000 S1); the gap is variety, not size. Val stays
  unchanged so all earlier numbers remain comparable. Val2 is also a fully independent
  check of the stacker, which is trained on val.
- The current LightGBM never saw the rest or val2 regions, so their uncertain pairs are
  chosen exactly as on test (no leakage).

## 4. Overnight safe mode (the PC must not crash)

| Safeguard | What it means |
|---|---|
| **No parallel jobs** | Every step runs after the previous one; CPU and RAM never shared by two heavy jobs |
| **Thread caps** | Polars (`POLARS_MAX_THREADS=10`), LightGBM and RapidFuzz limited to 10 of 12 threads |
| **Memory guard** | Before each heavy step: wait if free RAM is below about 3 GB; rest keeps only uncertain pairs; test features written to disk in parts; expected peak under 10 GB of 16 GB |
| **Stop on error** | Each step runs only if the previous one succeeded; nothing half-finished is used |
| **Logs** | Every step writes timestamped progress to a log file for the morning |
| **No automatic upload** | The run only prepares files; uploading stays manual |
| **Keep awake** | Desktop on AC: sleep and hibernate are "never" (checked); no pending update restart (checked); pause Windows Update for 7 days; the app is also asked to keep the PC awake; heavy apps closed |

## 5. Steps in order

### Step 0. Code changes (while awake, about 30 to 40 min)
1. Transformer text for records without an address: `address unavailable`
   (saved as a model setting; the submission 3 model keeps its old format).
2. `predict` also saves, in parts on disk, the features of every pair with LightGBM
   probability >= 0.001 (needed by the stacker) for test and val2.
3. New `stack` stage: competition features on the blended score, shallow LightGBM
   (15 leaves, depth 4, L2 5), estimated with leave-one-state-out on val; final stacker
   trained on all of val; scored on val2; applied to test.
4. New report: F0.5 per country and per state for rest and val2 (LightGBM only).
5. New `val2` subset (West Bengal, Kerala, Bihar) carved out of the training data; rest
   excludes it. Blend reports also score val2.
6. Thread caps and memory guard in the run script.
7. Quick test of each change on a small sample; confirm the submission 3 model still
   scores identically; commit and push. After this: no code changes until the run ends.

### Step 1. Build `rest` and `val2` (CPU, about 1 min)
`python src/run_pipeline.py --stage normalize --subset rest`
`python src/run_pipeline.py --stage normalize --subset val2`
- Check: rest about 1,430,000 S1, val2 about 115,000 S1; zero overlap among tune, val,
  val2 and rest.

### Step 2. Blocking for rest and val2 (CPU, about 22 min)
`python src/run_pipeline.py --stage block --subset rest`
`python src/run_pipeline.py --stage block --subset val2`
- Check: files written, no errors; val2 blocking recall reported (it has labels).

### Step 3. Score rest and val2 with the current LightGBM (CPU, about 50 min)
`python src/run_pipeline.py --stage predict --subset rest`
`python src/run_pipeline.py --stage predict --subset val2`
- Saves uncertain pairs (0.01 to 0.99) with features; for val2 also every pair with
  probability >= 0.001 (to score the stacker on val2).
- Check: rest roughly 450,000 to 750,000 uncertain pairs; 35% to 45% true matches.

### Step 3b. Per-state check: is the gap India or France? (CPU, about 2 min)
- LightGBM-only F0.5 on rest and val2 per country and per state vs LightGBM-only on val
  (0.9774; India/Tamil Nadu and US separately).
- Other Indian states about equal to Tamil Nadu: the gap is more likely France.
- Other Indian states clearly lower: test India explains much of the gap; the 4x data is
  the fix.
- Informational only; the run continues either way.

### Step 4. Train the transformer on tune + rest (GPU, about 1.5 to 2 h)
`python src/run_pipeline.py --stage xenc-train --xenc-tag e5s_big --xenc-epochs 3 --xenc-use-rest`
- `intfloat/multilingual-e5-small` (MIT, 118M), about 820,000 uncertain pairs
  (tune + rest, val2 excluded), 3 epochs.
- Check: val AUC on uncertain pairs above 0.982 (best so far).

### Step 5. Rescore test, saving the wider feature set (CPU, about 45 min)
`python src/run_pipeline.py --stage predict --subset test`
- Runs after step 4 (not in parallel).

### Step 6. Transformer blend on val and val2 (GPU + CPU, about 15 min)
`python src/run_pipeline.py --stage xenc-blend --xenc-tag e5s_big`
- Blend weight and threshold chosen on val only.
- Val2 check with the same weight and threshold: new transformer vs the submission 3
  transformer (`e5s_2ep`) vs LightGBM alone. Neither transformer ever saw val2.

### Step 7. Stacker (CPU, about 10 min)
`python src/run_pipeline.py --stage stack --xenc-tag e5s_big`
- Leave-one-state-out val score of the stacker vs the blend.
- Final stacker trained on all of val; threshold chosen on val.
- Val2 score of the final stacker (fully independent: trained on val, never saw val2).

### Step 8. Automatic decision (section 6)

### Step 9. Transformer on test (GPU, about 25 min)
`python src/run_pipeline.py --stage xenc-score --subset test --xenc-tag e5s_big`

### Step 10. Stacker (or blend) on test, one-owner rule, threshold (CPU, about 5 min)

### Step 11. Files and validation (CPU, about 5 min)
`python src/run_pipeline.py --stage submit` (validator must print PASS)

### Step 12. Checks and filing (about 5 min)
- Every S1 once; ids only from test S2/S3; no duplicates; no em dashes, comments or AI
  mentions; compare with submission 3 per country.
- `submission\submission_4\`: matching_results.tsv, code zip, SUMMARY.md. Commit and push.

## 6. Automatic decision rule (step 8)

Uses the **leave-one-state-out val score** of the final method (stacker, or blend if the
stacker does not help), **plus the val2 guard**:

- **Val2 guard:** on val2 the final method must be at least as good as the submission 3
  method (blend with `e5s_2ep`). If it is worse on val2, the run stops, whatever the val
  score (possible overfit to val-like data).

| Honest val F0.5 (and val2 guard passed) | Action (automatic) |
|---|---|
| **0.9867 or higher** (at least +0.001 over 0.9857, the best so far) | Build submission 4 files (steps 9 to 12); upload stays manual |
| 0.9850 to 0.9867 | Build the files but mark them "small gain"; decide together in the morning |
| below 0.9850 | Stop; submission 3 stays our best |

## 7. Checkpoints (the run stops if any fails)

| After | Expected |
|---|---|
| Step 0 | Sample tests pass; submission 3 model still scores identically |
| Step 1 | rest about 1,430,000 S1, val2 about 115,000 S1; no overlaps |
| Step 2 | files written; no errors |
| Step 3 | rest 450,000 to 750,000 uncertain pairs; 35% to 45% true matches |
| Step 4 | val AUC on uncertain pairs above 0.982 |
| Step 6 | val2: new transformer not worse than the submission 3 method |
| Step 7 | stacker leave-one-state-out gain positive; val2 not worse |
| Step 11 | validator PASS |
| Step 12 | all checks clean |

## 8. Time and machine load (sequential)

| Phase | Time | Load |
|---|---|---|
| Step 0 | about 30 to 40 min (awake) | light |
| Steps 1 to 3b | about 1 h 20 min | CPU heavy |
| Step 4 | about 1.5 to 2 h | GPU heavy |
| Step 5 | about 45 min | CPU heavy |
| Steps 6 to 8 | about 25 min | GPU + CPU |
| Steps 9 to 12 | about 40 min | GPU then CPU |
| **Total after step 0** | **about 4.5 to 5.5 h** | |

## 9. Safety and rules

- Val is never used to train LightGBM or the transformer; the stacker uses val with
  leave-one-state-out scoring for the estimate.
- Val2 is never used to train or tune anything; it is only an independent check.
- Existing models and submission files are untouched (new model in `work/xenc/e5s_big/`).
- No code changes while jobs run.
- Same MIT model (118M), only provided training data, no external data or APIs.

## 10. Tested and not included

| Idea | Result |
|---|---|
| Dual threshold for no-address pairs | -0.0001 (held-out); same-name S1 ambiguity |
| Widening the transformer band | at most about +0.0005 (about 480 reachable errors) |
| Cross-encoder hyperparameter sweep | expected +0.0005 to +0.002; not worth delaying |
| Stacker grid search | two very different settings differ by 0.0001; not worth 30 min |
| 5-fold grid search for LightGBM / transformer | about 4 h CPU / days of GPU; expected gain small |
| Replacing LightGBM (CatBoost, XGBoost, neural nets) | same family or weaker on tabular data; LightGBM already generalizes well |
| Enlarging val from 7% to 10% | accuracy already about +/-0.0003; variety is the issue, addressed by val2 |
| France fixes from reviewed documents (ALL fallback, departments, postal codes, bis/ter, legal forms) | already handled or negligible (97% region coverage, 0.5% postal codes, legal forms recognised) |
| Diagnostic France-only submission | not wanted |
| Running steps in parallel | dropped for stability (would save about 45 min) |

## 11. Follow-up levers (separate decisions after submission 4)

| Lever | Targets | Notes |
|---|---|---|
| Learned re-ranking in blocking | part of the 9,766 true pairs never found (largest error pool) | CPU, measurable on val |
| Dense retrieval (embeddings + nearest neighbours per state) | transliterated names missed by blocking | GPU, about 2 to 3 h |
| France pseudo-labelling | France (no labels) | only if step 3b points at France; leaderboard is the only check |
| LightGBM on tune + rest; CatBoost blend member | model | about +0.002 and +0.001 to +0.003 |
| Final package | `Documentation_template.md`, zip in required structure | before the deadline |
