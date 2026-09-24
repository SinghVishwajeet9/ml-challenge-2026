# Plan — Business Entity Resolution Challenge

Informed by [PROFILING_FINDINGS.md](PROFILING_FINDINGS.md). Constraints from the problem
statement: MIT/Apache-2.0 licensed model, ≤8B params, no external lookups/APIs, F_0.5
macro-averaged scoring (precision weighted 2× over recall), output must be
`output/matching_results.tsv` + `output/candidate_pairs.tsv`, validated with
`utils/validate_submission.py`.

## 0. Verified facts that shape the design
- **Each S2/S3 record matches at most one S1 entity** (7,638,365 matched IDs, none in
  two lists). ~26% of S2/S3 records (~2.7M) are unmatched distractors.
- **Matched pairs always share the country label** (0 mismatches in a 153k-pair
  sample) → country is a safe hard block; France forms its own blocks in test.
- **Recall matters more than "precision-heavy" suggests**: singletons are only 5.6% and
  the mean entity has ~3.5 matches. Predicting 1 of 4 true matches scores F0.5 = 0.625;
  predicting all 4 plus 1 false positive scores 0.83. Tune, don't guess.

## 1. Data audit — done
See PROFILING_FINDINGS.md: field completeness, country distribution, ground-truth match
cardinality, name/address noise patterns, true-match-vs-random similarity calibration.

## 2. Preprocessing / normalization
- **Name normalization**: lowercase, strip punctuation, expand common legal-suffix
  abbreviations (Pvt→Private, Ltd→Limited, Corp→Corporation, & ↔ and), tokenize.
  Keep a "core name" with legal suffixes/stopwords removed as a separate field.
- **Transliteration handling**: Devanagari/Tamil → Latin via `indic-transliteration`
  (MIT); accents/other scripts via `anyascii` (ISC). **Avoid `unidecode` (GPL).**
- **Address normalization**: separate normalizers for US-style vs. India-style
  addresses, expanding abbreviations (St→Street, Rd→Road), extracting house/plot
  number, PIN/ZIP, city, state; strip landmark phrases ("Near SBI ATM") into a separate
  field. Generic fallback normalizer for unseen countries (France: accent stripping,
  rue/avenue/bd, 5-digit postcode).
- Flag likely gibberish names (e.g. `Dovacira`) so the model can lean on address.

## 3. Blocking / candidate generation (determines recall ceiling)
- Per country, union (OR) of several candidate sources:
  - Rare name tokens / phonetic keys (inverted index, IDF-weighted, top-K per S1).
  - Address keys (house number + locality, PIN/ZIP + street token).
  - Char n-gram TF-IDF top-K (`sklearn` + `sparse_dot_topn`) on name and address.
  - *(Optional)* e5-small embeddings + faiss top-K if recall plateaus.
- Cap block sizes / skip frequent tokens ("hyderabad", "road") to avoid explosion.
- 3.3% of S2/S3 have no address → name blocking must stay typo-tolerant.
- **Target: ≈98% pair recall at ≤50 candidates per S1**, measured before modelling.
- `candidate_pairs.tsv` = exactly the set the model scores (after any pre-filter).

## 4. Feature engineering
- **Pairwise**: name/address token-Jaccard, RapidFuzz ratio / token_set / Jaro-Winkler,
  TF-IDF cosine, house-number / PIN / city / state match flags, length diffs,
  gibberish-name flag, source (S2/S3), address-missing flag.
- **Group-level** (big win): rank of candidate within its S1, gap to S1's top score,
  number of candidates; for each S2/S3 record, is this S1 its best S1 and the margin
  to its second-best S1; similarity to already-confident candidates of the same S1.
- Use RapidFuzz (C++), not `difflib` — difflib is too slow at this scale.

## 5. Matching model
- LightGBM (MIT) on pairs: positives from ground truth, negatives = blocked-but-unmatched
  candidates (naturally hard). Subsample negatives to fit memory.
- Optional small permissively-licensed embedding model (e.g. `multilingual-e5-small`,
  MIT) as a feature; must stay MIT/Apache-2.0 and ≤8B params.

## 6. Decision policy
- **Exclusivity**: assign each S2/S3 record to at most one S1 (its best-scoring S1 if
  above threshold).
- **Per-entity set selection**: choose each S1's match set to maximise expected F0.5
  from calibrated probabilities (not a single global cutoff); empty set allowed.
- Tune all thresholds on the held-out validation regions only.

## 7. Post-processing & validation
- Enforce hard constraints: one row per S1 test entity; IDs only from test S2/S3; no
  duplicates; matches ⊆ candidates.
- Local F0.5 scorer (macro per S1, singletons included) on held-out regions.
- Run `utils/validate_submission.py` before every leaderboard upload.

## 8. Packaging
- `code/business_entity_resolution/src/` — full pipeline, runnable end-to-end.
- `code/business_entity_resolution/README.md` — exact reproduction steps.
- `code/business_entity_resolution/requirements.txt` — pinned deps.
- Fill in `Documentation_template.md`.

## 9. Infrastructure — PC first, Colab for GPU, AWS for the full run
- **Local PC (primary)**: Ryzen 5 5600 (12 threads), 16 GB RAM, RTX 3060. Faster than
  free Colab (2 vCPU, ~12 GB) for this CPU-bound workload. All dev work happens here.
- **Colab (secondary)**: GPU embeddings (optional), or working away from the PC.
- **AWS (once, at the end)**: r6i/r7i.4xlarge (128 GB) for full training + full test
  inference; also the fallback if test inference doesn't fit in 16 GB.
- **Code lives in `.py` modules under `src/`; notebooks only call them.** Same code on
  PC/Colab/AWS, synced via private GitHub repo (`ml challenge repo/`). All paths come
  from CLI args. Pipeline stays fully offline.
- **Every stage writes Parquet and is resumable** (convert → normalize → block →
  features → train → predict).

### Data tiers (region-sliced, not random)
Random S1 sampling thins out look-alike neighbours and inflates precision. Instead,
slice by **whole regions** (US states / Indian states): keep *every* S1/S2/S3 record in
the chosen regions, including distractors, plus the true matches of chosen S1s.

| Tier | Size | Where | Purpose |
|---|---|---|---|
| Dev | ~5% (~110k S1, ~500k S2/S3) | PC | Fast iteration on normalization, blocking, features |
| Tune | ~20–25% (~500k S1), disjoint regions | PC | Train LightGBM, tune decision policy; held-out part = validation |
| Full | all train + all test | AWS (test per-country on PC may fit) | Final training + submission |

Region lists live in a config file so slices are reproducible. The validation regions
are never used for development (also mimics unseen-country generalisation).

## 10. Execution roadmap (who does what, where, in order)

### Phase 0 — Setup (PC)
| # | Who | Step |
|---|---|---|
| 1 | Claude | Scaffold repo (`src/`, `requirements.txt`, `.gitignore` excluding data), venv, `git init`, first commit |
| 2 | You | Create a **private** GitHub repo and push |

### Phase 1 — Foundation (PC)
| # | Step | Check |
|---|---|---|
| 3 | TSV → Parquet (all data, once) + local F0.5 scorer | Row counts match profiling |
| 4 | Region extraction + Dev/Tune slices | Printed slice sizes |
| 5 | Normalization (name/address, per-country, transliteration, generic fallback) | Spot-check examples |

### Phase 2 — Blocking (PC; Colab only for optional embeddings)
| # | Step | Target |
|---|---|---|
| 6 | Blocking v1: rare name tokens + address keys, per country | Measure recall + candidates/S1 on Dev |
| 7 | Add char-TF-IDF top-K; tune K and caps; confirm on Tune | ≈98% recall at ≤50 candidates/S1 |
| 8 | *(Optional, Colab T4 or local 3060)* e5 embeddings if recall plateaus | — |

### Phase 3 — Matching model (PC)
| # | Step |
|---|---|
| 9 | Pairwise features |
| 10 | Group-level features |
| 11 | Train LightGBM on Tune, score validation regions |
| 12 | Decision step: exclusivity + per-entity F0.5 set selection |
| 13 | Error analysis → iterate 9–12 |

### Phase 4 — First leaderboard submission (PC, early)
| # | Who | Step |
|---|---|---|
| 14 | Claude | Run on test per country, write both TSVs, run `validate_submission.py` |
| 15 | You | Upload `matching_results.tsv`; compare with local validation F0.5 |
| 16 | Both | Iterate on PC; submit periodically |

### Phase 5 — Full-scale run (AWS, once validation plateaus)
| # | Who | Step |
|---|---|---|
| 17 | You | Billing alert, S3 bucket, upload Parquet, launch r6i/r7i.4xlarge (spot) |
| 18 | Claude | Exact commands (clone, install, run all stages) |
| 19 | You | Run, download outputs, **stop the instance** (budget ~2–4 h) |

### Phase 6 — Packaging (PC)
| # | Step |
|---|---|
| 20 | Fill in `Documentation_template.md` |
| 21 | Build `<team>_submission.zip`; verify it reproduces from a clean clone |

**Things only you can do:** GitHub account and pushing, running Colab cells, AWS
account/billing/instances, portal submissions.
