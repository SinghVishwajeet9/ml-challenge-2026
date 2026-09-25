# Submission 3 summary

Business Entity Resolution challenge. Goal: for every Source 1 (S1) business, find the
matching Source 2 / Source 3 (S2/S3) records. Metric: F0.5, averaged per S1 entity
(precision weighted twice as much as recall).

## Scores so far

| Submission | Validation F0.5 | Leaderboard F0.5 | Main change |
|---|---|---|---|
| 1 | 0.965 | 0.952 | First full pipeline (normalization, blocking, LightGBM) |
| 2 | 0.977 | 0.962 | Rarity and competition features, blocking fixes, France regions |
| 3 | 0.985 | 0.975 | Transformer cross-encoder blended with LightGBM |

Validation = held-out training regions (6 US / Indian states never used for development),
scored with the same F0.5 formula as the leaderboard. The leaderboard has so far been about
0.010 to 0.015 below validation (0.010 for submission 3).

## Files in this folder

- `matching_results.tsv`: final matches for all 1,732,544 test S1 entities (validator: PASS)
- `business_entity_resolution_code.zip`: the code that produced it

## Pipeline

1. **Convert**: raw TSVs to Parquet (quoting disabled, since names contain `"`; placeholder
   addresses such as `None` and `<NULL>` treated as missing).
2. **Slices**: training data split into whole-region slices (tune for training, val held
   out), keeping the real share of unmatched records (about 26%) and singletons (5.6%).
3. **Normalization**
   - Names: Indian scripts (Devanagari, Tamil, Telugu, Kannada, Bengali, Gurmukhi)
     transliterated to Latin, with Tamil-specific fixes; legal forms standardized (Pvt,
     Ltd, LLC, SAS, SARL ...); core, consonant-skeleton, joined-up, initials and website
     forms of each name.
   - Addresses: state detection (fixed lists plus states learned from training addresses),
     house numbers, units, landmarks, per-country abbreviations (US, India, France).
   - France (no training data): city to region learned from French addresses themselves,
     without labels (region coverage 65% to 97%).
4. **Blocking** (candidate generation, 40 candidates per S1)
   - Searched per state, so run time grows linearly with data size.
   - Four sources: rare name words and word pairs, rare address words and numbers,
     joined-name / website key, character-trigram TF-IDF; pooled and re-ranked.
   - State links learned from training data (Telangana / Andhra Pradesh, Washington / DC).
   - Recall about 98% of true pairs.
5. **Features** (about 100 per candidate pair): fuzzy name and address scores, rarity-weighted
   (IDF) overlap, legal forms, house number / state / unit agreement, and competition
   features (how an S1 compares with other S1s competing for the same S2/S3 record).
6. **LightGBM** scores every candidate pair.
7. **Transformer cross-encoder** (new in submission 3) re-scores the uncertain pairs.
8. **Decision**: blend, one-owner rule (each S2/S3 record goes to at most one S1), threshold
   chosen on validation.

## What is new in submission 3

- Model: `intfloat/multilingual-e5-small` (MIT licence, 118M parameters, a multilingual
  BERT-family encoder, runs locally on the GPU, no external API).
- Input: both records as one text pair (raw names and addresses, native scripts kept),
  plus the LightGBM feature vector, combined in a small network.
- Trained 2 epochs on about 220,000 uncertain training pairs (tune slice only).
- Applied only to pairs where LightGBM is uncertain (probability 0.01 to 0.99); on test
  that was 3.6M of 69.3M candidate pairs.
- Final score = 60% transformer + 40% LightGBM on those pairs, threshold 0.62 (chosen on
  validation).
- On the uncertain validation pairs: AUC 0.976 (LightGBM 0.951).
- Compared with submission 2: 104,942 matched pairs removed, 102,594 added; the largest
  changes were in India (109,542 S1 entities changed), then US (45,282) and France (29,260).

## Precision and recall (validation, submission 3 settings)

| Threshold | Precision | Recall | F0.5 |
|---|---|---|---|
| 0.50 | 99.49% | 96.70% | 0.9841 |
| 0.70 | 99.78% | 96.16% | 0.9848 |
| 0.90 | 99.91% | 94.69% | 0.9804 |

Precision is already very high; remaining losses are mostly missed matches (records with
no address, crowded same-name businesses, transliterated names).

## Tested and not used (measured on validation)

| Idea | Result |
|---|---|
| Second model comparing each candidate with the S1's confident matches | +0.0005 (noise) |
| Second stage using first-model probabilities for competition | -0.0006 |
| Per-country thresholds, rank rules | at most +0.002 |
| "At least one match per S1" | worse (0.9771 vs 0.9774) |
| Per-record statistics features (from past competition winners) | +0.0001 |
| Unit/floor features alone | about +0.001 at most (kept in code) |
| Train/test overlap | none found (0.00%) |
| ID number patterns | none (correlation 0.002) |

## Rules compliance

- Only the provided training data is used for training; no external databases, APIs,
  geocoding or internet data.
- Libraries: Polars, NumPy, SciPy, RapidFuzz, scikit-learn, sparse-dot-topn, anyascii,
  LightGBM, PyTorch, Transformers, PEFT (MIT / BSD / Apache / ISC licences).
- Transformer: `intfloat/multilingual-e5-small`, MIT licence, 118M parameters (limit 8B);
  pretrained weights downloaded once, then everything runs offline.
- French addresses (test only, no labels) were inspected to design generic address rules
  and the label-free city to region mapping.

## Next experiments

- Same transformer trained 4 epochs: validation 0.9857 (vs 0.9850).
- MuRIL (Indian-language model, Apache 2.0) with LoRA on a 50% sample: running.
- Possible CatBoost as an extra blend member.
