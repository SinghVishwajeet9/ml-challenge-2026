# Data Profiling Findings - Business Entity Resolution Challenge

Profiling was done on the **training set** (`dataset/train/`), using full-file scans for
counts/distributions and sampled pairs (3,000-20,000) for similarity calibration. No test data,
external lookups, or APIs were used.

## 1. Volume

| File | Rows |
|---|---|
| train_source1.tsv | 2,206,821 |
| train_source2.tsv | 5,034,616 |
| train_source3.tsv | 5,285,603 |
| train_ground_truth.tsv | 2,206,821 |
| test_source1.tsv | 1,732,544 |
| test_source2.tsv | 4,887,273 |
| test_source3.tsv | 5,082,316 |

This is a large-scale problem - pairwise cross-joining Source 1 against Source 2/3 is
infeasible (2.2M × 10M+). Blocking / candidate generation must scale via inverted
indexes / hashing, not brute force.

## 2. Field completeness

- `business_name` and `country`: **100% populated** in all three sources.
- `business_address`: 100% populated in Source 1, but **empty in ~3.3%** of both
  Source 2 (168,967 rows) and Source 3 (175,916 rows) records.
- Country split (train): US ≈ 60%, India ≈ 40%. Test adds **France**, unseen in
  training - country must be treated as an open string label, never hardcoded/one-hot
  to {US, India}.

## 3. Ground truth shape

- **Singleton rate: 5.6%** (123,247 / 2,206,821 Source‑1 entities have no match).
- Match-count distribution peaks at 3-4 matches per entity, mean 3.46 (including
  singletons):

  | # matches | count |
  |---|---|
  | 0 | 123,247 |
  | 1 | 119,157 |
  | 2 | 375,212 |
  | 3 | 530,841 |
  | 4 | 484,115 |
  | 5 | 321,957 |
  | 6 | 164,868 |
  | 7 | 63,968 |
  | 8 | 18,680 |
  | 9 | 4,205 |
  | 10 | 534 |
  | 11 | 37 |

- Among non-singleton entities: average ≈1.78 matches from Source 2, ≈1.89 from
  Source 3 - matches are spread across both sources, not concentrated in one.
- **Implication:** singleton detection matters (5.6% of the macro-average), but the
  bulk of the score comes from correctly resolving multi-match entities - recall and
  precision on the "normal" case dominate.

## 4. Name noise patterns

- Non-ASCII (transliterated script) business names: **0% in Source 1**, **15.2% in
  Source 2**, **11.5% in Source 3** (Devanagari/Tamil script variants of the same
  business, e.g. `Best Infotech Limited` ↔ `बेस्ट इंफोटेक लिमिटेड`).
- Common variation types observed in true-match pairs:
  - Word-order shuffling: `Bnp Group of Companies-Delhi` ↔ `of Group Bnp
    Companies-Delhi Limited`
  - Legal-suffix noise: suffixes (Pvt/Ltd/LLC/Co/Private/Limited) added, dropped, or
    reordered across sources
  - Punctuation/hyphen insertion: `Choice Transit Enterprises` ↔ `The Choice
    Transit-Enterprises`
  - Character-level typos/corruption: `East Workshop` ↔ `EAST WORSKHP`, `Best
    Infotech Limited` ↔ `Best Ing0fche Limited`, `LLC` ↔ `1lc`

### Outlier pattern - unrelated names, identical addresses

A subset of true matches have **completely unrelated business names** but
**near-identical addresses**, e.g.:

```
S1-358737955  Cab Mills Pvt Ltd        | P No 13, Bajrang Dham Raipura, Kota, Rajasthan
S2-11844647   Dovacira                 | Rajasthan, KOTA, #187 P NO 13, BAJRANG DHAM RAIPURA

S1-221456935  Sahyog Institute         | ...Kukatpally, Balanagar, Hyderabad, Telangana
S3-68040586   Dovalyrafaye             | ...Kukatpally, Balanagar, Hyderabad, ...Telangana
```

This looks like deliberately injected noise to penalize name-only matching strategies.
**Blocking/matching cannot rely on name similarity alone**, or it will systematically
miss this class of true positive.

## 5. Address noise patterns

- Component reordering (city/state/street order differs across sources)
- Abbreviation inconsistency: `St` vs `Street`, `Rd` vs `Road`, state postal code
  (`RJ`) vs full name (`Rajasthan`)
- Case differences: ALL CAPS vs Title Case
- House-number/unit formatting variants: `Plot No: 51` vs `#51`, unit/floor info
  present in one source and missing in another
- Non-Latin script fragments appearing standalone in otherwise-Latin addresses (e.g.
  `தமிழ்நாடு` for "Tamil Nadu")
- India and US addresses have **structurally different layouts** (US: street, city,
  state; India: locality/PIN/state, more variable ordering) - a single normalization
  regex won't serve both well.

## 6. Similarity calibration (true match vs. random pair)

Computed on normalized (lowercased, punctuation-stripped) text using token-Jaccard and
`difflib.SequenceMatcher` character ratio.

| Metric | True-match median | True-match 10th pct | Random-pair median | Random-pair 90th pct |
|---|---|---|---|---|
| name_jaccard | 0.67 | 0.00 | 0.00 | 0.00 |
| name_char_ratio | 0.88 | 0.47 | 0.25 | 0.38 |
| addr_jaccard | 0.67 | 0.25 | 0.00 | 0.05 |
| addr_char_ratio | 0.84 | 0.46 | 0.24 | 0.35 |

- The bulk of true matches separate cleanly from random pairs on both name and
  address similarity (medians ~0.65-0.88 vs. ~0.24-0.25 for random pairs).
- **But the 10th percentile of true matches (name_jaccard = 0.0) overlaps the random
  distribution** - this is the garbled-name / address-only-match tail described in §4.
  A strict AND-threshold on name similarity alone would drop these true positives
  during blocking.

## 7. Key takeaways for pipeline design

1. **Blocking must be address-anchored as well as name-anchored** (e.g., street-number
   + city/locality keys, or an OR-combination of name-token and address-token
   inverted indexes) so name-garbled-but-address-identical true matches survive into
   the candidate set.
2. **Country-specific address normalization** is worth building separately for
   US-style and India-style addresses; treat `country` as an open label so France (test-only)
   degrades gracefully to a generic normalizer.
3. **Transliteration handling** needed for the ~13% of Source 2/3 names in Devanagari/
   Tamil script - either a rule-based transliteration step, or lean on address
   features when the name script doesn't match Source 1 (always Latin).
4. Matching-model features should include both name and address similarity
   (Jaccard + edit-ratio, at minimum), and the candidate-survival rule during blocking
   should not require both signals to be strong simultaneously.
5. Given F_0.5's precision weighting (2× precision vs. recall) and the low-similarity
   true-match tail, the final matching threshold should be tuned conservatively but
   informed by both signals - requiring both name AND address to be weak before
   rejecting a candidate, rather than rejecting on either being weak alone.

## 8. Addendum (repo setup, full-file checks)

- **Test country mix differs from train**: test S1 is India 46.8%, US 38.3%,
  France 15.0% (train S1: US 60%, India 40%). France has no postcodes in addresses and
  is concentrated in a handful of cities (Bordeaux, Nantes, Lille, Roubaix...), so
  France blocks will be dense.
- **More scripts than Devanagari/Tamil**: Gurmukhi, Kannada, Telugu and Bengali also
  appear in names, and Indian state names appear in native script inside addresses
  (`बिहार`, `తెలంగాణ`, `পশ্চিমবঙ্গ`).
- **Placeholder addresses**: literal `None` / `<NULL>` strings occur; `convert` maps
  them to null.
- **Name junk**: prefixes/suffixes like `***`, `--`, `#99067`, `| www.site.com`,
  `(Limited)`, `[Limited]`.
- **Test names contain literal `"`** - TSVs must be read with quoting disabled.
- **Each S2/S3 record matches at most one S1**; **matched pairs always share country**.
