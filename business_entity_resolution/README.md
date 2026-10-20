# Business Entity Resolution — Amazon ML Challenge

Matches Source 2 / Source 3 business records to deduplicated Source 1 entities.
The pipeline is fully local: no external data, APIs or lookups are used.

```
data (.tsv) ─► preprocess ─► blocking ─► features ─► matcher ─► threshold/singletons ─► output/
               (Step 1)      (Step 2)    (Step 3)    (Step 4)      (Step 5)            (Step 6)
```

| Step | Module | Status |
|---|---|---|
| 1 Cleaning & normalization | `src/normalize.py`, `src/preprocess.py` | done |
| 2 Candidate generation (blocking) | `src/geo.py`, `src/blocking.py` | done |
| 3 Pairwise features | `src/features.py` | done |
| 4 Matching model | `src/train.py` | done |
| 5 Threshold tuning, one-to-one rule | `src/train.py`, `src/decide.py` | done |
| 6 Test inference + outputs | `src/predict.py` | done |

## Folder layout

```
AMAZON_ML/                          <- open this folder in VS Code
├── dataset/
│   ├── train/   train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   └── test/    test_source1.tsv, test_source2.tsv, test_source3.tsv
├── code/business_entity_resolution/
│   ├── src/            all pipeline code (run commands from here)
│   ├── README.md
│   └── requirements.txt
├── output/             matching_results.tsv, candidate_pairs.tsv (final deliverables)
└── work/               cache created by the pipeline (not submitted)
```

Paths are found automatically: `config.py` walks up from `src/` to the first folder
containing `dataset/`. To use other locations, set `BER_TRAIN_DIR`, `BER_TEST_DIR`,
`BER_WORK_DIR` or `BER_OUTPUT_DIR`.

## Setup

Python 3.11 or 3.12 (the pinned pandas/numpy versions need at least 3.11).

```powershell
cd code\business_entity_resolution
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Running

All commands run from `code/business_entity_resolution/src`.

```powershell
python run_all.py            # data -> output/ in one go (8 steps)
python run_all.py --from 6   # resume from a step if one was interrupted
```

| Step | Script | Produces |
|---|---|---|
| 1, 2 | `preprocess.py --split train/test` | `work/<split>_s*.parquet` cleaned records |
| 3 | `blocking.py --split train --sample-partitions 0.3` | training candidates (30% of states, whole) |
| 4 | `features.py --split train` | `work/train_features/` with labels |
| 5 | `train.py` | `work/model.txt`, `work/decision.json` |
| 6 | `blocking.py --split test` | `output/candidate_pairs.tsv` |
| 7 | `features.py --split test` | `work/test_features/` |
| 8 | `predict.py` | `output/matching_results.tsv` |

Check the outputs with the challenge validator (from the project root):
```powershell
python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test
```

### Fast development loop

A random slice of the files rarely contains matching records, so use an aligned sample:

```powershell
python make_dev_sample.py --n-s1 3000 --out ..\..\..\dev_data
$env:BER_TRAIN_DIR = "..\..\..\dev_data"; $env:BER_WORK_DIR = "..\..\..\work_dev"
python preprocess.py --split train
python blocking.py --split train
Remove-Item Env:BER_TRAIN_DIR, Env:BER_WORK_DIR    # back to the full data
```
(macOS/Linux: `BER_TRAIN_DIR=../../../dev_data BER_WORK_DIR=../../../work_dev python preprocess.py --split train`)

## Method summary

**Cleaning (`normalize.py`).** Unicode NFKC, accent stripping, Indic-script names
transliterated to Latin (legal words such as प्राइवेट लिमिटेड mapped directly), DBA/AKA/FKA
names split into primary + alias, URL/handle names reduced to their stem, junk removed
(IDs, phone numbers, `--`/`>>` prefixes, honorifics), leetspeak repaired (`G1obal`),
legal forms removed for the *core* name. Addresses: abbreviations expanded, ordinals and
leading zeros normalized, PO Box / `null` removed, state recognized from whole
comma-separated components (country-scoped tables). Country is an open set; unknown
countries (France) receive the same generic cleaning.

**Region learning (`geo.py`).** Blocking compares records only within the same
(country, region). US and Indian states come from lookup tables, but France (test only)
has none, and many S2/S3 records name only a city or district. Regions are therefore
also *learned from the data*: label A contains label B when almost every record naming B
also names A (e.g. every S1 "Bordeaux" record says "Nouvelle-Aquitaine"); large labels with
no container are regions; S2/S3 labels such as départements ("Gironde") are then mapped
through the records they co-occur with. On the French test records this finds 3 regions
and assigns one to 97% of S2/S3 records (the rest have empty addresses). Validation: with
India's state table deleted, learned regions reach 92.6% pair recall vs 93.2% with the
table and 90.4% with no partitioning.

**Blocking (`blocking.py`).** Records are compared only within the same (country, region)
partition (Telangana/Andhra Pradesh share one, as the data mixes them). Five keys:

- **E, exact joint search (main key).** Every S2/S3 record finds its 5 most similar S1 records
  in its partition by exact TF-IDF cosine(name) + cosine(address): character n-grams, no
  compression, computed with `sparse_dot_topn` (Apache-2.0) without building the full score
  matrix. N-grams found in more than 5% of records are ignored here: they carry little
  information but make the product nearly dense (3.6x faster, -0.5% recall in Delhi). Searching from the S2/S3 side fits the data: in training, every S2/S3 record belongs
  to at most one S1 business, and 26% belong to none (decoys).
- **A/B, compressed name and address search.** TF-IDF → 128-d SVD → FAISS, in both directions.
- **S, sound skeleton.** S2/S3 names written in an Indic script are searched again by a
  consonant skeleton of the name plus address: sound-alike letters merged, vowels dropped,
  doubles collapsed, so "istarn inphra" and "eastern infra" both become "strn nfr".
  Indic-script records whose true S1 was in the top 5 rose from 80% to 91% in Delhi.
- **C, shared address numbers** (≥ 3 digits).
- **D, no-state rescue.** S2/S3 records without a parseable region are matched by name
  across their country.

Candidates are ranked per S1 record (exact-key pairs first, then keys agreeing and similarity)
and trimmed to 15 per S1 by a learned ranker: a small LightGBM trained in the training blocking
step (pass 1 collects examples from S1 records in the answer key, pass 2 trims) on cheap signals,
including how far a pair is behind the S1 record's best candidate and whether the S1 is the
candidate record's best option. In Delhi the searches find 98.0% of true pairs; the fixed formula
kept 97.7% of those at 30 per S1, the ranker keeps 99.0% at 15. Saved as work/ranker.txt and
reused on the test split. Memory is bounded by the largest partition; the test split runs one country
at a time.

Why the exact key: in Delhi (121k S1 records, the most crowded partition), the true S1
record was the single closest match for 89% of S2/S3 records with exact name+address
similarity, versus 74% with the compressed vectors.

**Matcher and decision (`train.py`, `decide.py`, `predict.py`).** LightGBM (MIT licence,
a gradient-boosted tree model with no pretrained weights) is trained on the labelled
candidate pairs, holding out 20% of S1 entities with all their pairs for early stopping and
tuning. The decision rule is tuned directly on the official macro F0.5 of the held-out
entities, singletons included: a probability threshold, and optionally a one-to-one rule
(an S2/S3 record is assigned only to its most probable S1). Test pairs are scored part by
part; every S1 record gets a row, and matches are always a subset of the candidates.

**Decoys.** In training, 26% of S2/S3 records belong to no S1 business. Most are
"siblings" of a real business: the same name core and street, but a different building
number and often a different legal form ("Private Limited" vs "Limited"). True matches
keep the business's numbers (sometimes adding a prefix such as "BLOCK C-622" or dropping a
digit), while decoys replace them. In Delhi, the business's numbers were still present in
86% of true matches but only 11% of look-alike decoys, so the features compare numbers in
both directions with that tolerance, and compare legal forms as equal / shortened /
conflicting.

**Features (`features.py`).** 79 numbers per candidate pair. Name: fuzzy ratios
(RapidFuzz) on the core name, its space-free form and the DBA alias; IDF-weighted word
overlap, so a shared rare word counts more than a shared "Global"; typo-tolerant word
coverage in both directions; legal-form agreement. Address: word overlap, fuzzy ratio,
shared building/PIN numbers, shared place labels. Blocking signals. Context: how the pair
ranks among its S1 record's candidates and among the candidate's S1 options (each S2/S3
record belongs to at most one business). The country label is never a feature, so France
is handled like any other country. Features are streamed to disk in parts, so the ~69M
test pairs never need to fit in memory.

## Validation results (aligned dev sample)

3,000 scored S1 entities + 65k S1 and 310k S2/S3 distractors.

| Stage | Metric | Value |
|---|---|---|
| Blocking | pair recall (cap 40) | 96.5% |
| Blocking | pair recall (cap 80) | 97.6% |
| Blocking | reduction ratio | 99.99% |
| Blocking | peak RAM | 1.1 GB |
| Full pipeline (train.py) | F0.5 on 600 held-out entities | 0.974 |

Blocking recall on a quarter of Delhi (30k S1, 208k S2/S3, the hardest case):

| Blocking | Pair recall | Entity full recall |
|---|---|---|
| Compressed search only (v1, leaderboard 0.897) | 88.7% | 71.8% |
| + exact joint search (leaderboard 0.936) | 95.2% | 87.1% |

Matcher on a quarter of Delhi (20% of S1 records held out, official macro F0.5):

| Version | Candidates per S1 | Pair recall | F0.5 |
|---|---|---|---|
| Exact search, original features (leaderboard 0.936) | 50 | 94.8% | 0.9553 |
| + number / legal-form features | 30 | 93.9% | 0.9623 |
| + sound-skeleton search and features (leaderboard 0.953) | 30 | 95.9% | 0.9705 |
| + learned trimming (leaderboard 0.956) | 15 | 97.7% | 0.9834 |
| + record-format features (current; trained on 30% of states) | 15 | 97.7% | 0.9851 |

Record-format features describe how the S2/S3 record is written (word count, casing,
punctuation, address digits and commas) and how that differs from the S1 record. From the
record alone, decoys can be told from real records with AUC 0.76 on training data.

Full-scale blocking (15% of training states, 313,865 S1 / 1.8M S2+S3): 5.8 min on a
16 GB laptop.

Per country: India 93.2% pair recall (Indic-script names are the hardest), US higher.

Remaining blocking misses are mainly Indic-script names whose address is also partial,
S2/S3 records with neither address nor state, and invented DBA names with no shared text.
