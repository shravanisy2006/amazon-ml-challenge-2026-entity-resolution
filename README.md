# Business Entity Resolution: Amazon ML Challenge 2026

Matching noisy business records across three independent data sources, with no shared IDs, using only names and addresses.

**Public leaderboard: F0.5 = 0.969**, built during the 72-hour Amazon ML Challenge 2026 hackathon on Unstop. The full pipeline runs on a 16 GB laptop, CPU only, in about 4–5 hours.

---

## The problem

Three sources list the same real-world businesses, each written differently: abbreviations, typos, reordered addresses, trade names, and names in Indian scripts (Devanagari, Tamil, Kannada and more). Source 1 is a deduplicated reference list of ~1.7M businesses (test). The task: for each one, find every matching record among ~10M records in Sources 2 and 3.

What makes it hard:
- **Decoys.** About 26% of Source 2/3 records match no business. Most are "siblings" of a real one: same name core, same street, but a different building number or legal form.
- **Transliteration.** "इस्टर्न इन्फ्रा" has to match "Eastern Infra".
- **An unseen country.** The test set includes France, which never appears in training.
- **Scale.** 12.5M training records and 11.7M test records on a normal laptop.

Scoring is macro **F0.5** per Source 1 business, which weights precision twice as much as recall.

## Approach

```
clean ─► partition ─► candidate search ─► learned ranker ─► pair features ─► LightGBM ─► decision rule
         (country +    (5 search keys)     (top 15 per      (79 per pair)               (threshold +
          region)                          business)                                    one-to-one)
```

1. **Cleaning:** normalise names and addresses, transliterate Indian scripts to Latin, split DBA/trade names, detect states.
2. **Partitioning:** compare records only within the same country and region. French regions are learned from the data itself.
3. **Candidate search:** exact TF-IDF similarity on name + address combined (`sparse_dot_topn`), a "sound skeleton" search for transliterated names ("istarn inphra" → "strn nfr" ← "eastern infra"), compressed-vector search with FAISS, and shared building numbers.
4. **Learned ranker:** a small LightGBM model keeps the 15 best candidates per business. That's higher recall than a fixed rule keeping 30.
5. **Features:** 79 per pair, including name and address similarity, building numbers kept or replaced (the main decoy signal), legal-form agreement, and how the record is written.
6. **Matcher:** LightGBM binary classifier, threshold tuned directly on F0.5, and a one-to-one rule (each record belongs to at most one business).

Full methodology, measurements and ablations: [`METHODOLOGY.md`](METHODOLOGY.md).

## Results

| Version | Main change | Public F0.5 |
|---|---|---|
| 1 | Baseline: compressed-vector blocking | 0.897 |
| 2 | Exact name + address search | 0.936 |
| 3 | Decoy-aware features, sound-skeleton search | 0.953 |
| 4 | Learned trimming of candidates (50 → 15 per business) | 0.956 |
| 5 | Record-format features, more training data | **0.969** |

## Repository structure

```
business_entity_resolution/
├── README.md            detailed run instructions
├── requirements.txt     pinned dependencies
└── src/
    ├── run_all.py       runs the whole pipeline (8 steps)
    ├── preprocess.py    cleaning (uses normalize.py)
    ├── geo.py           learned regions
    ├── blocking.py      candidate search + learned ranker
    ├── features.py      pair features
    ├── train.py         LightGBM matcher + threshold tuning
    ├── predict.py       writes the answer files
    ├── decide.py        decision rule
    ├── evaluate.py      local F0.5 scorer
    ├── diagnose.py      per-country sanity check
    ├── make_dev_sample.py  small aligned sample for fast experiments
    └── config.py        paths and settings
```

## How to run

**The challenge data is not included.** It belongs to the organisers. Place it like this, next to the `business_entity_resolution` folder:

```
dataset/train/   train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
dataset/test/    test_source1.tsv, test_source2.tsv, test_source3.tsv
```

Then, with Python 3.11 or 3.12:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r business_entity_resolution/requirements.txt
cd business_entity_resolution/src
python run_all.py
```

The answers appear in `output/matching_results.tsv` and `output/candidate_pairs.tsv`. An interrupted run can resume with `python run_all.py --from <step>`.

## Tech stack

Python · pandas / PyArrow · scikit-learn · LightGBM · FAISS · sparse_dot_topn · RapidFuzz · indic_transliteration

All libraries are open source (MIT, BSD or Apache-2.0). No external data, APIs or lookups are used.
