"""Central configuration: paths and parameters. Edit DATA_DIR for your machine."""
import os
from pathlib import Path

def _find_root() -> Path:
    """
    Project root = the nearest folder (walking up from this file) that contains
    a `dataset/` folder. Works both for the submission layout
    (root/code/business_entity_resolution/src/config.py) and a flat folder.
    """
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "dataset").is_dir():
            return p
    return here.parents[2] if len(here.parents) > 2 else here

ROOT = _find_root()

# Override any of these with environment variables if your data lives elsewhere.
TRAIN_DIR  = Path(os.environ.get("BER_TRAIN_DIR", ROOT / "dataset" / "train"))
TEST_DIR   = Path(os.environ.get("BER_TEST_DIR", ROOT / "dataset" / "test"))
WORK_DIR   = str(os.environ.get("BER_WORK_DIR", ROOT / "work"))      # cache, not submitted
OUTPUT_DIR = str(os.environ.get("BER_OUTPUT_DIR", ROOT / "output"))

TRAIN_FILES = {
    "s1": str(TRAIN_DIR / "train_source1.tsv"),
    "s2": str(TRAIN_DIR / "train_source2.tsv"),
    "s3": str(TRAIN_DIR / "train_source3.tsv"),
    "gt": str(TRAIN_DIR / "train_ground_truth.tsv"),
}
TEST_FILES = {
    "s1": str(TEST_DIR / "test_source1.tsv"),
    "s2": str(TEST_DIR / "test_source2.tsv"),
    "s3": str(TEST_DIR / "test_source3.tsv"),
}

def check_files(files: dict):
    """Fail early with a readable message instead of a pandas traceback."""
    missing = [p for p in files.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit("Missing input files:\n  " + "\n  ".join(missing) +
                         f"\nProject root detected as: {ROOT}\n"
                         "Put the data in <root>/dataset/train and <root>/dataset/test, "
                         "or set BER_TRAIN_DIR / BER_TEST_DIR.")

PREPROCESS_CHUNK = 200_000       # rows cleaned at a time (lower it if memory is tight)

RANDOM_SEED = 42
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# ─── blocking (Step 2) ───
# Vectors: TF-IDF char n-grams -> TruncatedSVD. Lower SVD_DIM / raise SVD_FIT_SAMPLE
# trade-offs only affect blocking recall, never correctness.
SVD_DIM = 128
SVD_FIT_SAMPLE = 300_000        # records used to fit the vectorizers + SVD
TOPK_NAME = 30                  # nearest S2/S3 by name, per S1, within partition
TOPK_ADDR = 30                  # nearest S2/S3 by address, per S1, within partition
TOPK_REVERSE = 5                # nearest S1 for each S2/S3 record (name and address)
TOPK_NOSTATE = 20               # nearest S1 for each S2/S3 record lacking a state
MIN_SIM = 0.25                  # cosine floor for vector neighbours
NUM_MAX_POSTINGS = 200          # skip address numbers shared by more records
MAX_CANDIDATES = 15             # hard cap per S1 after merging (keeps best-scored)
QUERY_BATCH = 4096
HNSW_MIN_SIZE = 50_000          # partitions at least this big use HNSW (approximate)
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 80
HNSW_EF_SEARCH = 64

# ─── features (Step 3) ───
FEATURE_CHUNK = 250_000         # pairs per feature part (lower it if memory is tight)

# ─── exact joint search (blocking key E) ───
TOPK_EXACT = 5                  # S1 records kept per S2/S3 record
EXACT_MIN_SIM = 0.2             # floor on cos(name) + cos(address) (range 0..2)
EXACT_QUERY_BATCH = 50_000      # S2/S3 records per sparse multiplication (progress step)
EXACT_MAX_DF = 0.05             # ignore n-grams found in more than 5% of records (speed)
N_THREADS = os.cpu_count() or 1

# ─── learned trimming (ranker) ───
RANKER_SAMPLE = 0.3             # share of training S1 records whose pairs train the ranker
RANKER_MAX_ROWS = 8_000_000     # cap on ranker training rows (memory)
