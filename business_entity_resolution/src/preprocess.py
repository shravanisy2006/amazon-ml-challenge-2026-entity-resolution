"""
preprocess.py — clean every record once and cache it.

    python preprocess.py --split train          # all three training sources
    python preprocess.py --split test
    python preprocess.py --split train --nrows 50000   # quick dev run

Output: WORK_DIR/<split>_<source>.parquet  (one table per source)
Set-valued fields are stored as space-joined strings to keep memory low.
"""
import argparse, os, time
from multiprocessing import Pool
import pandas as pd
import config
from normalize import clean_name, clean_address, clean_country

COLS = ["entity_id", "src", "country", "name_raw", "addr_raw",
        "name_norm", "name_core", "name_compact", "name_alias", "name_is_url",
        "name_script", "addr_norm", "addr_tokens", "addr_nums", "state", "has_addr",
        "place_labels"]

# the only columns the blocking stage needs (keeps its memory low)
BLOCK_COLS = ["entity_id", "country", "state", "name_compact", "name_alias",
              "addr_tokens", "addr_nums", "place_labels", "name_core", "name_script"]

def _clean_rows(rows):
    out = []
    for eid, name, addr, country in rows:
        c = clean_country(country)
        n = clean_name(name)
        a = clean_address(addr, c)
        out.append((eid, eid[:2], c, name, addr,
                    n["norm"], n["core"], n["compact"], n["alias"], n["is_url"],
                    n["script"], a["norm"], " ".join(sorted(a["tokens"])),
                    " ".join(sorted(a["nums"])), a["state"], bool(a["norm"]),
                    "|".join(dict.fromkeys(a["labels"]))))
    return out

def _compact(df):
    """Arrow-backed strings use far less memory than Python string objects."""
    for col in df.columns:
        if df[col].dtype != bool:
            df[col] = df[col].astype("str")      # pyarrow-backed in pandas >= 3
    return df

def clean_file(path, nrows=None, workers=config.N_WORKERS,
               chunk=config.PREPROCESS_CHUNK, sub=20_000):
    """
    Stream the file in chunks so memory never holds more than one chunk of
    raw rows; each cleaned chunk is stored compactly before the next is read.
    """
    frames, done, t0 = [], 0, time.time()
    reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                         chunksize=chunk, nrows=nrows)
    with Pool(workers) as pool:
        for df in reader:
            rows = list(zip(df.entity_id, df.business_name, df.business_address, df.country))
            del df
            batches = [rows[i:i + sub] for i in range(0, len(rows), sub)]
            parts = pool.map(_clean_rows, batches)
            frames.append(_compact(pd.DataFrame([r for p in parts for r in p], columns=COLS)))
            done += len(rows)
            del rows, batches, parts
            print(f"    {os.path.basename(path)}: {done:,} rows cleaned ({time.time()-t0:.0f}s)", flush=True)
    return pd.concat(frames, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--nrows", type=int, default=None)
    args = ap.parse_args()
    files = config.TRAIN_FILES if args.split == "train" else config.TEST_FILES
    config.check_files({k: v for k, v in files.items() if k != "gt"})
    os.makedirs(config.WORK_DIR, exist_ok=True)
    for src in ("s1", "s2", "s3"):
        t0 = time.time()
        df = clean_file(files[src], nrows=args.nrows)
        out = os.path.join(config.WORK_DIR, f"{args.split}_{src}.parquet")
        df.to_parquet(out, index=False)
        print(f"{src}: {len(df):,} rows in {time.time()-t0:.1f}s -> {out}")

if __name__ == "__main__":
    main()
