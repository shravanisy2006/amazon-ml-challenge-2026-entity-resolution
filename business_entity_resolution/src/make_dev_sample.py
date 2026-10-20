"""
make_dev_sample.py — build a small, *aligned* training sample for fast iteration.

Random slices of the three files rarely contain matching records, so this
picks N ground-truth S1 entities, pulls all their true S2/S3 matches, and
adds M random S2/S3 records as distractors.

    python make_dev_sample.py --n-s1 3000 --n-distract 300000 --out dev_data
    then run preprocess/blocking with BER_TRAIN_DIR pointing at the sample
    (see README for the PowerShell syntax)
"""
import argparse, os
import pandas as pd
import config

def _pull(path, ids, chunk=500_000):
    parts, found = [], set()
    for c in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk):
        h = c[c.entity_id.isin(ids)]
        if len(h):
            parts.append(h); found |= set(h.entity_id)
        if len(found) == len(ids):
            break
    return pd.concat(parts) if parts else pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=3000)
    ap.add_argument("--n-distract", type=int, default=300_000)
    ap.add_argument("--n-s1-distract", type=int, default=65_000,
                    help="extra S1 records (not scored) so S1 is realistically dense")
    ap.add_argument("--out", default="dev_data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    f = config.TRAIN_FILES
    config.check_files(f)
    gt = pd.read_csv(f["gt"], sep="\t", dtype=str, keep_default_na=False) \
           .sample(a.n_s1, random_state=config.RANDOM_SEED)
    true_ids = {x for v in gt.matched_entity_ids for x in v.split(",") if x}
    gt.to_csv(os.path.join(a.out, "train_ground_truth.tsv"), sep="\t", index=False)
    s1_pos = _pull(f["s1"], set(gt.source1_entity_id))
    s1_neg = pd.read_csv(f["s1"], sep="\t", dtype=str, keep_default_na=False,
                         nrows=a.n_s1_distract)
    pd.concat([s1_pos, s1_neg]).drop_duplicates("entity_id").to_csv(
        os.path.join(a.out, "train_source1.tsv"), sep="\t", index=False)
    for src in ("s2", "s3"):
        want = {i for i in true_ids if i.startswith(src.upper() + "-")}
        pos = _pull(f[src], want)
        neg = pd.read_csv(f[src], sep="\t", dtype=str, keep_default_na=False,
                          nrows=a.n_distract // 2)
        pd.concat([pos, neg]).drop_duplicates("entity_id").to_csv(
            os.path.join(a.out, f"train_source{src[1]}.tsv"), sep="\t", index=False)
    print("dev sample written to", a.out)

if __name__ == "__main__":
    main()
