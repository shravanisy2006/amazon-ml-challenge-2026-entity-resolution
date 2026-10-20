"""
diagnose.py — per-country summary of the answer files.

    python diagnose.py

In training, every country looks the same: 94.4% of S1 businesses have at
least one match and there are 3.46 matches per business on average. Since
precision is high, a country whose predicted numbers fall well below these is
losing recall there. This matters most for France, which has no training labels.
"""
import os
import pandas as pd
import config

TRAIN_SHARE_MATCHED, TRAIN_AVG_MATCHES = 0.944, 3.46

def counts(path, col):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return dict(zip(df.source1_entity_id, df[col].map(lambda v: len([x for x in v.split(",") if x]))))

def main():
    s1 = pd.read_parquet(os.path.join(config.WORK_DIR, "test_s1.parquet"), columns=["entity_id", "country"])
    s1["matches"] = s1.entity_id.map(counts(os.path.join(config.OUTPUT_DIR, "matching_results.tsv"), "matched_entity_ids"))
    s1["cands"] = s1.entity_id.map(counts(os.path.join(config.OUTPUT_DIR, "candidate_pairs.tsv"), "candidate_entity_ids"))
    print(f"training reference: {TRAIN_SHARE_MATCHED:.1%} of shops matched, {TRAIN_AVG_MATCHES:.2f} matches per shop\n")
    print(f"{'country':10s} {'shops':>10s} {'matched':>9s} {'matches/shop':>13s} {'vs training':>12s} {'candidates/shop':>16s}")
    for c, g in s1.groupby("country"):
        m = g.matches.mean()
        print(f"{c:10s} {len(g):>10,} {(g.matches > 0).mean():>9.1%} {m:>13.2f} {m / TRAIN_AVG_MATCHES:>11.0%} {g.cands.mean():>16.1f}")

if __name__ == "__main__":
    main()
