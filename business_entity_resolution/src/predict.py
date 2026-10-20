"""
predict.py — Step 6: score the test pairs and write output/matching_results.tsv.

    python predict.py

Reads the test feature parts one at a time (memory stays low), applies the
model and the tuned decision rule, and writes one row for every test S1
record (empty list when no match). Every match is a subset of the candidates.
"""
import json, os, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import config
from features import FEATURES
from decide import decide
from blocking import write_id_lists

def main():
    t0 = time.time()
    rule = json.load(open(os.path.join(config.WORK_DIR, "decision.json")))
    if "n_models" in rule:        # ensemble: average the judges' probabilities
        models = [lgb.Booster(model_file=os.path.join(config.WORK_DIR, f"model_{i}.txt"))
                  for i in range(rule["n_models"])]
    else:                         # single judge written by an earlier version
        models = [lgb.Booster(model_file=os.path.join(config.WORK_DIR, "model.txt"))]
    th, oto = rule["threshold"], rule["one_to_one"]
    d = os.path.join(config.WORK_DIR, "test_features")
    keep = []
    for part in sorted(os.listdir(d)):
        f = pd.read_parquet(os.path.join(d, part))
        p = np.mean([m.predict(f[FEATURES]) for m in models], axis=0)
        m = p >= th                               # anything below can never be chosen
        keep.append(pd.DataFrame({"s1_id": f.s1_id.values[m], "cand_id": f.cand_id.values[m], "p": p[m]}))
        print(f"  {part}: {len(f):,} pairs, {int(m.sum()):,} above threshold ({time.time()-t0:.0f}s)", flush=True)
    kept = pd.concat(keep, ignore_index=True)
    final = decide(kept.s1_id.values, kept.cand_id.values, kept.p.values, th, oto)
    all_s1 = pd.read_parquet(os.path.join(config.WORK_DIR, "test_s1.parquet"),
                             columns=["entity_id"]).entity_id.tolist()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, "matching_results.tsv")
    final = final.sort_values("s1_id", kind="stable")       # group each S1's matches
    write_id_lists(final.s1_id, final.cand_id, all_s1, path, "matched_entity_ids")
    n_matched = final.s1_id.nunique()
    print(f"{len(final):,} matches; {n_matched:,} of {len(all_s1):,} S1 records have at least one")
    print(f"saved {path} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
