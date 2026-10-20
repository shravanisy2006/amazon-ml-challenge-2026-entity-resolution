"""
decide.py — Step 5: turn match probabilities into final match lists.

Two rules, both tuned on held-out training entities for the official F0.5:
  1. keep a pair only if its probability >= threshold (F0.5 rewards caution)
  2. one-to-one: an S2/S3 record describes one business, so if it passes the
     threshold for several S1 records, keep only its most probable S1.
"""
import numpy as np
import pandas as pd

def decide(s1_ids, cand_ids, probs, threshold, one_to_one=True):
    df = pd.DataFrame({"s1_id": s1_ids, "cand_id": cand_ids, "p": probs})
    df = df[df.p >= threshold]
    if one_to_one and len(df):
        df = df.sort_values("p", ascending=False).drop_duplicates("cand_id")
    return df

def to_lists(df):
    return df.groupby("s1_id").cand_id.apply(set).to_dict()
