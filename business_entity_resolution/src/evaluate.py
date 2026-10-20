"""
evaluate.py — the official metric, reimplemented for local validation.

Macro F0.5 over Source-1 entities. Singletons (no true matches) score 1.0
for an empty prediction and 0.0 for any prediction.
"""
import pandas as pd

def _parse(cell) -> set:
    if cell is None or (isinstance(cell, float) and cell != cell):
        return set()
    return {x.strip() for x in str(cell).split(",") if x.strip()}

def f05_single(pred: set, true: set, beta: float = 0.5) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)

def score(pred: dict, truth: dict) -> dict:
    """pred/truth: {s1_id: set(ids)}. Scored over the ids in `truth`."""
    scores, single, multi = [], [], []
    for s1, true in truth.items():
        s = f05_single(pred.get(s1, set()), true)
        scores.append(s)
        (single if not true else multi).append(s)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {"f05": mean(scores), "f05_singletons": mean(single),
            "f05_non_singletons": mean(multi), "n": len(scores),
            "n_singletons": len(single)}

def load_id_lists(path: str, col: str) -> dict:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {r["source1_entity_id"]: _parse(r[col]) for _, r in df.iterrows()}

def blocking_report(cands, truth: dict, n_s23: int) -> dict:
    """Recall ceiling of a candidate set (cands: DataFrame s1_id, cand_id)."""
    got = cands.groupby("s1_id").cand_id.apply(set).to_dict()
    hit = tot = full = ns = 0
    for s1, true in truth.items():
        if not true:
            continue
        ns += 1
        h = len(true & got.get(s1, set()))
        hit += h; tot += len(true); full += (h == len(true))
    n_pairs = len(cands)
    return {"pair_recall": hit / max(tot, 1), "entity_full_recall": full / max(ns, 1),
            "avg_candidates": n_pairs / max(len(truth), 1),
            "reduction_ratio": 1 - n_pairs / max(len(truth) * n_s23, 1)}

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit("usage: python evaluate.py predictions.tsv ground_truth.tsv")
    pred = load_id_lists(sys.argv[1], "matched_entity_ids")
    truth = load_id_lists(sys.argv[2], "matched_entity_ids")
    print(score(pred, truth))
