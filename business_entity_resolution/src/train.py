"""
train.py — Step 4 (+ Step 5 tuning): learn the matcher and choose the decision rule.

    python train.py

Reads work/train_features/, holds out 20% of S1 entities (all of an entity's
pairs stay together), trains LightGBM (MIT licence) with early stopping, then
picks the probability threshold and one-to-one rule that maximise the official
macro F0.5 on the held-out entities, singletons included. Saves
work/model.txt and work/decision.json.
"""
import json, os, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import config
from features import load_features, FEATURES
from evaluate import load_id_lists, score
from decide import decide, to_lists

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=config.RANDOM_SEED)

# Ensemble: N_MODELS judges trained with different random seeds (row and feature
# sampling differ), probabilities averaged. Set to 1 for a single judge.
N_MODELS = 1

def best_rule(val, truth, table=False):
    """Threshold and one-to-one choice that maximise the official macro F0.5."""
    best = None
    if table:
        print("threshold  one-to-one  F0.5    singletons  others")
    for oto in (False, True):
        for th in np.arange(0.30, 0.96, 0.05):
            kept = decide(val.s1_id.values, val.cand_id.values, val.p.values, th, oto)
            r = score(to_lists(kept), truth)
            if table:
                print(f"  {th:.2f}      {str(oto):5s}     {r['f05']:.4f}  {r['f05_singletons']:.4f}      {r['f05_non_singletons']:.4f}")
            if best is None or r["f05"] > best[0]:
                best = (r["f05"], float(round(th, 2)), oto)
    return best

def main():
    t0 = time.time()
    df = load_features("train")
    print(f"{len(df):,} pairs, {int(df.y.sum()):,} true ({time.time()-t0:.0f}s)")
    truth_all = load_id_lists(config.TRAIN_FILES["gt"], "matched_entity_ids")
    ids = np.array(sorted(set(df.s1_id)))
    rng = np.random.default_rng(config.RANDOM_SEED)
    val_ids = set(rng.choice(ids, int(0.2 * len(ids)), replace=False))
    is_val = df.s1_id.isin(val_ids).values

    dtr = lgb.Dataset(df.loc[~is_val, FEATURES], df.y[~is_val], free_raw_data=True)
    dva = lgb.Dataset(df.loc[is_val, FEATURES], df.y[is_val], reference=dtr)
    Xv = df.loc[is_val, FEATURES]
    val = df.loc[is_val, ["s1_id", "cand_id"]].reset_index(drop=True)
    # held-out truth includes S1 entities with no candidates and true singletons
    truth = {k: truth_all.get(k, set()) for k in val_ids}
    models, preds = [], []
    for i in range(N_MODELS):
        print(f"training LightGBM judge {i+1}/{N_MODELS} ...", flush=True)
        model = lgb.train(dict(PARAMS, seed=config.RANDOM_SEED + i), dtr, num_boost_round=2000,
                          valid_sets=[dva], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        models.append(model)
        preds.append(model.predict(Xv, num_iteration=model.best_iteration))
        print(f"judge {i+1}: best iteration {model.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
    if N_MODELS > 1:
        single = best_rule(val.assign(p=preds[0]), truth)
        print(f"single judge:  F0.5 = {single[0]:.4f}")
    val["p"] = np.mean(preds, axis=0)
    best = best_rule(val, truth, table=True)
    print(f"BEST: F0.5 = {best[0]:.4f} at threshold {best[1]}, one-to-one = {best[2]}")

    for i, m in enumerate(models):
        m.save_model(os.path.join(config.WORK_DIR, f"model_{i}.txt"), num_iteration=m.best_iteration)
    with open(os.path.join(config.WORK_DIR, "decision.json"), "w") as fh:
        json.dump({"threshold": best[1], "one_to_one": best[2], "val_f05": best[0],
                   "n_models": N_MODELS, "features": FEATURES}, fh, indent=2)
    model = models[0]
    imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("top features:", ", ".join(imp.index[:8]))
    print(f"saved model and decision rule ({time.time()-t0:.0f}s total)")

if __name__ == "__main__":
    main()
