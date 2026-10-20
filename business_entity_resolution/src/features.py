"""
features.py — Step 3: describe every candidate pair with numbers.

    python features.py --split train     # needs work/train_candidates.pkl
    python features.py --split test

Output: WORK_DIR/<split>_features/part_*.parquet
        one row per (s1_id, cand_id) with 47 float features, plus the label
        `y` (1 = same business) for the training split. Written in parts so
        the 69M test pairs never have to fit in memory at once.

Feature groups
  name      fuzzy ratios on the core name, compact form, DBA alias; IDF-weighted
            word overlap (a shared rare word like "Orellana" counts far more
            than a shared "Global"); typo-tolerant word coverage; legal form
  address   word overlap, fuzzy ratio, shared building/PIN numbers, shared
            place labels (city/district)
  blocking  similarities and which keys found the pair
  context   how this pair compares with the S1 record's other candidates,
            and with the candidate's other S1 options. Each S2/S3 record
            belongs to at most one S1 business, so being the candidate's best
            option is strong evidence.
No feature uses the country label, so unseen countries (France) are treated
exactly like known ones.
"""
import argparse, math, os, time
from collections import Counter
from multiprocessing import Pool
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import config
from normalize import LEGAL_WORDS, sound_skeleton

REC_COLS = ["entity_id", "name_norm", "name_core", "name_compact", "name_alias",
            "name_is_url", "name_script", "addr_norm", "addr_tokens", "addr_nums",
            "has_addr", "place_labels", "name_raw", "addr_raw"]

FEATURES = [
    # name
    "n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw_compact", "n_ratio_compact",
    "n_alias_tset", "n_best_tset", "n_idf_jacc", "n_soft_cov1", "n_soft_cov2",
    "n_first_eq", "n_len1", "n_len2", "n_len_diff", "n_legal_match", "n_is_url2",
    "n_nonlatin2", "n_has_alias2", "n_exact_core",
    # legal form relation (decoys often change it: private -> public, llp -> limited)
    "n_legal_equal", "n_legal_subset", "n_legal_conflict", "n_legal_missing",
    "n_pub_priv_conflict", "n_filler_diff",
    # sound skeleton similarity (Indic-script names vs English spellings)
    "n_skel_ratio", "n_skel_tset",
    # address
    "a_has1", "a_has2", "a_tok_jacc", "a_tset", "a_ratio", "a_num_shared",
    "a_num_jacc", "a_num_cov1", "a_num_cov2", "a_n_nums1", "a_n_nums2", "a_label_jacc",
    # building numbers, typo/prefix tolerant: true matches keep the shop's numbers
    # (sometimes adding a prefix like "BLOCK C-622" or dropping a digit), while
    # decoys replace them
    "a_main_kept", "a_kept_fz", "a_known_fz", "a_n_new", "a_n_lost",
    # how the S2/S3 record is written, and how that differs from the S1 record:
    # decoys differ a little in word count, casing, punctuation and address digits
    # (the record alone separates decoys from real records with AUC 0.76)
    "r2_words", "r2_upper", "r2_punct", "r2_nlen", "r2_adig", "r2_alen", "r2_acomma", "r2_aupper",
    "rd_words", "rd_upper", "rd_punct", "rd_nlen", "rd_adig", "rd_alen", "rd_acomma", "rd_aupper",
    # source
    "is_s3",
    # blocking
    "blk_name_sim", "blk_addr_sim", "blk_exact_sim", "blk_score", "n_keys",
    "via_name", "via_addr", "via_num", "via_nostate", "via_exact", "via_skel",
    # context (filled after the per-pair pass)
    "s1_n_cands", "s1_rank", "s1_gap_best", "c_n_s1", "c_rank", "c_gap_best",
]

N_PAIR_FEATS = FEATURES.index("is_s3") + 1       # computed per pair in workers
CONTEXT_S1 = ["s1_n_cands", "s1_rank", "s1_gap_best"]
CONTEXT_C = ["c_n_s1", "c_rank", "c_gap_best"]

# ─────────────────────────── per-pair computation ───────────────────────────
_IDF, _IDF_DEFAULT = {}, 1.0

def _init_worker(idf, default):
    global _IDF, _IDF_DEFAULT
    _IDF, _IDF_DEFAULT = idf, default

def _idf(t):
    return _IDF.get(t, _IDF_DEFAULT)

def _jacc(a, b):
    if not a and not b:
        return -1.0          # both empty: "unknown", not "different"
    u = len(a | b)
    return len(a & b) / u if u else 0.0

def _soft_cov(t1, t2):
    """IDF share of t1's words that have a close (typo-tolerant) word in t2."""
    if not t1 or not t2:
        return 0.0
    tot = hit = 0.0
    for t in t1:
        w = _idf(t); tot += w
        if t in t2 or max(fuzz.ratio(t, u) for u in t2) >= 80:
            hit += w
    return hit / tot if tot else 0.0

_FILLER_SIGNAL = {"center", "centre", "services", "service", "partners", "group",
                  "enterprises"}

def _fz(x, pool):
    """x equals a number in pool, or one is a digit-dropped version of the other."""
    return any(x == y or (len(x) >= 2 and len(y) >= 2 and (x in y or y in x)) for y in pool)

def _legal_relation(l1, l2):
    """(equal, subset, conflict, missing) for two sets of legal-form words."""
    if not l1 or not l2:
        return 0.0, 0.0, 0.0, 1.0
    if l1 == l2:
        return 1.0, 0.0, 0.0, 0.0
    if l1 < l2 or l2 < l1:
        return 0.0, 1.0, 0.0, 0.0
    return 0.0, 0.0, 1.0, 0.0

def _number_features(n1, n2):
    """n1 = shop (S1) numbers, n2 = S2/S3 numbers (lists of digit strings)."""
    if not n1 or not n2:
        return -1.0, -1.0, -1.0, float(len(n2)), float(len(n1))
    main = max(n1, key=len)
    kept = [_fz(x, n2) for x in n1]
    known = [_fz(x, n1) for x in n2]
    return (float(_fz(main, n2)), sum(kept) / len(kept), sum(known) / len(known),
            float(len(known) - sum(known)), float(len(kept) - sum(kept)))

def _fmt(name, addr):
    name, addr = name or "", addr or ""
    return (len(name.split()), sum(c.isupper() for c in name) / max(len(name), 1),
            sum(1 for c in name if not (c.isalnum() or c.isspace())), len(name),
            sum(c.isdigit() for c in addr), len(addr), addr.count(","),
            sum(c.isupper() for c in addr) / max(len(addr), 1))

def _pair_features(rows):
    out = np.zeros((len(rows), N_PAIR_FEATS), np.float32)
    for i, r in enumerate(rows):
        (core1, comp1, norm1, a1, at1, an1, ha1, pl1,
         core2, comp2, norm2, al2, url2, scr2, a2, at2, an2, ha2, pl2, is_s3,
         rn1, ra1, rn2, ra2) = r
        g1, g2 = _fmt(rn1, ra1), _fmt(rn2, ra2)
        w1, w2 = set(core1.split()), set(core2.split())
        tset = fuzz.token_set_ratio(core1, core2) / 100
        alias_tset = fuzz.token_set_ratio(core1, al2) / 100 if al2 else 0.0
        inter, union = w1 & w2, w1 | w2
        idf_j = (sum(_idf(t) for t in inter) / sum(_idf(t) for t in union)) if union else 0.0
        l1 = set(norm1.split()) & LEGAL_WORDS
        l2 = set(norm2.split()) & LEGAL_WORDS
        legal = -1.0 if not (l1 and l2) else float(bool(l1 & l2))
        f1 = core1.split()[0] if core1 else ""
        f2 = core2.split()[0] if core2 else ""
        n1, n2 = set(an1.split()), set(an2.split())
        leq, lsub, lcon, lmis = _legal_relation(l1, l2)
        pub_priv = float(("public" in l1 and "private" in l2) or ("private" in l1 and "public" in l2))
        fw1 = set(norm1.split()) & _FILLER_SIGNAL
        fw2 = set(norm2.split()) & _FILLER_SIGNAL
        sk1, sk2 = sound_skeleton(core1), sound_skeleton(core2)
        t1, t2 = set(at1.split()), set(at2.split())
        p1, p2 = set(pl1.split("|")) - {""}, set(pl2.split("|")) - {""}
        sh = len(n1 & n2)
        out[i] = (
            fuzz.ratio(core1, core2) / 100,
            fuzz.token_sort_ratio(core1, core2) / 100,
            tset,
            fuzz.partial_ratio(core1, core2) / 100 if core1 and core2 else 0.0,
            JaroWinkler.normalized_similarity(comp1, comp2),
            fuzz.ratio(comp1, comp2) / 100,
            alias_tset,
            max(tset, alias_tset),
            idf_j,
            _soft_cov(w1, w2),
            _soft_cov(w2, w1),
            float(bool(f1) and f1 == f2),
            len(comp1), len(comp2),
            abs(len(comp1) - len(comp2)) / max(len(comp1), len(comp2), 1),
            legal,
            float(url2), float(scr2 != "latin"), float(bool(al2)),
            float(bool(comp1) and comp1 == comp2),
            leq, lsub, lcon, lmis, pub_priv, float(len(fw1 ^ fw2)),
            fuzz.ratio(sk1, sk2) / 100, fuzz.token_set_ratio(sk1, sk2) / 100,
            float(ha1), float(ha2),
            _jacc(t1, t2) if (ha1 and ha2) else -1.0,
            fuzz.token_set_ratio(a1, a2) / 100 if (ha1 and ha2) else -1.0,
            fuzz.ratio(a1, a2) / 100 if (ha1 and ha2) else -1.0,
            sh,
            _jacc(n1, n2),
            sh / len(n1) if n1 else -1.0,
            sh / len(n2) if n2 else -1.0,
            len(n1), len(n2),
            _jacc(p1, p2),
            *_number_features(list(n1), list(n2)),
            *g2, *(x - y for x, y in zip(g2, g1)),
            float(is_s3),
        )
    return out

# ─────────────────────────── data loading ───────────────────────────
def load_records(split, ids1, ids23):
    """Load only the cleaned records that appear in the candidate set."""
    parts = {}
    for src, ids in (("s1", ids1), ("s2", ids23), ("s3", ids23)):
        df = pd.read_parquet(os.path.join(config.WORK_DIR, f"{split}_{src}.parquet"),
                             columns=REC_COLS)
        parts[src] = df[df.entity_id.isin(ids)]      # stays in compact Arrow form
    return parts["s1"].set_index("entity_id"), \
        pd.concat([parts["s2"], parts["s3"]]).set_index("entity_id")

def build_idf(s1_cores):
    """IDF of core-name words, from the S1 records (one per business)."""
    df = Counter(w for c in s1_cores for w in set(c.split()))
    n = len(s1_cores)
    idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
    return idf, math.log(n + 1) + 1      # unseen words count as rarest

def _quick_score(df):
    """Single summary similarity used to rank pairs for the context features."""
    return (df.n_best_tset.values + np.where(df.a_tset.values >= 0, df.a_tset.values, 0)
            + 0.5 * df.n_idf_jacc.values + df.blk_exact_sim.values).astype(np.float32)

def _group_rank(keys, score):
    """Rank (1 = best) of each score within its key group, plus group max."""
    order = np.lexsort((-score, keys))
    k = keys[order]
    first = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
    sizes = np.diff(np.r_[first, len(k)])
    rank = np.empty(len(k), np.float32)
    rank[order] = np.arange(len(k)) - np.repeat(first, sizes) + 1
    best = np.empty(len(k), np.float32)
    best[order] = np.repeat(score[order][first], sizes)
    count = np.empty(len(k), np.float32)
    count[order] = np.repeat(sizes, sizes)
    return rank, best, count

def _s1_chunks(s1_ids, target):
    """Chunk boundaries that never split one S1 record's candidates."""
    change = np.r_[0, np.flatnonzero(s1_ids[1:] != s1_ids[:-1]) + 1, len(s1_ids)]
    bounds, start = [], 0
    for c in change[1:]:
        if c - start >= target or c == len(s1_ids):
            bounds.append((start, c)); start = c
    return bounds

def build_features(cands, rec1, rec23, idf, idf_default, out_dir, labels=None,
                   workers=config.N_WORKERS, chunk=config.FEATURE_CHUNK, sub=20_000,
                   verbose=True):
    """
    Writes features as parquet parts in out_dir, so memory holds one chunk at a
    time. Context features over each candidate's S1 options need every chunk,
    so they are computed at the end from compact integer arrays and added back.
    """
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    cands = cands.reset_index(drop=True)          # blocking output: S1 rows contiguous
    reused = 0
    cand_code = pd.Index(rec23.index).get_indexer(cands.cand_id.values).astype(np.int32)
    q_all = np.empty(len(cands), np.float32)
    parts = []
    with Pool(workers, initializer=_init_worker, initargs=(idf, idf_default)) as pool:
        for pi, (a, b) in enumerate(_s1_chunks(pd.factorize(cands.s1_id)[0], chunk)):
            path = os.path.join(out_dir, f"part_{pi:04d}.parquet")
            # Resume: a part written by an interrupted run is reused only if it is
            # readable and holds exactly this chunk's pairs, in the same order.
            if os.path.exists(path):
                try:
                    import pyarrow.parquet as pq
                    if not set(FEATURES[:N_PAIR_FEATS]) <= set(pq.read_schema(path).names):
                        raise ValueError("part written by an older version")
                    g = pd.read_parquet(path, columns=["s1_id", "cand_id", "n_best_tset",
                                                       "a_tset", "n_idf_jacc", "blk_exact_sim"])
                    same = (len(g) == b - a
                            and (g.s1_id.to_numpy(object) == cands.s1_id.iloc[a:b].to_numpy(object)).all()
                            and (g.cand_id.to_numpy(object) == cands.cand_id.iloc[a:b].to_numpy(object)).all())
                except Exception:
                    same = False
                if same:
                    q_all[a:b] = _quick_score(g)
                    parts.append((path, a, b)); reused += 1
                    continue
            c = cands.iloc[a:b]
            r1 = rec1.loc[c.s1_id.values]
            r2 = rec23.loc[c.cand_id.values]
            rows = list(zip(
                r1.name_core, r1.name_compact, r1.name_norm, r1.addr_norm, r1.addr_tokens,
                r1.addr_nums, r1.has_addr, r1.place_labels,
                r2.name_core, r2.name_compact, r2.name_norm, r2.name_alias, r2.name_is_url,
                r2.name_script, r2.addr_norm, r2.addr_tokens, r2.addr_nums, r2.has_addr,
                r2.place_labels, [x.startswith("S3-") for x in c.cand_id.values],
                r1.name_raw, r1.addr_raw, r2.name_raw, r2.addr_raw))
            del r1, r2
            feats = pool.map(_pair_features, [rows[i:i + sub] for i in range(0, len(rows), sub)])
            del rows
            f = pd.DataFrame(np.vstack(feats), columns=FEATURES[:N_PAIR_FEATS])
            f.insert(0, "s1_id", c.s1_id.values)
            f.insert(1, "cand_id", c.cand_id.values)
            for col in ["blk_name_sim", "blk_addr_sim", "blk_exact_sim", "blk_score", "n_keys",
                        "via_name", "via_addr", "via_num", "via_nostate", "via_exact", "via_skel"]:
                f[col] = c[col].values.astype(np.float32)
            q = _quick_score(f)
            q_all[a:b] = q
            s1_code = pd.factorize(f.s1_id)[0]
            rank, best, count = _group_rank(s1_code, q)
            f["s1_n_cands"], f["s1_rank"], f["s1_gap_best"] = count, rank, best - q
            if labels is not None:
                f["y"] = np.fromiter(((x, y) in labels for x, y in zip(f.s1_id, f.cand_id)),
                                     dtype=np.int8, count=len(f))
            f.to_parquet(path, index=False)
            parts.append((path, a, b))
            if verbose:
                print(f"    {b:,}/{len(cands):,} pairs ({time.time()-t0:.0f}s)", flush=True)

    # remove leftovers that do not belong to this candidate set
    keep = {os.path.basename(p) for p, _, _ in parts}
    for old in os.listdir(out_dir):
        if old not in keep:
            os.remove(os.path.join(out_dir, old))
    if verbose:
        if reused:
            print(f"    reused {reused} parts saved by an earlier, interrupted run", flush=True)
        print("    candidate-side context ...", flush=True)
    rank, best, count = _group_rank(cand_code, q_all)
    for path, a, b in parts:
        f = pd.read_parquet(path)
        f["c_n_s1"], f["c_rank"], f["c_gap_best"] = count[a:b], rank[a:b], best[a:b] - q_all[a:b]
        cols = ["s1_id", "cand_id"] + FEATURES + (["y"] if "y" in f else [])
        f[cols].to_parquet(path, index=False)
    if verbose:
        print(f"    wrote {len(parts)} parts to {out_dir} ({time.time()-t0:.0f}s)")

def load_features(split, columns=None):
    """Read all feature parts of a split into one DataFrame (train-sized data)."""
    d = os.path.join(config.WORK_DIR, f"{split}_features")
    return pd.concat([pd.read_parquet(os.path.join(d, p), columns=columns)
                      for p in sorted(os.listdir(d))], ignore_index=True)

def true_pairs(gt_path, s1_ids):
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt.source1_entity_id.isin(s1_ids)]
    return {(s, x) for s, v in zip(gt.source1_entity_id, gt.matched_entity_ids)
            for x in v.split(",") if x}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    args = ap.parse_args()
    t0 = time.time()
    from blocking import load_candidates
    cands = load_candidates(args.split)
    print(f"{len(cands):,} candidate pairs for {cands.s1_id.nunique():,} S1 records")
    s1_ids = set(cands.s1_id)
    rec1, rec23 = load_records(args.split, s1_ids, set(cands.cand_id))
    idf, idf_default = build_idf(rec1.name_core.astype(object).tolist())
    labels = true_pairs(config.TRAIN_FILES["gt"], s1_ids) if args.split == "train" else None
    print(f"records loaded ({time.time()-t0:.0f}s); computing features ...")
    out_dir = os.path.join(config.WORK_DIR, f"{args.split}_features")
    build_features(cands, rec1, rec23, idf, idf_default, out_dir, labels=labels)
    if labels is not None:
        y = load_features(args.split, columns=["y"]).y
        print(f"labels: {int(y.sum()):,} true pairs among {len(y):,} candidates")
    print(f"done ({time.time()-t0:.0f}s total)")

if __name__ == "__main__":
    main()
