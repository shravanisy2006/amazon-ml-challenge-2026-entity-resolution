"""
blocking.py — Step 2: candidate generation.

    python blocking.py --split train --sample-partitions 0.15   # training data
    python blocking.py --split test

Produces WORK_DIR/<split>_candidates.parquet (s1_id, cand_id, blocking signals)
and, for test, output/candidate_pairs.tsv in the official format.
The test split is processed one country at a time to bound memory.

Keys (all within the same country; A/B/C also within the same state):
  A  name vectors      TF-IDF char 2-4grams on the compact core name (and DBA
                       alias) -> SVD -> exact inner-product search (FAISS)
  B  address vectors   same, on the order-free set of street/city tokens
  C  address numbers   exact match on normalized numbers (>=3 digits)
  D  no-state rescue   S2/S3 records whose state could not be parsed are
                       matched country-wide by name (reverse search)
  S  sound skeleton    Indic-script S2/S3 names searched again by consonant
                       skeleton + address ("istarn inphra" ~ "eastern infra")
  E  exact joint       (main key) every S2/S3 record finds its top-k S1 records
                       by exact TF-IDF cosine(name) + cosine(address), no
                       compression. Each S2/S3 record belongs to at most one
                       business, so searching from its side fits the data.
Country is an open set: every distinct country string becomes its own
partition, so unseen countries (France) are handled identically.
"""
import argparse, os, time
from collections import defaultdict
import numpy as np
import pandas as pd
import faiss
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import config
import lightgbm as lgb
from geo import assign_regions
from normalize import sound_skeleton
from features import _group_rank

# ─────────────────────────── vector encoders ───────────────────────────
class Encoder:
    """TF-IDF char n-grams -> SVD -> L2-normalized float32 vectors."""
    def __init__(self, dim=config.SVD_DIM, ngram=(2, 4), analyzer="char"):
        self.tfidf = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram,
                                     min_df=3, max_features=2**18,
                                     sublinear_tf=True, dtype=np.float32)
        self.svd = TruncatedSVD(n_components=dim, random_state=config.RANDOM_SEED)

    def fit(self, texts):
        texts = [t for t in texts if t]
        dim = min(self.svd.n_components, max(2, len(texts) // 2))
        self.svd.n_components = dim
        self.svd.fit(self.tfidf.fit_transform(texts))
        return self

    def encode(self, texts, batch=200_000):
        out = []
        for i in range(0, len(texts), batch):
            x = self.svd.transform(self.tfidf.transform(texts[i:i + batch]))
            out.append(normalize(x).astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, self.svd.n_components), np.float32)

# States that the data mixes for the same business. Found from training ground
# truth: Telangana was carved out of Andhra Pradesh (2014), so older records
# carry the old state. Only affects which records are compared, never labels.
PARTITION_MERGE = {"telangana": "andhra pradesh"}

def _part_state(state: str) -> str:
    return PARTITION_MERGE.get(state, state)

def _addr_key(tokens: str, state: str) -> str:
    """Order-free street/city text: address tokens minus the state words."""
    drop = set(state.split())
    return " ".join(t for t in tokens.split() if t not in drop)

# ─────────────────────────── search helpers ───────────────────────────
F_NAME, F_ADDR, F_NUM, F_NOSTATE, F_EXACT, F_SKEL = 1, 2, 4, 8, 16, 32
_POPCOUNT = np.array([bin(i).count("1") for i in range(64)], np.int8)
_SIM_COLS = ((F_NAME | F_NOSTATE, 0), (F_ADDR, 1), (F_EXACT | F_SKEL, 2))   # flag -> sim column

def _knn(index_vecs, query_vecs, k):
    """Exact cosine kNN (vectors are unit length). Returns sims, idx."""
    if len(index_vecs) == 0 or len(query_vecs) == 0 or k <= 0:
        return (np.zeros((len(query_vecs), 0), np.float32),
                np.zeros((len(query_vecs), 0), np.int64))
    k = min(k, len(index_vecs))
    if len(index_vecs) >= config.HNSW_MIN_SIZE:
        index = faiss.IndexHNSWFlat(index_vecs.shape[1], config.HNSW_M,
                                    faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = max(config.HNSW_EF_SEARCH, 2 * k)
    else:
        index = faiss.IndexFlatIP(index_vecs.shape[1])
    index.add(index_vecs)
    sims, idx = [], []
    for i in range(0, len(query_vecs), config.QUERY_BATCH):
        s, j = index.search(query_vecs[i:i + config.QUERY_BATCH], k)
        sims.append(s); idx.append(j)
    return np.vstack(sims), np.vstack(idx)

class PairBuffer:
    """Collects (s1_row, s23_row, sim, flag) as compact int/float arrays."""
    def __init__(self):
        self.parts = []

    def add(self, s1_rows, s23_rows, sims, flag):
        m = sims >= config.MIN_SIM
        if m.any():
            self.parts.append((s1_rows[m].astype(np.int64), s23_rows[m].astype(np.int64),
                               sims[m].astype(np.float32),
                               np.full(int(m.sum()), flag, np.int8)))

    def add_search(self, q_rows, t_rows, sims, idx, flag, query_is_s1):
        """Map a kNN result (local indices) back to global row numbers."""
        if idx.size == 0:
            return
        valid = idx >= 0
        q = np.repeat(q_rows, idx.shape[1]).reshape(idx.shape)[valid]
        t = t_rows[idx[valid]]
        s = sims[valid]
        self.add(q, t, s, flag) if query_is_s1 else self.add(t, q, s, flag)

    def reduce(self, n23):
        """Merge duplicate pairs: best name sim, best address sim, OR of key flags."""
        if not self.parts:
            return (np.zeros(0, np.int64), np.zeros(0, np.int64),
                    np.zeros((0, 3), np.float32), np.zeros(0, np.int8))
        a = np.concatenate([p[0] for p in self.parts])
        b = np.concatenate([p[1] for p in self.parts])
        f = np.concatenate([p[3] for p in self.parts])
        # raw searches carry one sim column; already-reduced parts carry three
        s = np.concatenate([
            p[2] if p[2].ndim == 2 else
            np.stack([np.where((p[3] & fl) > 0, p[2], 0) for fl, _ in _SIM_COLS], 1)
            for p in self.parts]).astype(np.float32)
        key = a * n23 + b
        order = np.argsort(key, kind="stable")
        key, s, f = key[order], s[order], f[order]
        uniq, start = np.unique(key, return_index=True)
        sims = np.stack([np.maximum.reduceat(s[:, c], start) for c in range(3)], 1)
        return uniq // n23, uniq % n23, sims, np.bitwise_or.reduceat(f, start)

def rank_score(sim, flags):
    """
    Blocking score. Pairs from the exact joint search always rank first (that
    search has by far the best recall); then keys agreeing and similarities.
    """
    return (4.0 * ((flags & (F_EXACT | F_SKEL)) > 0) + sim[:, 2]
            + _POPCOUNT[flags] + sim[:, 0] + sim[:, 1])

def _cap(s1r, s23r, sim, flags, cap):
    """Keep the best `cap` candidates per S1 by rank_score."""
    if len(s1r) == 0:
        return s1r, s23r, sim, flags
    order = np.lexsort((-rank_score(sim, flags), s1r))
    s1r, s23r, sim, flags = s1r[order], s23r[order], sim[order], flags[order]
    first = np.r_[0, np.flatnonzero(np.diff(s1r)) + 1]
    rank = np.arange(len(s1r)) - np.repeat(first, np.diff(np.r_[first, len(s1r)]))
    keep = rank < cap
    return s1r[keep], s23r[keep], sim[keep], flags[keep]

# ─────────────────────────── learned trimming ───────────────────────────
# The searches find far more true matches than survive the per-S1 cap. A small
# LightGBM "ranker" decides which candidates to keep, using cheap blocking
# signals and, crucially, how each pair compares with its neighbours: how far
# it is behind the S1 record's best candidate, and whether this S1 is the
# candidate record's best option. On a quarter of Delhi it kept 99.0% of the
# found true pairs at 15 per S1, versus 97.7% at 30 per S1 for the fixed formula.
def ranker_features(a, b, sims, flags):
    """a, b: S1 / S2-S3 row numbers; sims: (n, 3) name, address, exact; flags: key bits."""
    cols = [sims[:, 0], sims[:, 1], sims[:, 2], _POPCOUNT[flags].astype(np.float32)]
    cols += [((flags & bit) > 0).astype(np.float32)
             for bit in (F_NAME, F_ADDR, F_NUM, F_NOSTATE, F_EXACT, F_SKEL)]
    n1 = n2 = None
    for keys in (a, b):
        for j in range(3):
            rank, best, count = _group_rank(keys, sims[:, j].astype(np.float32))
            cols += [rank, best - sims[:, j]]
        cols.append(count)
    return np.column_stack(cols).astype(np.float32)

def train_ranker(collected):
    X = np.vstack([x for x, _ in collected])
    y = np.concatenate([t for _, t in collected])
    if len(y) > config.RANKER_MAX_ROWS:
        keep = np.random.default_rng(config.RANDOM_SEED).choice(len(y), config.RANKER_MAX_ROWS, replace=False)
        X, y = X[keep], y[keep]
    print(f"  training ranker on {len(y):,} pairs ({int(y.sum()):,} true)", flush=True)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                  verbose=-1, seed=config.RANDOM_SEED)
    return lgb.train(params, lgb.Dataset(X, y.astype(np.int8)), num_boost_round=300)

def _cap_by(a, b, sims, flags, score, cap):
    """Keep the `cap` highest-scoring candidates of every S1 record."""
    if len(a) == 0:
        return a, b, sims, flags, score
    order = np.lexsort((-score, a))
    a, b, sims, flags, score = a[order], b[order], sims[order], flags[order], score[order]
    first = np.r_[0, np.flatnonzero(np.diff(a)) + 1]
    rank = np.arange(len(a)) - np.repeat(first, np.diff(np.r_[first, len(a)]))
    k = rank < cap
    return a[k], b[k], sims[k], flags[k], score[k]

def _select(reduced, n23, ranker, collect, truth_keys, train_mask):
    """Score one partition's pairs (ranker or fixed formula) and trim to the cap."""
    a, b, sims, flags = reduced
    if len(a) == 0:
        return a, b, sims, flags, np.zeros(0, np.float32)
    if ranker is not None:
        X = ranker_features(a, b, sims, flags)
        score = np.concatenate([ranker.predict(X[i:i + 2_000_000])
                                for i in range(0, len(X), 2_000_000)]).astype(np.float32)
        del X
    else:
        if collect is not None:
            m = train_mask[a]
            if m.any():
                X = ranker_features(a, b, sims, flags)[m]   # neighbours from the full partition
                collect.append((X, np.isin(a[m] * n23 + b[m], truth_keys)))
        score = rank_score(sims, flags).astype(np.float32)
    return _cap_by(a, b, sims, flags, score, config.MAX_CANDIDATES)

# ─────────────────────────── main routine ───────────────────────────
def fit_encoders(names, addrs, skels):
    """
    Fit on a random sample of texts:
      compressed encoders (TF-IDF -> SVD, for FAISS search) and
      exact TF-IDF vectorizers (for the exact joint search).
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    def samp(arr):
        arr = np.asarray(arr, dtype=object)
        return arr[rng.choice(len(arr), min(len(arr), config.SVD_FIT_SAMPLE), replace=False)].tolist()
    sn, sa = samp(names), samp(addrs)
    # n-grams found in more than EXACT_MAX_DF of records (e.g. "delhi" in every
    # Delhi address) carry little information but make the sparse product
    # nearly dense; dropping them made the search 3.6x faster for -0.5% recall
    exact_n = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2,
                              max_df=config.EXACT_MAX_DF, sublinear_tf=True,
                              dtype=np.float32).fit([t for t in sn if t])
    exact_a = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2,
                              max_df=config.EXACT_MAX_DF, sublinear_tf=True,
                              dtype=np.float32).fit([t for t in sa if t])
    exact_s = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2,
                              max_df=config.EXACT_MAX_DF, sublinear_tf=True,
                              dtype=np.float32).fit([t for t in samp(skels) if t])
    return (Encoder(ngram=(2, 4), analyzer="char").fit(sn),
            Encoder(ngram=(3, 4), analyzer="char_wb").fit(sa), exact_n, exact_a, exact_s)

def exact_joint_search(tv_name, tv_addr, q_names, q_addrs, t_names, t_addrs, k, log=None):
    """
    Exact cosine(name) + cosine(address) between every query and every target
    (TF-IDF, no compression), keeping each query's top-k targets. Computed with
    sparse_dot_topn (Apache-2.0), never materialising the full score matrix.
    Returns (query_rows, target_rows, scores).
    """
    if len(q_names) == 0 or len(t_names) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    T = sp.hstack([normalize(tv_name.transform(t_names)),
                   normalize(tv_addr.transform(t_addrs))]).tocsr().T.tocsr()
    qs, ts, ss = [], [], []
    for i in range(0, len(q_names), config.EXACT_QUERY_BATCH):
        Q = sp.hstack([normalize(tv_name.transform(q_names[i:i + config.EXACT_QUERY_BATCH])),
                       normalize(tv_addr.transform(q_addrs[i:i + config.EXACT_QUERY_BATCH]))]).tocsr()
        M = sp_matmul_topn(Q, T, top_n=k, threshold=config.EXACT_MIN_SIM,
                           n_threads=config.N_THREADS).tocoo()
        qs.append(M.row.astype(np.int64) + i); ts.append(M.col.astype(np.int64))
        ss.append(M.data.astype(np.float32))
        if log and len(q_names) > config.EXACT_QUERY_BATCH:
            log(f"    exact search: {min(i + config.EXACT_QUERY_BATCH, len(q_names)):,}"
                f"/{len(q_names):,} records")
    return np.concatenate(qs), np.concatenate(ts), np.concatenate(ss)

def generate_candidates(s1: pd.DataFrame, s23: pd.DataFrame, verbose=True, encoders=None,
                        ranker=None, truth_pairs=None, labelled_s1=None):
    """
    Returns a DataFrame of (s1_id, cand_id) candidate pairs with blocking
    signals. Memory stays bounded by the largest (country, state) partition:
    vectors are computed per partition and discarded after searching it.
    """
    t0 = time.time()
    log = (lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)) if verbose else (lambda *a: None)
    # compact Arrow strings on disk -> plain Python strings for the search code
    s1 = s1.reset_index(drop=True).astype(object)
    s23 = s23.reset_index(drop=True).astype(object)
    n23 = len(s23)
    log("learning regions from place labels ...")
    s1, s23 = assign_regions(s1, s23, verbose=verbose)

    # name rows: every S2/S3 record, plus a second row for its DBA alias
    has_alias = np.flatnonzero(s23.name_alias.values != "")
    name_row_to_s23 = np.concatenate([np.arange(n23), has_alias])
    name_text = np.concatenate([s23.name_compact.values,
                                s23.name_alias.str.replace(" ", "").values[has_alias]])
    s1_name = s1.name_compact.values
    s1_addr = np.array([_addr_key(t, s) for t, s in zip(s1.addr_tokens, s1.state)], dtype=object)
    s23_addr = np.array([_addr_key(t, s) for t, s in zip(s23.addr_tokens, s23.state)], dtype=object)
    has_a1, has_a23 = s1_addr != "", s23_addr != ""
    # sound skeletons: all S1 names, and S2/S3 names written in an Indic script
    s1_skel = np.array([sound_skeleton(x) for x in s1.name_core.values], dtype=object)
    indic23 = s23.name_script.values != "latin"
    s23_skel = np.array([sound_skeleton(x) if ind else "" for x, ind in
                         zip(s23.name_core.values, indic23)], dtype=object)

    if encoders is None:
        log("fitting encoders on a sample ...")
        encoders = fit_encoders(np.concatenate([s1_name, name_text]),
                                np.concatenate([s1_addr, s23_addr]),
                                np.concatenate([s1_skel, s23_skel[indic23]]))
    enc_name, enc_addr, tv_name, tv_addr, tv_skel = encoders
    # training mode: collect ranker examples for a sample of S1 records
    collect, truth_keys, train_mask = None, None, None
    if truth_pairs is not None:
        collect = []
        ta = pd.Index(s1.entity_id).get_indexer([p[0] for p in truth_pairs])
        tb = pd.Index(s23.entity_id).get_indexer([p[1] for p in truth_pairs])
        ok = (ta >= 0) & (tb >= 0)
        truth_keys = np.sort(ta[ok].astype(np.int64) * n23 + tb[ok])
        train_mask = np.random.default_rng(config.RANDOM_SEED).random(len(s1)) < config.RANKER_SAMPLE
        if labelled_s1 is not None:
            # learn only from S1 records that are in the answer key (all of them in
            # the real training data, including singletons); unlabelled ones would
            # teach the ranker that their true matches are non-matches
            train_mask &= s1.entity_id.isin(labelled_s1).values

    p1 = np.array([_part_state(x) for x in s1.state.values], dtype=object)
    p23 = np.array([_part_state(x) for x in s23.state.values], dtype=object)
    c1, c23 = s1.country.values, s23.country.values
    g1 = pd.DataFrame({"c": c1, "p": p1}).groupby(["c", "p"]).indices
    g23 = pd.DataFrame({"c": c23, "p": p23}).groupby(["c", "p"]).indices
    gn = pd.DataFrame({"c": c23[name_row_to_s23], "p": p23[name_row_to_s23]}).groupby(["c", "p"]).indices
    nums23 = s23.addr_nums.values; nums1 = s1.addr_nums.values

    results, empty = [], np.array([], np.int64)
    parts = sorted(g1.items(), key=lambda kv: -len(kv[1]))
    log(f"searching {len(parts)} partitions (largest: {parts[0][0]}, {len(parts[0][1]):,} S1) ...")
    for pi, (part, i1) in enumerate(parts):
        buf = PairBuffer()
        jn = gn.get(part, empty)
        ja = g23.get(part, empty); ja = ja[has_a23[ja]]
        i1a = i1[has_a1[i1]]
        if len(jn):
            v1n, vnn = enc_name.encode(s1_name[i1].tolist()), enc_name.encode(name_text[jn].tolist())
            s, j = _knn(vnn, v1n, config.TOPK_NAME)                # A: S1 -> S2/S3
            buf.add_search(i1, name_row_to_s23[jn], s, j, F_NAME, query_is_s1=True)
            s, j = _knn(v1n, vnn, config.TOPK_REVERSE)             # A: S2/S3 -> S1
            buf.add_search(name_row_to_s23[jn], i1, s, j, F_NAME, query_is_s1=False)
            del v1n, vnn
        if len(ja) and len(i1a):
            v1a, v23a = enc_addr.encode(s1_addr[i1a].tolist()), enc_addr.encode(s23_addr[ja].tolist())
            s, j = _knn(v23a, v1a, config.TOPK_ADDR)               # B: S1 -> S2/S3
            buf.add_search(i1a, ja, s, j, F_ADDR, query_is_s1=True)
            s, j = _knn(v1a, v23a, config.TOPK_REVERSE)            # B: S2/S3 -> S1
            buf.add_search(ja, i1a, s, j, F_ADDR, query_is_s1=False)
            del v1a, v23a
        if len(jn) and config.TOPK_EXACT > 0:                             # E: exact joint
            qr, tr, sc = exact_joint_search(
                tv_name, tv_addr, name_text[jn].tolist(), s23_addr[name_row_to_s23[jn]].tolist(),
                s1_name[i1].tolist(), s1_addr[i1].tolist(), config.TOPK_EXACT, log=log)
            buf.add(i1[tr], name_row_to_s23[jn][qr], sc, F_EXACT)
            # S: Indic-script names searched again by sound skeleton + address
            jq = jn[jn < n23]
            jq = jq[indic23[jq]]
            if len(jq):
                qr, tr, sc = exact_joint_search(
                    tv_skel, tv_addr, s23_skel[jq].tolist(), s23_addr[jq].tolist(),
                    s1_skel[i1].tolist(), s1_addr[i1].tolist(), config.TOPK_EXACT)
                buf.add(i1[tr], jq[qr], sc, F_SKEL)
        post = defaultdict(list)                                          # C: numbers
        for r in g23.get(part, ()):
            for n in nums23[r].split():
                if len(n) >= 3:
                    post[n].append(r)
        a_rows, b_rows = [], []
        for r in i1:
            for n in nums1[r].split():
                hits = post.get(n) if len(n) >= 3 else None
                if hits and len(hits) <= config.NUM_MAX_POSTINGS:
                    a_rows.extend([r] * len(hits)); b_rows.extend(hits)
        if a_rows:
            buf.add(np.array(a_rows), np.array(b_rows),
                    np.zeros(len(a_rows), np.float32) + config.MIN_SIM, F_NUM)
        results.append(_select(buf.reduce(n23), n23, ranker, collect, truth_keys, train_mask))
        if verbose and (pi < 5 or pi % 20 == 0):
            log(f"  partition {pi+1}/{len(parts)} {part}: {len(i1):,} S1, {len(jn):,} name rows")

    log("no-state rescue ...")                                             # D
    buf = PairBuffer()
    ns_rows = np.flatnonzero(p23[name_row_to_s23] == "")
    ns_c = c23[name_row_to_s23][ns_rows]
    by_c1 = pd.Series(c1).groupby(c1).indices
    for c, rel in pd.Series(ns_c).groupby(ns_c).indices.items():
        jn = ns_rows[rel]
        i1 = by_c1.get(c, empty)
        if len(i1) == 0 or (p1[i1] == "").all():
            continue    # country has no states: already searched as one partition
        s, j = _knn(enc_name.encode(s1_name[i1].tolist()),
                    enc_name.encode(name_text[jn].tolist()), config.TOPK_NOSTATE)
        buf.add_search(name_row_to_s23[jn], i1, s, j, F_NOSTATE, query_is_s1=False)
    results.append(_select(buf.reduce(n23), n23, ranker, collect, truth_keys, train_mask))

    log("merging ...")
    results = [r for r in results if len(r[0])]
    a = np.concatenate([r[0] for r in results]); b_ = np.concatenate([r[1] for r in results])
    s_ = np.concatenate([r[2] for r in results]); f_ = np.concatenate([r[3] for r in results])
    sc = np.concatenate([r[4] for r in results])
    key = a * n23 + b_
    order = np.argsort(key, kind="stable")
    key, s_, f_, sc = key[order], s_[order], f_[order], sc[order]
    uniq, start = np.unique(key, return_index=True)
    a, b_ = uniq // n23, uniq % n23
    s_ = np.stack([np.maximum.reduceat(s_[:, j], start) for j in range(3)], 1)
    f_ = np.bitwise_or.reduceat(f_, start)
    sc = np.maximum.reduceat(sc, start)
    a, b_, s_, f_, sc = _cap_by(a, b_, s_, f_, sc, config.MAX_CANDIDATES)
    df = pd.DataFrame({
        "s1_id": s1.entity_id.values[a], "cand_id": s23.entity_id.values[b_],
        "blk_name_sim": s_[:, 0], "blk_addr_sim": s_[:, 1], "blk_exact_sim": s_[:, 2],
        "blk_score": sc, "via_exact": (f_ & F_EXACT) > 0,
        "via_name": (f_ & F_NAME) > 0, "via_addr": (f_ & F_ADDR) > 0,
        "via_num": (f_ & F_NUM) > 0, "via_nostate": (f_ & F_NOSTATE) > 0,
        "via_skel": (f_ & F_SKEL) > 0,
        "n_keys": _POPCOUNT[f_]})
    log(f"done: {len(df):,} pairs for {df.s1_id.nunique():,} S1 entities")
    return df, encoders, collect

def sample_partitions(s1, s23, frac, seed=config.RANDOM_SEED):
    """
    Keep a random fraction of whole (country, state) partitions. Used to build
    training data: blocking inside a kept partition sees exactly the same
    competition as at full scale. S2/S3 records without a state are kept too,
    so the no-state key is exercised.
    """
    key1 = pd.Series([f"{c}|{_part_state(s)}" for c, s in zip(s1.country, s1.state)])
    key23 = pd.Series([f"{c}|{_part_state(s)}" for c, s in zip(s23.country, s23.state)])
    parts = sorted(set(key1))
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(parts, max(1, int(round(frac * len(parts)))), replace=False))
    kept_c = {k.split("|")[0] for k in keep}
    s1k = s1[key1.isin(keep).values]
    s23k = s23[(key23.isin(keep) | ((s23.state == "").values & s23.country.isin(kept_c).values)).values]
    return s1k.reset_index(drop=True), s23k.reset_index(drop=True)

# ─────────────────────────── io ───────────────────────────
def save_candidates(cands, split):
    path = os.path.join(config.WORK_DIR, f"{split}_candidates.parquet")
    cands.to_parquet(path, index=False)
    return path

def load_candidates(split):
    """Parquet (current format) or pickle (written by earlier versions)."""
    pq = os.path.join(config.WORK_DIR, f"{split}_candidates.parquet")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    return pd.read_pickle(os.path.join(config.WORK_DIR, f"{split}_candidates.pkl"))

def write_id_lists(s1_col, cand_col, all_s1_ids, path, col, chunk=2_000_000):
    """
    Official TSV: one row per S1 id (all of them, empty list allowed), S2 ids
    then S3 ids, comma-separated, no duplicates. Rows are written in chunks,
    so tens of millions of pairs never need to be held as Python strings at
    once. Pairs must be grouped by S1 (blocking output is; predict sorts).
    """
    s1_col = pd.Series(s1_col).reset_index(drop=True)
    cand_col = pd.Series(cand_col).reset_index(drop=True)
    n = len(s1_col)
    codes = pd.factorize(s1_col)[0]
    change = np.r_[0, np.flatnonzero(np.diff(codes)) + 1, n]
    written = set()
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        start = 0
        while start < n:
            end = n if start + chunk >= n else int(change[np.searchsorted(change, start + chunk)])
            s = s1_col.iloc[start:end].tolist()
            c = cand_col.iloc[start:end].tolist()
            i = 0
            while i < len(s):
                sid, ids, k = s[i], set(), i
                while k < len(s) and s[k] == sid:
                    ids.add(c[k]); k += 1
                if sid in written:
                    raise ValueError(f"pairs for {sid} are not grouped together")
                written.add(sid)
                ids = sorted(ids)
                ids = [x for x in ids if x.startswith("S2-")] + [x for x in ids if x.startswith("S3-")]
                fh.write(f"{sid}\t{','.join(ids)}\n")
                i = k
            start = end
        for sid in all_s1_ids:
            if sid not in written:
                fh.write(f"{sid}\t\n")
                written.add(sid)

def load_blk(split, src, country=None):
    from preprocess import BLOCK_COLS
    return pd.read_parquet(os.path.join(config.WORK_DIR, f"{split}_{src}.parquet"),
                           columns=BLOCK_COLS,
                           filters=[("country", "==", country)] if country else None)

def run_by_country(split):
    """
    Blocking for a full split without loading everything at once. Partitions
    never cross countries, so running one country at a time gives the same
    result with a fraction of the memory. Encoders are fitted once on a
    sample of the whole split so similarities are comparable everywhere.
    """
    from preprocess import BLOCK_COLS
    t0 = time.time()
    s1_all = pd.read_parquet(os.path.join(config.WORK_DIR, f"{split}_s1.parquet"),
                             columns=["entity_id", "country", "name_compact", "addr_tokens", "state"])
    rng = np.random.default_rng(config.RANDOM_SEED)
    names, addrs, skels = [], [], []
    for src in ("s1", "s2", "s3"):
        d = pd.read_parquet(os.path.join(config.WORK_DIR, f"{split}_{src}.parquet"),
                            columns=["name_compact", "addr_tokens", "state", "name_core", "name_script"])
        take = rng.choice(len(d), min(len(d), config.SVD_FIT_SAMPLE // 3), replace=False)
        d = d.iloc[take].astype(object)
        names += d.name_compact.tolist()
        addrs += [_addr_key(t, s) for t, s in zip(d.addr_tokens, d.state)]
        skels += [sound_skeleton(c) for c, sc in zip(d.name_core, d.name_script)
                  if src == "s1" or sc != "latin"]
    print(f"[{time.time()-t0:6.0f}s] fitting encoders on {len(names):,} sampled records ...", flush=True)
    encoders = fit_encoders(np.array(names, dtype=object), np.array(addrs, dtype=object),
                            np.array(skels, dtype=object))
    del names, addrs, skels
    rpath = os.path.join(config.WORK_DIR, "ranker.txt")
    if not os.path.exists(rpath):
        raise SystemExit("ranker.txt not found: run the training blocking step (step 3) first")
    ranker = lgb.Booster(model_file=rpath)
    out = []
    for country in sorted(s1_all.country.dropna().unique()):
        s1 = load_blk(split, "s1", country)
        s23 = pd.concat([load_blk(split, "s2", country), load_blk(split, "s3", country)],
                        ignore_index=True)
        print(f"[{time.time()-t0:6.0f}s] country '{country}': {len(s1):,} S1, {len(s23):,} S2/S3", flush=True)
        c, _, _ = generate_candidates(s1, s23, encoders=encoders, ranker=ranker)
        c["s1_id"] = c.s1_id.astype("string[pyarrow]")
        c["cand_id"] = c.cand_id.astype("string[pyarrow]")
        out.append(c)
        del s1, s23
    return pd.concat(out, ignore_index=True), s1_all.entity_id.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--sample-partitions", type=float, default=None,
                    help="train only: keep this fraction of whole states (e.g. 0.15)")
    ap.add_argument("--tsv-only", action="store_true",
                    help="test only: rewrite output/candidate_pairs.tsv from saved candidates")
    args = ap.parse_args()

    if args.split == "test":
        if args.tsv_only:
            cands = load_candidates("test")
            all_s1 = pd.read_parquet(os.path.join(config.WORK_DIR, "test_s1.parquet"),
                                     columns=["entity_id"]).entity_id.tolist()
        else:
            cands, all_s1 = run_by_country("test")
            print("saved", save_candidates(cands, "test"), flush=True)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        path = os.path.join(config.OUTPUT_DIR, "candidate_pairs.tsv")
        print("writing", path, "...", flush=True)
        write_id_lists(cands.s1_id, cands.cand_id, all_s1, path, "candidate_entity_ids")
        print("saved", path)
        return

    s1 = load_blk(args.split, "s1")
    s23 = pd.concat([load_blk(args.split, "s2"), load_blk(args.split, "s3")], ignore_index=True)
    if args.sample_partitions:
        s1, s23 = sample_partitions(s1, s23, args.sample_partitions)
        print(f"sampled partitions: {len(s1):,} S1, {len(s23):,} S2/S3")
    from evaluate import load_id_lists, blocking_report
    truth = load_id_lists(config.TRAIN_FILES["gt"], "matched_entity_ids")
    truth = {k: truth[k] for k in s1.entity_id if k in truth}
    print("pass 1: fixed-formula trimming, collecting examples for the ranker ...", flush=True)
    _, encoders, collected = generate_candidates(
        s1, s23, truth_pairs=[(s, x) for s, v in truth.items() for x in v], labelled_s1=set(truth))
    ranker = train_ranker(collected)
    del collected
    ranker.save_model(os.path.join(config.WORK_DIR, "ranker.txt"))
    print("pass 2: learned trimming ...", flush=True)
    cands, _, _ = generate_candidates(s1, s23, encoders=encoders, ranker=ranker)
    print("saved", save_candidates(cands, args.split))
    if True:
        rep = blocking_report(cands[cands.s1_id.isin(truth)], truth, len(s23))
        print("blocking quality:", {k: round(v, 4) for k, v in rep.items()})

if __name__ == "__main__":
    main()
