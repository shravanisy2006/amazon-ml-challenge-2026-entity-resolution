"""
geo.py — learn each country's region structure from the data itself.

Partitioning by state is what keeps blocking precise, but
  * some countries have no state table (France appears only in test), and
  * many S2/S3 records name a city or district but no parseable state.

Place labels are the digit-free comma components of an address
("bordeaux", "nouvelle aquitaine", "gironde"). From co-occurrence alone:

  1. Region of every S1 record
       - country with a state table: the parsed state
       - otherwise: the label that "contains" the others. Label A is the parent
         of B when almost every record naming B also names A
         (P(A|B) >= PARENT_CONF) and A is more frequent. Large parentless
         labels are the regions.
  2. label -> region from S1   (e.g. bordeaux -> nouvelle aquitaine,
                                kansas city -> missouri)
  3. label -> region from S2/S3 records whose region is already known
       (e.g. gironde -> nouvelle aquitaine), which covers labels S1 never uses.
  4. Any record without a region gets one from its labels.

Nothing here is specific to a country and no external data is used; the
statistics come only from the records of the split being processed.
"""
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

PARENT_CONF = 0.95      # P(parent | child) needed to call a label a child
MAP_CONF = 0.90         # P(region | label) needed to map a label to a region
MIN_LABEL_FRAC = 0.002  # labels rarer than this (share of country S1) are ignored
MIN_LABEL_COUNT = 30
MIN_REGION_FRAC = 0.01  # induced regions must cover at least 1% of the country's S1
TABLE_COUNTRY_MIN_STATED = 0.5

def _labels(col):
    return [x.split("|") if x else [] for x in col]

def _induce_s1_regions(labels):
    """For a country without a state table: region per S1 record, by containment."""
    n = len(labels)
    cnt = Counter(l for ls in labels for l in set(ls))
    minc = max(MIN_LABEL_COUNT, MIN_LABEL_FRAC * n)
    freq = {l for l, c in cnt.items() if c >= minc}
    co = Counter()
    for ls in labels:
        f = set(ls) & freq
        for a in f:
            for b in f:
                if a != b:
                    co[(a, b)] += 1
    parent = {}
    for b in freq:
        best = max(((co[(a, b)] / cnt[b], cnt[a], a) for a in freq
                    if a != b and cnt[a] > cnt[b]), default=None)
        if best and best[0] >= PARENT_CONF:
            parent[b] = best[2]
    def root(l):
        seen = set()
        while l in parent and l not in seen:
            seen.add(l); l = parent[l]
        return l
    regions = {l for l in freq if l not in parent and cnt[l] >= MIN_REGION_FRAC * n}
    out = []
    for ls in labels:
        roots = [root(l) for l in ls if l in freq]
        roots = [r for r in roots if r in regions]
        out.append(Counter(roots).most_common(1)[0][0] if roots else "")
    return out, regions

def _learn_map(labels, regions, min_count):
    """label -> region when the label almost always co-occurs with one region."""
    tally = defaultdict(Counter)
    for ls, r in zip(labels, regions):
        if r:
            for l in set(ls):
                tally[l][r] += 1
    out = {}
    for l, c in tally.items():
        tot = sum(c.values())
        r, k = c.most_common(1)[0]
        if tot >= min_count and k / tot >= MAP_CONF:
            out[l] = r
    return out

def _apply(labels, regions, mapping):
    out = []
    for ls, r in zip(labels, regions):
        if not r:
            votes = Counter(mapping[l] for l in ls if l in mapping)
            r = votes.most_common(1)[0][0] if votes else ""
        out.append(r)
    return out

def assign_regions(s1: pd.DataFrame, s23: pd.DataFrame, verbose=True):
    """Returns copies of s1/s23 with `state` filled where it can be inferred."""
    s1, s23 = s1.copy(), s23.copy()
    for country in sorted(set(s1.country)):
        m1 = (s1.country == country).values
        m23 = (s23.country == country).values
        lab1 = _labels(s1.place_labels.values[m1])
        lab23 = _labels(s23.place_labels.values[m23])
        st1 = list(s1.state.values[m1])
        st23 = list(s23.state.values[m23])
        n1 = len(lab1)
        minc = max(MIN_LABEL_COUNT, int(MIN_LABEL_FRAC * n1))
        stated = np.mean([bool(x) for x in st1]) if n1 else 0
        if stated < TABLE_COUNTRY_MIN_STATED:          # no state table: induce
            induced, regs = _induce_s1_regions(lab1)
            st1 = [a or b for a, b in zip(st1, induced)]
            how = f"induced {len(regs)} regions"
        else:
            how = "state table"
        mapping = _learn_map(lab1, st1, minc)                          # step 2
        st1 = _apply(lab1, st1, mapping)
        st23_known = _apply(lab23, st23, mapping)
        mapping.update({k: v for k, v in _learn_map(lab23, st23_known, minc).items()
                        if k not in mapping})                          # step 3
        new23 = _apply(lab23, st23, mapping)                           # step 4
        before = np.mean([bool(x) for x in st23]) if st23 else 0
        after = np.mean([bool(x) for x in new23]) if new23 else 0
        s1.loc[m1, "state"] = st1
        s23.loc[m23, "state"] = new23
        if verbose:
            print(f"  geo[{country}]: {how}; S1 with region {np.mean([bool(x) for x in st1]):.1%}; "
                  f"S2/S3 with region {before:.1%} -> {after:.1%}")
    return s1, s23
