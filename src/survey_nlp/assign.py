import re
import numpy as np
import pandas as pd
from .util import center


def _rx(kws):
    """Case-insensitive whole-word pattern. A keyword starting with "re:" is used as a raw regex."""
    parts = [k[3:] if k.startswith("re:") else r"\b" + re.escape(k) + r"\b" for k in kws]
    return re.compile("|".join(parts), re.I)


def run(clauses, E, themes, C, cfg) -> pd.DataFrame:
    """clauses/E may be a subset (a scope); clause_id values are preserved in the output."""
    c = cfg["assign"]
    if c.get("center"):                                 # must match themes.center (same clause set -> same mean)
        E = center(E)
    S = E @ C.T                                         # all clauses x all themes in one matmul
    n = len(S)
    r = np.arange(n)
    order = np.argsort(-S, axis=1)[:, :2] if S.shape[1] > 1 else np.zeros((n, 2), int)
    s1, s2 = S[r, order[:, 0]], S[r, order[:, 1]]
    A = np.zeros(S.shape, bool)
    A[r, order[:, 0]] = s1 >= c["min_sim"]
    sec = (s2 >= c["min_sim"]) & ((s1 - s2) <= c["secondary_margin"]) & (order[:, 1] != order[:, 0])
    A[r[sec], order[sec, 1]] = True
    K = np.zeros_like(A)
    min_prec = c.get("keyword_min_precision", 0.0)
    for j, t in enumerate(themes):                      # whole-word keyword match, case-insensitive
        kws = list(t.get("curated_keywords", []))       # hand-written keywords REPLACE the discovered ones
        if not kws and not c.get("use_discovered_keywords", False):
            kws = []                                    # default: match by meaning only. Discovered keywords were right only ~1 time in 3
                                                        # when they were the sole reason for a match; hand-written ones stay in force
        elif not kws and "assign_keywords" in t:
            kws = list(t["assign_keywords"])            # discovered, already filtered for being distinctive of this theme
        elif not kws:
            for k in t["keywords"]:
                if len(k) < c["keyword_min_len"]:
                    continue
                hit = clauses["clause"].str.contains(_rx([k]), regex=True).to_numpy()
                # generic words ("good", "quite") match clauses of every theme: keep a discovered keyword only if
                # most clauses containing it have this theme as their top-1 semantic theme
                if hit.any() and (order[hit, 0] == j).mean() >= min_prec:
                    kws.append(k)
        if kws:
            K[:, j] = clauses["clause"].str.contains(_rx(kws), regex=True).to_numpy()
    F = np.zeros_like(A)
    fb = c.get("fallback_min_sim")
    if fb is not None:                                  # a clause nothing else matched joins its best theme if it is at all close
        lone = ~(A | K).any(axis=1) & (s1 >= fb)
        F[r[lone], order[lone, 0]] = True
    ii, jj = np.nonzero(A | K | F)
    cap = c.get("max_themes_per_clause")
    if cap:                                             # keep each clause's strongest themes: similarity + a bonus per kind of match
        strength = S[ii, jj] + 0.15 * A[ii, jj] + 0.15 * K[ii, jj]
        order_ = np.lexsort((-strength, ii))            # by clause, then strongest first
        rank = np.zeros(len(ii), int)
        first = np.r_[True, ii[order_][1:] != ii[order_][:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(ii)), 0))
        rank[order_] = np.arange(len(ii)) - start
        keep = rank < cap
        ii, jj = ii[keep], jj[keep]
    via = np.where(A[ii, jj] & K[ii, jj], "both",
                   np.where(A[ii, jj], "semantic", np.where(K[ii, jj], "keyword", "fallback")))
    return pd.DataFrame({"clause_id": clauses.clause_id.to_numpy()[ii], "theme_idx": jj,
                         "sim": S[ii, jj], "via": via})
