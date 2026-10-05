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
        if not kws:
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
    ii, jj = np.nonzero(A | K)
    via = np.where(A[ii, jj] & K[ii, jj], "both", np.where(A[ii, jj], "semantic", "keyword"))
    return pd.DataFrame({"clause_id": clauses.clause_id.to_numpy()[ii], "theme_idx": jj,
                         "sim": S[ii, jj], "via": via})
