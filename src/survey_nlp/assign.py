import re
import numpy as np
import pandas as pd
from .util import axis_of, center, sha


def _rx(kws):
    """Case-insensitive whole-word pattern. A keyword starting with "re:" is used as a raw regex."""
    parts = [k[3:] if k.startswith("re:") else r"\b" + re.escape(k) + r"\b" for k in kws]
    return re.compile("|".join(parts), re.I)


def _general_sim(E, cfg):
    """Cosine of each clause (raw embedding) to its closest example of a general comment ("It is good", "I like it")."""
    from . import embed
    ex = list(cfg["assign"]["general_examples"])
    V = embed.encode_cached([sha("general:" + x.lower()) for x in ex], ex, cfg)
    return (E @ V.T).max(1)


def _name_words(themes, c):
    """For every theme, the words of its own name that are distinctive: a theme called "Refreshing Taste" is matched by the word
    "refreshing" but not by "taste" (the name of another theme, shared by many clauses). Only words the discovery found frequent in
    the theme and clearly rarer elsewhere (its assign_keywords) are used, so a name never turns into a noisy keyword."""
    words = [set(re.findall(r"[a-z']+", t["theme_name"].lower())) for t in themes]
    out = []
    for j, t in enumerate(themes):
        others = set().union(*[w for k, w in enumerate(words) if k != j and len(w) == 1])        # a word that is the whole name of another theme
        out.append([w for w in sorted(words[j]) if len(w) >= c["keyword_min_len"] and w in (t.get("assign_keywords") or []) and w not in others])
    return out


def run(clauses, E, themes, C, cfg) -> pd.DataFrame:
    """clauses/E may be a subset (a scope); clause_id values are preserved in the output.

    A theme with `general: true` (e.g. "Non-specific") is the home of generic remarks. When one exists, a clause that matches
    nothing else is routed to it instead of being dropped, and the general_examples exclusion below is not used (those
    clauses are wanted in that theme, not left without one)."""
    c = cfg["assign"]
    E_raw = E
    if c.get("center"):                                 # themes.center: centres were built in this centred space, so score in it
        E = center(E, strength=cfg["themes"].get("center_strength", 1.0), axis=axis_of(cfg))
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
    name_kws = _name_words(themes, c)
    for j, t in enumerate(themes):                      # whole-word keyword match, case-insensitive
        # Only words a person wrote for the theme (required themes, curation) match by keyword. The words discovered from the
        # data are right only about 1 time in 3 and made a clause match 3.6 themes on average; set
        # assign.use_discovered_keywords: true to bring them back.
        kws = t.get("curated_keywords") or (t["keywords"] if c.get("use_discovered_keywords") else
                                            (name_kws[j] if c.get("name_keywords", True) else []))
        kws = [k for k in kws if k.startswith("re:") or len(k) >= c["keyword_min_len"]]
        if kws:                                         # a keyword starting with "re:" is a raw regex
            K[:, j] = clauses["clause"].str.contains(_rx(kws), regex=True).to_numpy()
    general = [j for j, t in enumerate(themes) if t.get("general")]
    if c.get("general_examples") and not general:       # generic remarks are left without a theme (only when no theme wants them)
        left = _general_sim(E_raw, cfg) >= c.get("general_min_sim", 0.9)
        A[left] = False
    F = np.zeros_like(A)
    fb = c.get("fallback_min_sim")
    spec_cols = np.ones(S.shape[1], bool)
    spec_cols[general] = False
    if fb is not None and spec_cols.any():              # a clause no specific theme matched joins its best specific theme if it is at all close
        lone = ~(A | K).any(axis=1)                     # nothing matched, the general theme included
        S_spec = np.where(spec_cols[None, :], S, -np.inf)
        best_spec = S_spec.argmax(1)
        lone &= S_spec[r, best_spec] >= fb
        if general:                                     # a clause closer to the general theme than to any specific one stays general
            lone &= S_spec[r, best_spec] >= S[:, general].max(axis=1)
        F[r[lone], best_spec[lone]] = True
    elif fb is not None:
        lone = ~(A | K).any(axis=1) & (s1 >= fb)
        F[r[lone], order[lone, 0]] = True
    if general:                                         # the general theme is only for clauses no specific theme took
        taken = (A | K | F)[:, spec_cols].any(axis=1)
        for g in general:
            A[taken, g] = K[taken, g] = F[taken, g] = False
    if general and c.get("general_catch_all", True):    # whatever is still unmatched is a generic remark
        rest = ~(A | K | F).any(axis=1)
        F[r[rest], general[0]] = True
    ii, jj = np.nonzero(A | K | F)
    cap = c.get("max_themes_per_clause")
    if cap:                                             # keep each clause's strongest themes: similarity + a bonus per kind of match
        strength = S[ii, jj] + 0.15 * A[ii, jj] + 0.15 * K[ii, jj]
        if general:                                     # the general theme only fills a place no specific theme wants
            strength = strength - 0.5 * np.isin(jj, general)
        order_ = np.lexsort((-strength, ii))            # by clause, then strongest first
        rank = np.zeros(len(ii), int)
        first = np.r_[True, ii[order_][1:] != ii[order_][:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(ii)), 0))
        rank[order_] = np.arange(len(ii)) - start
        keep = (rank < cap) | K[ii, jj]                 # a word you wrote for a theme always counts: it is never pushed out by the cap
        ii, jj = ii[keep], jj[keep]
    if general and "text_key" in clauses:               # answer level too: a general remark next to a specific theme in the same answer is dropped
        tk = clauses["text_key"].to_numpy()
        is_gen = np.isin(jj, general)
        has_spec = set(tk[ii[~is_gen]].tolist())
        keep = ~(is_gen & np.array([k in has_spec for k in tk[ii]], bool))
        ii, jj = ii[keep], jj[keep]
    via = np.where(A[ii, jj] & K[ii, jj], "both",
                   np.where(A[ii, jj], "semantic", np.where(K[ii, jj], "keyword", "fallback")))
    return pd.DataFrame({"clause_id": clauses.clause_id.to_numpy()[ii], "theme_idx": jj,
                         "sim": S[ii, jj], "via": via})
