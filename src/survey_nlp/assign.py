import re
import numpy as np
import pandas as pd
from .util import axis_of, center, sha


_STOP_NAME = {"not", "specific", "the", "and"}


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


def _name_words(themes, c, clauses=None):
    """For every theme, the words of its own name that are distinctive: a theme called "Refreshing Taste" is matched by the word
    "refreshing" but not by "taste" (the name of another theme, shared by many clauses). Only words the discovery found frequent in
    the theme and clearly rarer elsewhere (its assign_keywords) are used, so a name never turns into a noisy keyword.

    assign.name_keyword_max_share (optional, off when absent): a word of a LONGER name that appears in more than this share of all
    clauses is dropped ("taste" in "Original Taste" is in 30% of the clauses, so it would pull every taste remark into Original).
    A theme named by that one word alone keeps it, and so does a name whose words are all common ("White Chocolate").
    Remove the setting to go back to the old behaviour."""
    words = [set(re.findall(r"[a-z']+", t.get("match_name", t["theme_name"]).lower().split(" & ")[0])) for t in themes]       # a joined name: its first topic only. A name chosen later is only a label
    max_share = c.get("name_keyword_max_share")
    text = clauses["clause"].str.lower() if (max_share is not None and clauses is not None and len(clauses)) else None
    share = {}

    def too_common(w):
        if w not in share:
            share[w] = float(text.str.contains(r"\b" + re.escape(w) + r"\b", regex=True).mean())
        return share[w] > max_share
    out = []
    for j, t in enumerate(themes):
        others = set().union(*[w for k, w in enumerate(words) if k != j and len(w) == 1])        # a word that is the whole name of another theme
        ws = [w for w in sorted(words[j]) if len(w) >= c["keyword_min_len"] and w in (t.get("assign_keywords") or []) and w not in others]
        if text is not None and len(words[j]) > 1:
            common = [w for w in ws if too_common(w)]
            if len(common) < len(ws):                       # a distinctive word is left ("original"): the common one ("taste") goes.
                ws = [w for w in ws if w not in common]     # When every word is common ("White Chocolate") they stay: nothing is more specific
        out.append(ws)
    return out


def matching_words(themes, c, clauses=None):
    """The words that really match clauses to each theme (what `run` uses before the precision check): the words a person wrote, otherwise
    the distinctive words of the theme's name. Not the longer list of 'main words' the discovery found, which is only a description."""
    name_kws = _name_words(themes, c, clauses)
    return [list(t.get("curated_keywords") or (t["keywords"] if c.get("use_discovered_keywords") else (name_kws[j] if c.get("name_keywords", True) else [])))
            for j, t in enumerate(themes)]


def _keyword_hits(kws, texts, sim, theme, c, dropped, others=()):
    """The clauses a theme's keywords match, and the keywords kept.

    1. A keyword next to a word that names ANOTHER theme is a compound of that theme, not this one: "milk" in "milk chocolate" or "milk flavour"
       belongs to Chocolate or Flavour, so those clauses do not count as hits of the Milk theme ("others" = the other themes' name words).
    2. assign.keyword_min_precision: of the remaining clauses that contain the keyword (at least assign.keyword_precision_min_clauses, default 5),
       the share close in meaning to the theme (similarity at least assign.min_sim) must reach the threshold, otherwise the keyword is left out
       ("product", "small" in a Packaging theme would drag every answer that has the word into it). Also for words a person wrote. The left-out
       words are listed in the result's `attrs["dropped_keywords"]` so the page can say so."""
    thr = c.get("keyword_min_precision")
    need = c.get("keyword_precision_min_clauses", 5)
    mask = np.zeros(len(texts), bool)
    keep = []
    near_rx = None
    if others:
        alt = "|".join(re.escape(o) for o in sorted(others, key=len, reverse=True))
    for k in kws:
        hit = texts.str.contains(_rx([k]), regex=True).to_numpy()
        if others and not k.startswith("re:"):
            ke = re.escape(k)
            near = texts.str.contains(r"\b(?:" + alt + r")[ -]" + ke + r"\b|\b" + ke + r"[ -](?:" + alt + r")\b", regex=True, case=False).to_numpy()
            hit = hit & ~near
        n = int(hit.sum())
        if thr and n >= need and float((sim[hit] >= c["min_sim"]).mean()) < thr:
            dropped.setdefault(theme, []).append((k, n, int((sim[hit] >= c["min_sim"]).sum())))
            continue
        keep.append(k)
        mask |= hit
    return mask, keep


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
    for j, t in enumerate(themes):                      # a merged theme is as close as the closest of the themes it was made of
        if t.get("sub_centres"):
            S[:, j] = np.maximum(S[:, j], (E @ np.asarray(t["sub_centres"], np.float32).T).max(1))
    n = len(S)
    r = np.arange(n)
    k = max(2, min(int(c.get("max_themes_per_clause") or 2), S.shape[1]))     # how many themes a clause may reach by meaning
    order = np.argsort(-S, axis=1)[:, :k] if S.shape[1] > 1 else np.zeros((n, 2), int)
    s1, s2 = S[r, order[:, 0]], S[r, order[:, 1]]
    A = np.zeros(S.shape, bool)
    A[r, order[:, 0]] = s1 >= c["min_sim"]
    for rank in range(1, order.shape[1]):                                      # further themes: close enough AND within the margin of the best one
        sr = S[r, order[:, rank]]
        sec = (sr >= c.get("secondary_min_sim", c["min_sim"])) & ((s1 - sr) <= c["secondary_margin"]) & (order[:, rank] != order[:, 0])
        A[r[sec], order[sec, rank]] = True
    K = np.zeros_like(A)
    trusted = np.zeros(S.shape[1], bool)                # themes with words a person wrote: those words match whatever the similarity
    name_kws = _name_words(themes, c, clauses)
    dropped = {}                                        # theme -> [(keyword, clauses it appears in, of those close to the theme)]
    name_words = [set(re.findall(r"[a-z']{3,}", t.get("match_name", t["theme_name"]).lower())) for t in themes]
    name_words = [w - _STOP_NAME for w in name_words]
    for j, t in enumerate(themes):                      # whole-word keyword match, case-insensitive
        # Only words a person wrote for the theme (required themes, curation) match by keyword. The words discovered from the
        # data are right only about 1 time in 3 and made a clause match 3.6 themes on average; set
        # assign.use_discovered_keywords: true to bring them back.
        kws = t.get("curated_keywords") or (t["keywords"] if c.get("use_discovered_keywords") else
                                            (name_kws[j] if c.get("name_keywords", True) else []))
        kws = [k for k in kws if k.startswith("re:") or len(k) >= c["keyword_min_len"]]
        own = set(name_words[j])
        others = set().union(*[w for k2, w in enumerate(name_words) if k2 != j]) - own
        mask, kws = _keyword_hits(kws, clauses["clause"], S[:, j], t["theme_name"], c, dropped, others)
        if kws:                                         # a keyword starting with "re:" is a raw regex
            K[:, j] = mask
            trusted[j] = bool(t.get("curated_keywords"))
            if not trusted[j]:                          # a word taken from the theme's name only counts when the clause is also close in meaning
                K[:, j] &= S[:, j] >= c.get("keyword_min_sim", 0.25)   # ("pieces" in "mostly white pieces and they go slimy in milk" is not about Pieces)
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
        keep = (rank < cap) | (K[ii, jj] & trusted[jj])  # a word you wrote for a theme always counts: it is never pushed out by the cap
        ii, jj = ii[keep], jj[keep]
    if general and "text_key" in clauses:               # answer level too: a general remark next to a specific theme in the same answer is dropped
        tk = clauses["text_key"].to_numpy()
        is_gen = np.isin(jj, general)
        has_spec = set(tk[ii[~is_gen]].tolist())
        keep = ~(is_gen & np.array([k in has_spec for k in tk[ii]], bool))
        ii, jj = ii[keep], jj[keep]
    via = np.where(A[ii, jj] & K[ii, jj], "both",
                   np.where(A[ii, jj], "semantic", np.where(K[ii, jj], "keyword", "fallback")))
    out = pd.DataFrame({"clause_id": clauses.clause_id.to_numpy()[ii], "theme_idx": jj,
                        "sim": S[ii, jj], "via": via})
    out.attrs["dropped_keywords"] = dropped
    return out
