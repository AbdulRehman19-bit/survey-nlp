"""Quality report of one run's themes, so a weak run is visible without reading the results.

Per theme: how many of its clauses actually use the word(s) of its name (name coverage), how often its main words occur together
(coherence, NPMI), its size, whether its name is an adjective. Per run: size spread, the General comment share, how many themes are
below the coverage bar, how many pairs of themes look like duplicates. Everything is deterministic and local."""
import json
import re
import numpy as np
import pandas as pd

DEFAULT_BAR = 0.25                                              # themes.name_min_coverage
DEFAULT_MERGE_SIM, DEFAULT_MERGE_JACCARD = 0.85, 0.4            # themes.merge_sim / themes.merge_jaccard (a pair above either is a duplicate)


_DEGREE = {"too", "very", "so", "really", "quite", "bit", "slightly", "rather", "extremely", "overly", "pretty", "little", "not",
           "is", "was", "are", "were", "be", "feels", "feel", "felt", "seems", "looks", "looked", "smells", "tastes", "tasted"}


class WordUsage:
    """How the words of a survey are used: for each word, how often it comes right after a degree or linking word. Needs no language model."""

    def __init__(self, lowered_texts):
        self._predicative = {}                                  # word -> (uses, uses right after a degree / linking word)
        for t in lowered_texts:
            toks = re.findall(r"[a-z']+", t)
            for i, w in enumerate(toks):
                n, k = self._predicative.get(w, (0, 0))
                self._predicative[w] = (n + 1, k + (i > 0 and toks[i - 1] in _DEGREE))

    def predicative_share(self, word):
        """(uses, share of them right after "too / very / so / is / feels ..."): high for "greasy" ("is greasy", "too greasy"),
        close to 0 for a noun ("the texture is ..."). Plural / singular forms are pooled."""
        n = k = 0
        for w in {word, word.rstrip("s"), word + "s"}:
            a, b = self._predicative.get(w, (0, 0))
            n, k = n + a, k + b
        return n, (k / n if n else 0.0)


class ClauseIndex:
    """The clauses of a survey, lemmatized once, so the question "which clauses use this word or phrase?" is cheap."""

    def __init__(self, texts, nlp):
        self.nlp = nlp
        texts = [str(t) for t in texts]
        self.low = pd.Series([t.lower() for t in texts], dtype=object)
        self.lemma = pd.Series([" ".join(t.lemma_.lower() for t in d if t.is_alpha) for d in nlp.pipe(texts, batch_size=256)], dtype=object)
        self._masks = {}
        self.usage = WordUsage(self.low)

    def predicative_share(self, word):
        return self.usage.predicative_share(word)

    def mask(self, term):
        """Boolean array over the clauses: the lemma (or lemma phrase) of `term` occurs as whole words, or the clause has a form of it
        that the theme naming also counts ("crunch" is used by "crunchy", "pieces" by "piece")."""
        term = term.lower().strip()
        if term not in self._masks:
            from .themes import _stem_rx
            lem = " ".join(t.lemma_.lower() for t in self.nlp(term) if t.is_alpha)
            hit = self.low.str.contains(_stem_rx(term), regex=True)
            if lem:
                hit = hit | self.lemma.str.contains(r"(?<!\w)" + re.escape(lem) + r"(?!\w)", regex=True)
            self._masks[term] = hit.to_numpy()
        return self._masks[term]


def name_terms(name):
    """The words a theme name stands for: "Packaging & Price" -> ["packaging", "price"]; the "2" added to a repeated name is dropped."""
    return [p.strip().lower() for p in re.sub(r"\s+\d+$", "", str(name)).split(" & ") if p.strip()]


def name_coverage(index, name, members, aliases=()):
    """Share of the theme's clauses (`members`: positions in the index) that contain its name (any part of a joined name).
    `aliases` are words the name stands for: an aspect label ("Texture") that replaced a descriptor ("crispy") is covered by both."""
    members = np.asarray(members, int)
    if not len(members):
        return 0.0
    hit = np.zeros(len(index.low), bool)
    for t in [*name_terms(name), *aliases]:
        hit |= index.mask(t)
    return float(hit[members].mean())


def name_rates(index, name, members, aliases=()):
    """(share of the theme's clauses that use the name, share of ALL OTHER clauses of the survey that use it)."""
    members = np.asarray(members, int)
    hit = np.zeros(len(index.low), bool)
    for t in [*name_terms(name), *aliases]:
        hit |= index.mask(t)
    inside = np.zeros(len(hit), bool)
    inside[members] = True
    out = hit[~inside]
    return (float(hit[members].mean()) if len(members) else 0.0), (float(out.mean()) if len(out) else 0.0)


def name_lift(inside, outside):
    """How much more the theme uses its name than the rest of the survey: 6 = six times as often. Independent of how big the theme is,
    unlike a comparison with the whole survey (a name cannot beat 1 / theme share there)."""
    return inside / (outside + 0.02)


def name_ok(inside, outside, floor, min_lift=None):
    """A name fits its theme when the theme uses it at least `floor` of the time and, with `min_lift`, clearly more than the rest of the
    survey does. Both judge the same word: a name used by 15% of a theme and 1% of everything else is good; one used by 40% of the
    theme and 35% of the rest tells nothing."""
    return inside >= floor and (min_lift is None or name_lift(inside, outside) >= min_lift)


def npmi(masks):
    """Mean normalised pointwise mutual information over all pairs of words (each a boolean mask over the clauses). 1 = the words always
    occur together, 0 = independent, -1 = never together. Words that occur nowhere are left out; None when fewer than two remain."""
    masks = [m for m in masks if m.any()]
    vals = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            pi, pj, pij = masks[i].mean(), masks[j].mean(), (masks[i] & masks[j]).mean()
            if pij <= 0:
                vals.append(-1.0)
            elif pij >= 1:
                vals.append(1.0)
            else:
                vals.append(float(np.log(pij / (pi * pj)) / -np.log(pij)))
    return float(np.mean(vals)) if vals else None


def gini(sizes):
    """0 = all themes the same size, towards 1 = one theme holds everything."""
    x = np.asarray(sizes, float)
    if len(x) < 2 or x.sum() == 0:
        return 0.0
    return float(np.abs(x[:, None] - x[None, :]).sum() / (2 * len(x) ** 2 * x.mean()))


def keyword_tokens(kws):
    return {w.rstrip("s") if len(w) > 3 else w for k in kws for w in re.findall(r"[a-z']+", k.lower())}


def keyword_jaccard(a, b):
    """Overlap of two themes' main words (phrases are split into words, plurals folded)."""
    sa, sb = keyword_tokens(a), keyword_tokens(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def is_adjective(head, pos, index, min_uses=5, min_share=0.2):
    """Is this word used as a describing word in the answers? Yes when the learned part-of-speech table has it more often as an
    adjective than as a noun, or when at least `min_share` of its uses come right after a degree / linking word ("too greasy", "is rancid").
    The tagger alone is not enough: in short survey remarks it tags "greasy" and "rancid" as nouns."""
    n, a = pos.get(head, (0.0, 0.0))
    uses, share = index.predicative_share(head)
    return a > n or (uses >= min_uses and share >= min_share)


def is_noun_head(head, pos, usage, tag=None):
    """Can this word be the head of a theme's name, i.e. a thing the answers talk about? Not when it is used as a describing word
    (`is_adjective`), and not when the answers use it as a noun less than half the time ("away"); a word the answers never use
    is judged by its tag in isolation, `tag`, when given."""
    if is_adjective(head, pos, usage):
        return False
    if head in pos:
        return pos[head][0] >= 0.5
    return tag in ("NOUN", "PROPN") if tag else True


def _is_adjective(name, pos, index):
    """A name that is only a describing word ("Crispy"): the head word of every part of a joined name must be one."""
    parts = name_terms(name)
    return bool(parts) and all(is_adjective(t.split()[-1], pos, index) for t in parts)


def _scope_report(scope, clauses, assign, themes, C, cfg, index, pos):
    c = cfg["themes"]
    bar = c.get("quality_bar", DEFAULT_BAR)                  # a fixed yardstick, so runs with different naming settings stay comparable
    n = len(clauses)
    position = {cid: i for i, cid in enumerate(clauses["clause_id"].tolist())}
    docs = pd.factorize(clauses["text_key"])[0] if "text_key" in clauses else np.arange(n)
    per_answer = lambda m: np.bincount(docs, weights=m) > 0               # words co-occur in an ANSWER (a single clause is too short)
    rows = []
    for j, t in enumerate(themes):
        ids = assign.loc[assign["theme_idx"] == j, "clause_id"].unique()
        members = np.array(sorted(position[i] for i in ids if i in position), int)
        general, required = bool(t.get("general")), bool(t.get("required") or t.get("seeded"))
        rates = name_rates(index, t["theme_name"], members, t.get("name_aliases") or ())
        kw_masks = [per_answer(index.mask(k)) for k in (t.get("keywords") or [])[:c.get("n_keywords", 8)]]
        row = {"scope": scope, "theme": t["theme_name"], "general": general, "required": required, "mixed": bool(t.get("name_mixed")),
               "size": int(len(members)), "share": round(len(members) / n, 4) if n else 0.0,
               "name_coverage": round(rates[0], 4), "name_lift": round(name_lift(*rates), 2),
               "name_ok": bool(general or name_ok(*rates, c.get("name_min_coverage", DEFAULT_BAR), c.get("name_min_lift"))),
               "coherence_npmi": npmi(kw_masks), "adjective_name": False if general else _is_adjective(t["theme_name"], pos, index),
               "keywords": ", ".join((t.get("keywords") or [])[:c.get("n_keywords", 8)])}
        if row["coherence_npmi"] is not None:
            row["coherence_npmi"] = round(row["coherence_npmi"], 4)
        rows.append(row)
    # near-duplicates: among the themes found in the data (yours and General comment are never merged)
    free = [j for j, t in enumerate(themes) if not (t.get("general") or t.get("required") or t.get("seeded"))]
    pairs = []
    nearest = {}
    for a in range(len(themes)):
        others = [b for b in range(len(themes)) if b != a]
        if others and C is not None and len(C):
            sims = C[others] @ C[a]
            k = int(np.argmax(sims))
            nearest[a] = (themes[others[k]]["theme_name"], float(sims[k]))
    for x, a in enumerate(free):
        for b in free[x + 1:]:
            cos = float(C[a] @ C[b]) if C is not None and len(C) else 0.0
            jac = keyword_jaccard(themes[a].get("keywords") or [], themes[b].get("keywords") or [])
            if cos >= c.get("merge_sim", DEFAULT_MERGE_SIM) or jac >= c.get("merge_jaccard", DEFAULT_MERGE_JACCARD):
                pairs.append({"a": themes[a]["theme_name"], "b": themes[b]["theme_name"], "centre_cosine": round(cos, 3), "keyword_jaccard": round(jac, 3)})
    for j, r in enumerate(rows):
        r["nearest_theme"], r["nearest_cosine"] = (nearest[j][0], round(nearest[j][1], 3)) if j in nearest else (None, None)
    spec = [r for r in rows if not r["general"]]
    sizes = [r["size"] for r in spec]
    found = [r for r in spec if not r["required"]]
    covered = np.zeros(n, bool)
    covered[[position[i] for i in assign["clause_id"].unique() if i in position]] = True
    big = max(spec, key=lambda r: r["size"]) if spec else None
    cov = [r["name_coverage"] for r in spec]
    coh = [r["coherence_npmi"] for r in spec if r["coherence_npmi"] is not None]
    summary = {
        "scope": scope, "n_clauses": n, "n_themes": len(spec), "n_found_themes": len(found),
        "name_coverage_bar": bar, "n_below_bar": sum(r["name_coverage"] < bar and not r["mixed"] for r in found),
        "n_mixed": sum(r["mixed"] for r in spec), "n_weak_names": sum((not r["name_ok"]) and not r["mixed"] for r in found),
        "name_coverage_min": round(min(cov), 4) if cov else None, "name_coverage_mean": round(float(np.mean(cov)), 4) if cov else None,
        "coherence_npmi_mean": round(float(np.mean(coh)), 4) if coh else None,
        "largest_theme": big["theme"] if big else None, "largest_share": big["share"] if big else None,
        "gini": round(gini(sizes), 4),
        "general_share": round(sum(r["size"] for r in rows if r["general"]) / n, 4) if n else None,
        "unassigned_share": round(float(1 - covered.mean()), 4) if n else None,
        "n_adjective_names": sum(r["adjective_name"] for r in found),
        "n_near_duplicate_pairs": len(pairs)}
    return {"summary": summary, "themes": rows, "near_duplicates": pairs}


def compute(art, cfg, masks):
    """The report for a finished run. `art` has segment / assign / themes; `masks` maps scope -> boolean mask over the segment table."""
    import spacy
    from .themes import _pos_table
    nlp = spacy.load(cfg["themes"].get("spacy_model", "en_core_web_sm"), disable=["ner", "parser"])
    seg = art["segment"]
    pos = _pos_table(seg, nlp)
    out = {}
    for scope, m in masks.items():
        themes, C = art["themes"][scope]
        clauses = seg[m].reset_index(drop=True)
        out[scope] = _scope_report(scope, clauses, art["assign"][art["assign"]["scope"] == scope], themes, C, cfg, ClauseIndex(clauses["clause"], nlp), pos)
    return {"scopes": out}


def summary_line(report):
    """One line per scope for the command line."""
    lines = []
    for s, r in report["scopes"].items():
        m = r["summary"]
        f = lambda v, p=2: "n/a" if v is None else f"{v:.{p}f}"
        lines.append(f"[quality] {s}: {m['n_themes']} themes | name coverage min {f(m['name_coverage_min'])} mean {f(m['name_coverage_mean'])}, "
                     f"{m['n_below_bar']} below {m['name_coverage_bar']}, {m['n_weak_names']} weak ({m['n_mixed']} mixed) | coherence {f(m['coherence_npmi_mean'])} | "
                     f"largest {m['largest_theme']} {f(m['largest_share'])}, gini {f(m['gini'])} | General {f(m['general_share'])} | "
                     f"{m['n_adjective_names']} adjective names, {m['n_near_duplicate_pairs']} near-duplicate pairs")
    return "\n".join(lines)


def frames(report):
    """(one row per scope, one row per theme) for the workbook."""
    summary = pd.DataFrame([r["summary"] for r in report["scopes"].values()])
    themes = pd.DataFrame([t for r in report["scopes"].values() for t in r["themes"]])
    return summary, themes


def write_json(report, path):
    path.write_text(json.dumps(report, indent=2))
