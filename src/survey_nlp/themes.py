import inspect, json, re
from pathlib import Path
import numpy as np
import spacy
import umap
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from . import quality
from .util import axis_of, center, detail, sha


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _cluster(X, c, seed):
    """Returns (labels, idx) where idx indexes into X. label -1 = noise."""
    Z = umap.UMAP(n_components=c["umap_dims"], n_neighbors=c["umap_neighbors"],
                  min_dist=0.0, metric="cosine", random_state=seed).fit_transform(X)
    mcs = max(c["min_cluster_size"], int(c["min_cluster_frac"] * len(X)))
    lab = HDBSCAN(min_cluster_size=mcs, min_samples=c["min_samples"]).fit_predict(Z)
    if len(set(lab) - {-1}) >= c["min_themes"]:
        return lab, np.arange(len(X))
    # Fallback: agglomerative, k chosen by silhouette, on a subsample (O(n^2) memory)
    rng = np.random.default_rng(seed)
    sub = np.arange(len(X))
    if len(X) > c["fallback_max_clauses"]:
        sub = rng.choice(len(X), c["fallback_max_clauses"], replace=False)
    Xs = X[sub]
    best, best_s = None, -1.0
    for k in range(c["fallback_k_range"][0], c["fallback_k_range"][1] + 1):
        l = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(Xs)
        s = silhouette_score(Xs, l, metric="cosine", sample_size=min(5000, len(Xs)), random_state=seed)
        if s > best_s:
            best, best_s = l, s
    return best, sub


_GENERIC_NAMES = {"product", "products", "thing", "things", "stuff", "bit", "lot", "time", "way", "kind", "overall", "item", "one", "ones"}
# Words that judge or pad but do not name a topic. They are never shown as a theme's main words or used as its name.
_NOISE = _GENERIC_NAMES | {
    "good", "great", "nice", "bad", "better", "best", "worse", "worst", "fine", "okay", "decent", "lovely", "wonderful", "amazing",
    "awesome", "excellent", "perfect", "terrible", "horrible", "awful", "unpleasant", "pleasant", "enjoyable", "balanced", "easy",
    "hard", "dull", "love", "loved", "like", "liked", "likes", "hate", "hated", "enjoy", "enjoyed", "favourite", "favorite",
    "really", "very", "quite", "just", "also", "much", "many", "even", "still", "well", "little", "feel", "feels", "feeling",
    "left", "made", "make", "makes", "getting", "putting", "going", "think", "need", "want", "something", "everything",
    "doesn", "didn", "isn", "wasn", "couldn", "wouldn", "don", "aren", "hasn", "haven", "anything", "nothing", "people", "meal", "meals", "day", "buy", "bought", "long", "different", "same", "bit", "drink", "drinks"}


# Entries that were added to _NOISE while tuning on food and drink surveys. They are not judging words: in another survey "meal" or "buy" can be a topic.
_DOMAIN_FLAVOURED = {"meal", "meals", "day", "buy", "bought", "drink", "drinks", "people"}
_LANGUAGE_NOISE = _NOISE - _DOMAIN_FLAVOURED


def _opinion_words(rows_terms, mu, c, cfg):
    """Judging words found IN THE DATA: the content words of the survey whose meaning lies unusually far along the positive-negative axis
    ("nice", "unpleasant", "worse"), i.e. more than themes.filler_k standard deviations above the mean closeness of all its words. Relative to
    the survey's own vocabulary, so no word list and no fixed distance. Plural forms are included. Empty without a sentiment axis."""
    ax = axis_of(cfg) if cfg is not None else None
    df = {}
    for ts in rows_terms:
        for w in set(ts):
            df[w] = df.get(w, 0) + 1
    words = sorted(w for w, n in df.items() if n >= c.get("filler_min_df", 3))
    if ax is None or len(words) < 8:
        return set()
    from . import embed
    V = embed.encode_cached([sha("name:" + w) for w in words], words, cfg)
    p = np.abs(V @ ax)
    z = (p - p.mean()) / (p.std() + 1e-12)
    out = {w for w, zi in zip(words, z) if zi > c.get("filler_k", 2.0)}
    return out | {w + "s" for w in out}


def _noise_for(clauses, rows, c, cfg, mu=None):
    """(words that never name or describe a theme, generic names) for this survey, by themes.filler_mode:
    fixed = the original fixed list; language = that list without the food / drink flavoured entries; data = judging words found in the
    data (`_opinion_words`) only, generic nouns being left to the no-lift rule and the generic-theme fold; hybrid = data + the generic nouns."""
    mode = c.get("filler_mode", "fixed")
    if mode == "fixed":
        return _NOISE, _GENERIC_NAMES
    if mode == "language":
        return _LANGUAGE_NOISE, _GENERIC_NAMES
    nlp = spacy.load(c["spacy_model"], disable=["ner", "parser", "lemmatizer"])
    terms = [_clause_terms(d, frozenset()) for d in nlp.pipe(clauses["clause"].iloc[rows].tolist(), batch_size=256)]
    found = frozenset(_opinion_words(terms, mu, c, cfg))
    if mode == "hybrid":                                            # judging words from the data + only the closed set of generic nouns
        return found | _GENERIC_NAMES, _GENERIC_NAMES
    return found, frozenset()


def _pos_table(clauses, nlp, limit=6000):
    """For every word, how often it is a noun / proper noun / adjective in the answers: {word: (noun share, adjective share)}.
    "lingers" in "the aftertaste lingers" is a verb, "packaging" is a noun; a theme is named after nouns and plain attributes only."""
    texts = clauses["clause"].tolist()[:limit]
    cnt = {}
    for doc in nlp.pipe(texts, batch_size=256):
        for tok in doc:
            n, a, t = cnt.get(tok.lower_, (0, 0, 0))
            cnt[tok.lower_] = (n + (tok.pos_ in ("NOUN", "PROPN")), a + (tok.pos_ == "ADJ"), t + 1)
    return {w: (n / t, a / t) for w, (n, a, t) in cnt.items()}


def _clean_terms(ranked, texts, all_texts, pos, c, limit, nlp=None, dedupe=True, noise=None):
    """The informative words of one theme, in rank order. Dropped: judging / padding words, verbs, words used everywhere in the
    survey (no lift), a phrase that repeats a word, and a word the theme already has ("drink", "easy drink", "drink overall"
    keep one entry). A phrase replaces its own word when most of the clauses with the word use the phrase.
    Returns [(term, share in theme, lift)]."""
    noise = _NOISE if noise is None else noise
    kept, stems = [], []
    min_share = c.get("main_word_min_share", 0.06)
    min_lift = min(c.get("main_word_min_lift", 1.3), 0.9 * len(all_texts) / max(1, len(texts)))     # a theme that IS the survey cannot be more distinctive than 1
    for t in ranked:
        parts = t.split()
        if len(t) < 4 or any(p in noise or len(p) < 3 or not p.isalpha() or (nlp is not None and nlp.vocab[p].is_stop) for p in parts) \
                or len(set(parts)) < len(parts):
            continue                                                # judging words, function words ("it's", "doesn"), a repeated word
        ok = True
        for p in parts:
            n, a = pos.get(p, (1.0, 0.0))
            ok &= (n >= 0.5) or (a >= 0.5 and len(parts) == 1) or (a >= 0.5 and p == parts[0])
        if not ok:
            continue
        share = float(texts.str.contains(_stem_rx(t), regex=True).mean())
        lift = share / (float(all_texts.str.contains(_stem_rx(t), regex=True).mean()) + 0.02)
        if share < min_share or lift < min_lift:
            continue
        st = {p.rstrip("s") for p in parts}
        clash = [k for k, s in enumerate(stems) if s & st] if dedupe else []
        if clash:
            k = clash[0]
            if len(parts) > 1 and len(stems[k]) == 1 and share >= 0.6 * kept[k][1]:
                kept[k], stems[k] = (t, share, lift), st                    # the phrase covers most of the word's clauses: use it
            continue
        kept.append((t, share, lift))
        stems.append(st)
        if len(kept) >= limit:
            break
    return kept


def _stem_rx(t):
    """Whole-word pattern that also matches the plural / -ing / -ed form, so "crunch" counts "crunchy" and "pieces" counts "piece"."""
    return r"(?<!\w)" + re.escape(t.rstrip("s") if len(t) > 4 else t) + r"\w{0,3}(?!\w)"


def _pick_name(cand_terms, texts, all_texts, used, used_vecs, nlp, c, Ec, mu, cfg, m, index=None, pos=None, usage=None, generic=None):
    """Returns (name, mixed, alternatives, aliases). The name of one theme: the word or two-word phrase that best SUMMARISES its clauses.
    With `index` (quality.ClauseIndex) and themes.name_min_coverage the name is also CHECKED against the theme's clauses: if the best
    candidate is used by fewer than that share of them, the next-ranked candidates are tried in order (leaving out evaluation words and
    names that repeat an earlier theme's), and when none passes the best one is kept and the theme is reported as mixed.
    With themes.aspect_labels, a best candidate that cannot head a name (a describing word like "greasy", or a word the answers do not use as
    a noun) is replaced by the closest aspect label ("Texture") when the theme's centre is at least themes.aspect_label_min_sim close to it
    (the descriptor is then returned as an alias, so the label counts as covering the clauses that use it); otherwise by the best-ranked
    candidate that can head a name, and only when there is none does the describing word stay.
    A candidate must be a noun phrase (or a plain adjective), used by a fair share of the theme's clauses. It is scored by
      - how close its meaning is to the theme's centre,
      - how many of the theme's clauses use it,
      - how much more the theme uses it than the whole survey (a name must tell this theme from the others),
    and is pushed down when it is a filler word ("product", "bit"), an evaluation word ("good") or means the same as a name
    an earlier theme already took ("Taste" next to "Flavour"). A two-word phrase wins over its head word when most of the
    clauses that use the word use the phrase too ("white chocolate"). When two clearly different topics each cover many clauses
    the name has both ("Packaging & Price") instead of hiding one."""
    min_share = c.get("name_min_share", 0.12)
    cands, share, lift, nouns, masks = [], [], [], [], []
    for t in cand_terms:
        if len(t) < 4 or t.split()[-1] in (_GENERIC_NAMES if generic is None else generic):
            continue                                                # fragments ("don"), filler words
        mask = texts.str.contains(_stem_rx(t), regex=True).to_numpy()
        sh = float(mask.mean())
        if sh < min_share:
            continue
        last = nlp(t)[-1]
        adj = last.pos_ == "ADJ" and " " not in t and c.get("name_adjectives", True)
        if not (last.pos_ in ("NOUN", "PROPN") or adj):
            continue                                                # the head must be a noun, or a plain one-word adjective ("sweet", "fizzy")
        if t.title() in used or any(t.rstrip("s") == u.lower().rstrip("s") for u in used):
            continue
        cands.append(t)
        nouns.append(last.pos_ in ("NOUN", "PROPN"))
        masks.append(mask)
        share.append(sh)
        lift.append(sh / (float(all_texts.str.contains(_stem_rx(t), regex=True).mean()) + 0.02))
        if len(cands) >= c.get("name_candidates", 12):
            break
    if not cands:
        return None, False, [], []
    min_cov = c.get("name_min_coverage", 0.0) if index is not None else 0.0

    def validate(order, bad, say):
        """The first candidate in rank order that enough of the theme's clauses use; the best one, flagged mixed, if none does."""
        best = int(order[0])
        if not min_cov or _name_passes(index, say(best), m, c):
            return best, False
        for i in order[1:]:
            if not bad[i] and _name_passes(index, say(int(i)), m, c):
                return int(i), False
        return best, True

    if Ec is None or cfg is None or not c.get("name_by_meaning", True):
        i, mixed = validate(np.arange(len(cands)), np.zeros(len(cands), bool), lambda i: cands[i].title())
        return cands[i].title(), mixed, [t.title() for t in cands if t.title() != cands[i].title()], []
    from . import embed
    V0 = embed.encode_cached([sha("name:" + t) for t in cands], cands, cfg)
    V = center(V0, mu, c.get("center_strength", 1.0), axis_of(cfg)) if c.get("center") else V0
    centre = _unit(Ec[m].mean(0))
    share, lift = np.array(share), np.array(lift)
    score = V @ centre + c.get("name_share_weight", 0.5) * share + 0.1 * np.clip(np.log(np.maximum(lift, 1e-6)), 0, 2) - 0.01 * np.arange(len(cands))
    for i, t in enumerate(cands):
        parts = t.split()
        if len(parts) > 1:                                          # a phrase beats its own words when it covers most of their clauses
            for j, u in enumerate(cands):
                if u in parts and share[i] >= 0.6 * share[j]:
                    score[i] += 0.12
                    score[j] -= 0.1
        elif nouns[i] is False:
            score[i] -= c.get("name_adj_penalty", 0.12)             # "Packaging" is a better name than "Easy"
    ax = axis_of(cfg)
    bad = np.zeros(len(cands), bool)                                # evaluation words and repeats of an earlier name: never a fallback
    if ax is not None:
        bad |= np.abs(V0 @ ax) >= c.get("name_eval_max", 0.35)
        score = score - 1.0 * (np.abs(V0 @ ax) >= c.get("name_eval_max", 0.35))
    for u, uv in used_vecs.items():                                 # the same meaning as an earlier theme's name
        bad |= (V @ uv) >= c.get("name_dup_sim", 0.8)
        score = score - c.get("name_dup_penalty", 1.0) * np.clip((V @ uv - c.get("name_dup_sim", 0.8)) / 0.4, 0, 1)
    order = np.argsort(-score)

    def say(i):                                                     # "Flavour", not "Flavours", when the singular is also in the theme
        t = cands[i]
        one = t[:-1] if t.endswith("s") else t
        return (one if one != t and one in cands else t).title()
    best, mixed = validate(order, bad, say)
    name, name_vec, aliases = say(best), V[best], []
    labels = c.get("aspect_labels")
    if labels and not mixed and pos is not None and usage is not None:
        heads = [quality.is_noun_head(t.split()[-1], pos, usage, "NOUN" if nouns[i] else "ADJ") for i, t in enumerate(cands)]
        if not heads[best]:
            VL0 = embed.encode_cached([sha("name:" + l.lower()) for l in labels], [l.lower() for l in labels], cfg)
            VL = center(VL0, mu, c.get("center_strength", 1.0), axis_of(cfg)) if c.get("center") else VL0
            VL = VL / (np.linalg.norm(VL, axis=1, keepdims=True) + 1e-12)
            sims = VL @ centre
            free = [int(k) for k in np.argsort(-sims) if labels[k].title() not in used
                    and all(float(VL[k] @ uv) < c.get("name_dup_sim", 0.8) for uv in used_vecs.values())]
            if free and sims[free[0]] >= c.get("aspect_label_min_sim", 0.45):
                name, name_vec, aliases = labels[free[0]].title(), VL[free[0]], [cands[best]]
            else:
                alt = next((int(i) for i in order if heads[i] and not bad[i]
                            and (not min_cov or _name_passes(index, say(int(i)), m, c))), None)
                if alt is not None:
                    best, name, name_vec = alt, say(alt), V[alt]
    used_vecs[name] = name_vec
    if c.get("name_combine", False) and not aliases:
        for j in order[1:]:                                         # a second, clearly different topic that many clauses talk about
            t = cands[int(j)]
            both = float((masks[best] & masks[j]).sum()) / max(1, min(masks[best].sum(), masks[j].sum()))
            if (share[j] >= max(0.12, 0.6 * share[best]) and float(V[j] @ V[best]) < c.get("name_combine_sim", 0.4)
                    and both < c.get("name_combine_overlap", 0.3)         # said about different clauses, not two words for one thing
                    and not (set(t.split()) & set(cands[best].split())) and score[j] > score[best] - 0.25):
                name = f"{name} & {say(int(j))}"
                break
    return name, mixed, [say(int(i)) for i in order if not bad[i] and int(i) != best][:c.get("name_alternatives", 8)], aliases


def _aspect_text(doc, noise=None):
    """What a clause is ABOUT: its nouns and plain attributes, without judging or filler words ("Really citrusy flavour" -> "citrusy flavour")."""
    return " ".join(t.lower_ for t in doc if t.pos_ in ("NOUN", "PROPN", "ADJ") and t.is_alpha and not t.is_stop and t.lower_ not in (_NOISE if noise is None else noise))


def _with_aspects(clauses, Ecl, mu, axis, c, cfg, noise=None):
    """Clustering space = the clause meaning plus themes.aspect_weight x the meaning of its topic words. Whole-clause embeddings group
    answers that are WORDED alike ("the taste is X", "Really X flavour") as much as answers about the same thing; adding the topic
    words pulls the clusters towards topics. Only the clustering uses this: centres and assignment stay in the clause space."""
    w = c.get("aspect_weight", 0.5)
    if not w or cfg is None:
        return Ecl
    from . import embed
    nlp = spacy.load(c["spacy_model"], disable=["ner", "lemmatizer", "parser"])
    texts = [_aspect_text(d, noise) or t.lower() for d, t in zip(nlp.pipe(clauses["clause"].tolist(), batch_size=256), clauses["clause"])]
    A = embed.encode_cached([sha("aspect:" + t) for t in texts], texts, cfg)
    X = Ecl + w * center(A, mu, 0.0, axis)
    return (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def _clause_terms(doc, noise=None):
    """The topic words of a clause: nouns and plain attributes, judging / filler words (`noise`) left out, plurals folded ("pieces" = "piece")."""
    noise = _NOISE if noise is None else noise
    out = []
    for t in doc:
        if t.pos_ in ("NOUN", "PROPN", "ADJ") and t.is_alpha and not t.is_stop and len(t.lower_) >= 3 and t.lower_ not in noise:
            w = t.lower_
            out.append(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w)
    return out


def _aspect_candidates(clauses, Ecl, ok, c, k, noise=None):
    """Themes found from WORDS, not from dense regions of clauses. The topic words of the answers are grouped by what the clauses
    that use them are about, then every clause goes to the group most of its topic words belong to. Deterministic (no random
    projection), and a theme is a set of words that belong together ("packaging, label, bottle, colours", "price, value, worth")
    instead of answers that happen to be worded alike. A group holding more than themes.split_large_frac of the answers is split."""
    nlp = spacy.load(c["spacy_model"], disable=["ner", "lemmatizer", "parser"])
    T = [_clause_terms(d, noise) for d in nlp.pipe(clauses["clause"].iloc[ok].tolist(), batch_size=256)]
    df = {}
    for ts in T:
        for w in set(ts):
            df[w] = df.get(w, 0) + 1
    vocab = [w for w, n in df.items() if n >= max(c.get("aspect_min_df", 4), 0.01 * len(ok))]
    if len(vocab) < 3:
        return []
    idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(vocab), Ecl.shape[1]), np.float32)
    for r, ts in zip(ok, T):                                        # a word = the mean of the clauses that use it
        for w in set(ts):
            if w in idx:
                X[idx[w]] += Ecl[r]
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    lab = AgglomerativeClustering(n_clusters=min(k, len(vocab) - 1), metric="cosine", linkage="average").fit_predict(X)

    def votes(ts):                                                  # rarer words count more
        v = {}
        for w in ts:
            if w in idx:
                v[lab[idx[w]]] = v.get(lab[idx[w]], 0) + 1.0 / np.log(2 + df[w])
        return v
    frac = c.get("split_large_frac") or 1.0
    for _ in range(3):
        size = {}
        for ts in T:
            v = votes(ts)
            if v:
                b = max(v, key=v.get)
                size[b] = size.get(b, 0) + 1
        big = [l for l, n in size.items() if n > frac * len(ok)]
        if not big:
            break
        for l in big:                                               # an oversized group is divided with ward linkage, which keeps sizes even
            ids = np.where(lab == l)[0]
            if len(ids) < 6:
                continue
            parts = max(2, min(4, int(round(size[l] / (frac * len(ok) * 0.6)))))
            sub = AgglomerativeClustering(n_clusters=min(parts, len(ids)), linkage="ward").fit_predict(X[ids])
            nxt = lab.max() + 1
            for si in range(1, sub.max() + 1):
                lab[ids[sub == si]] = nxt + si - 1
    members = {}
    for r, ts in zip(ok, T):
        v = votes(ts)
        if v:
            members.setdefault(max(v, key=v.get), []).append(r)
    return [np.array(m) for m in members.values() if len(m) >= c["min_theme_clauses"]]


def _name_passes(index, name, members, c, aliases=()):
    """Does this name fit the theme? name_min_coverage is the floor; with themes.name_min_lift the name must also be clearly more typical
    of the theme than of the rest of the survey (quality.name_ok)."""
    inside, outside = quality.name_rates(index, name, members, aliases)
    return quality.name_ok(inside, outside, c.get("name_min_coverage", 0.0), c.get("name_min_lift"))


def _naming_context(clauses, c, noise=None, generic=None):
    """What naming needs about the whole survey, built once: the tagger, the learned part-of-speech table and, when names are
    checked against their members (themes.name_min_coverage), the lemmatized clauses."""
    check = c.get("name_min_coverage", 0.0) > 0
    nlp = spacy.load(c["spacy_model"], disable=["ner"] + ([] if check else ["lemmatizer"]))
    index = quality.ClauseIndex(clauses["clause"], nlp) if check else None
    all_texts = clauses["clause"].str.lower()
    return {"nlp": nlp, "pos": _pos_table(clauses, nlp), "all_texts": all_texts, "index": index, "noise": noise, "generic": generic,
            "usage": index.usage if index is not None else quality.WordUsage(all_texts)}


def _clause_phrases(doc):
    """Phrases a clause offers as keywords: its noun chunks (articles and other function words trimmed off the ends) and every run of
    two or three consecutive content words. Lower case, words in the order they are said."""
    out = set()
    for ch in doc.noun_chunks:
        toks = [t for t in ch]
        while toks and (toks[0].is_stop or not toks[0].is_alpha):
            toks = toks[1:]
        while toks and (toks[-1].is_stop or not toks[-1].is_alpha):
            toks = toks[:-1]
        if 1 <= len(toks) <= 3:
            out.add(" ".join(t.lower_ for t in toks))
    run = []
    for t in list(doc) + [None]:
        if t is not None and t.is_alpha and not t.is_stop:
            run.append(t.lower_)
            continue
        for n in (2, 3):
            out.update(" ".join(run[i:i + n]) for i in range(len(run) - n + 1))
        run = []
    return out


def _phrase_norm(p):
    """A phrase with plural endings taken off, for telling whether two phrases are the same thing ("white pieces" = "white piece")."""
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in p.split())


def _drop_variants(terms):
    """Indices to keep, in the given (best first) order: a phrase is dropped when its plural-folded text is a substring of a better
    phrase's ("sweet" after "too sweet", "piece" after "white pieces"), or contains one ("sweetness" after "sweet"). Better phrases
    count even when they were dropped themselves, so "sweetness" does not survive just because "sweet" did not."""
    kept, norms = [], []
    for i, t in enumerate(terms):
        n = _phrase_norm(t)
        if not any(n in k or k in n for k in norms):
            kept.append(i)
        norms.append(n)
    return kept


def _mmr(rel, V, k, lam):
    """Maximal marginal relevance: k indices, each the best trade-off (lam) between being relevant (rel, scaled 0..1) and being
    unlike what is already chosen (cosine of the unit rows of V). Ties go to the lower index, so the result is deterministic."""
    rel = np.asarray(rel, float)
    r = (rel - rel.min()) / (rel.max() - rel.min() + 1e-12) if len(rel) else rel
    chosen = []
    while len(chosen) < min(k, len(rel)):
        best, best_v = None, -np.inf
        for i in range(len(rel)):
            if i in chosen:
                continue
            v = lam * r[i] - (1 - lam) * (max(float(V[i] @ V[j]) for j in chosen) if chosen else 0.0)
            if v > best_v + 1e-12:
                best, best_v = i, v
        chosen.append(best)
    return chosen


def _theme_phrases(m, texts, ctx, c, Ec, mu, cfg, centre):
    """The keywords of one theme: candidate phrases (noun chunks and 2-3-word runs from its clauses, minus judging / filler words and
    verbs, as everywhere else) ranked by closeness in meaning to the theme's centre plus how much more the theme uses them than the
    survey (TF-IDF lift), repeats dropped (`_drop_variants`), then the themes.n_phrases chosen with MMR so the list is varied."""
    counts = {}
    for i in m:
        for ph in ctx["phrases"][i]:
            counts[ph] = counts.get(ph, 0) + 1
    ranked = [p for p, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if n >= c.get("phrase_min_count", 2)][:c.get("phrase_pool", 60)]
    clean = _clean_terms(ranked, texts, ctx["all_texts"], ctx["pos"], c, limit=len(ranked), nlp=ctx["nlp"], dedupe=False, noise=ctx.get("noise"))
    if not clean:
        return []
    terms, share, lift = [t for t, _, _ in clean], np.array([s_ for _, s_, _ in clean]), np.array([l for _, _, l in clean])
    k = c["n_phrases"]
    if Ec is None or cfg is None:                                   # no meaning vectors: most shared first
        return [terms[i] for i in _drop_variants([terms[i] for i in np.argsort(-share, kind="stable")])[:k]]
    from . import embed
    V0 = embed.encode_cached([sha("name:" + t) for t in terms], terms, cfg)
    V = center(V0, mu, c.get("center_strength", 1.0), axis_of(cfg)) if c.get("center") else V0
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    rel = V @ centre + c.get("phrase_lift_weight", 0.3) * np.clip(np.log(np.maximum(lift, 1e-6)), 0, 3) / 3
    order = np.argsort(-rel, kind="stable")
    keep = [int(order[i]) for i in _drop_variants([terms[i] for i in order])]
    pick = _mmr(rel[keep], V[keep], k, c.get("phrase_mmr_lambda", 0.6))
    return [terms[keep[i]] for i in pick]


def _representatives(texts, sims, own, n):
    """The n clauses closest to a theme's centre, each text once. From the clauses the theme owns (it is their closest theme); a theme
    that owns none (one you added) gives its closest clauses overall."""
    idx = np.where(own)[0] if own.any() else np.arange(len(sims))
    out, seen = [], set()
    for i in idx[np.argsort(-sims[idx], kind="stable")]:
        t = str(texts.iloc[i])
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
            if len(out) == n:
                break
    return out


def _name_themes(clauses, merged, c, Ec=None, mu=None, cfg=None, ctx=None):
    ctx = ctx or _naming_context(clauses, c)
    nlp, pos, all_texts, index, usage = ctx["nlp"], ctx["pos"], ctx["all_texts"], ctx["index"], ctx["usage"]
    noise, generic = ctx.get("noise"), ctx.get("generic")
    docs = [" ".join(clauses["clause"].iloc[m]) for m in merged]
    cv = CountVectorizer(ngram_range=(1, 2), stop_words="english",
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-']+\b")
    tf = cv.fit_transform(docs).toarray().astype(float)
    terms = np.array(cv.get_feature_names_out())
    w = tf / np.maximum(tf.sum(1, keepdims=True), 1)
    A = tf.sum() / len(docs)
    score = w * np.log(1 + A / np.maximum(tf.sum(0), 1))          # class-based TF-IDF
    out, used, used_vecs = [], set(), {}
    min_cov = c.get("name_min_coverage", 0.0)
    for i, m in enumerate(merged):
        order = np.argsort(-score[i])
        texts = clauses["clause"].iloc[m].str.lower()
        clean = _clean_terms(terms[order[:120]], texts, all_texts, pos, c, limit=c["n_keywords"], nlp=nlp, noise=noise)
        kws = [t for t, _, _ in clean]
        if c.get("n_phrases"):
            if "phrases" not in ctx:                                # parsed once for the whole survey, then shared by every naming round
                ctx["phrases"] = [_clause_phrases(d) for d in nlp.pipe(clauses["clause"].tolist(), batch_size=256)]
            centre = _unit(Ec[m].mean(0)) if Ec is not None else None
            kws = _theme_phrases(m, texts, ctx, c, Ec, mu, cfg, centre) or kws
        pool_terms = _clean_terms(terms[order[:120]], texts, all_texts, pos, c, limit=40, nlp=nlp, dedupe=False, noise=noise)   # naming sees phrases AND their words
        name, mixed, alts, aliases = _pick_name([t for t, _, _ in pool_terms], texts, all_texts, used, used_vecs, nlp, c, Ec, mu, cfg, m, index, pos, usage, generic)
        if not name:                                                # nothing informative: the theme's strongest raw term no earlier theme took
            free = [x.title() for x in terms[order[:40]].tolist() if x.title() not in used]
            name = free[0] if free else terms[order[0]].title()
            mixed, alts, aliases = bool(min_cov) and not _name_passes(index, name, m, c), [], []
        base, k = name, 2
        while name in used:                                         # two themes with one name would share an output column
            name, k = f"{base} {k}", k + 1
        used.add(name)
        # Keywords used to MATCH clauses: a term must be common inside the theme and clearly more common there than in
        # the whole survey ("taste" in a taste theme passes; "good", which every theme uses, does not).
        match_kws = []
        for t in terms[order[:c["n_keywords"] * 2]]:
            if len(t) < c.get("keyword_min_len_match", 3) or all(nlp.vocab[w].is_stop for w in t.split()):
                continue                                            # too short, or only function words ("it's", "quite")
            share_in = texts.str.contains(r"\b" + re.escape(t) + r"\b", regex=True).mean()
            share_all = all_texts.str.contains(r"\b" + re.escape(t) + r"\b", regex=True).mean()
            if share_in >= c.get("keyword_min_share", 0.10) and share_in >= c.get("keyword_min_lift", 2.0) * share_all:
                match_kws.append(t)
        out.append({"theme_name": name, "match_name": name, "keywords": kws or terms[order[:c["n_keywords"]]].tolist(), "assign_keywords": match_kws[:c["n_keywords"]],
                    "frequency": int(len(m)), "n_informative": sum(1 for _, sh, _ in pool_terms if sh >= c.get("generic_term_share", 0.10))})
        if index is not None:
            out[-1]["name_coverage"] = round(quality.name_coverage(index, name, m, aliases), 3)
        out[-1]["name_alternatives"] = alts                        # the next-ranked names, for the check against the theme's shown clauses
        if aliases:
            out[-1]["name_aliases"] = aliases                       # the descriptor an aspect label replaced: the label stands for it
        if mixed:
            out[-1]["name_mixed"] = True                            # no candidate name covers enough of the theme's clauses
    return out


def _name_and_split(clauses, merged, c, Ec, mu, cfg, n_target=None, ctx=None):
    """Names the themes and checks each name against its members. A theme no name fits (mixed) is split in two with ward linkage on its
    clause vectors (no random step); the split is kept only if both halves are big enough (themes.min_theme_clauses) and
    both get a name that passes, and no other theme becomes mixed because of it. Splitting adds themes, so the n_target largest are kept."""
    ctx = ctx or _naming_context(clauses, c)
    themes = _name_themes(clauses, merged, c, Ec, mu, cfg, ctx)
    if Ec is None or ctx["index"] is None:
        return themes, merged
    n_mixed = lambda ts: sum(bool(t.get("name_mixed")) for t in ts)
    min_n, tried, split = c.get("min_theme_clauses", 8), set(), False
    while True:
        todo = [i for i, t in enumerate(themes) if t.get("name_mixed") and merged[i].tobytes() not in tried and len(merged[i]) >= 2 * min_n]
        if not todo:
            break
        i = max(todo, key=lambda i: len(merged[i]))
        tried.add(merged[i].tobytes())
        lab = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(Ec[merged[i]])
        parts = sorted((merged[i][lab == k] for k in (0, 1)), key=lambda p: (-len(p), int(p.min())))
        if min(len(p) for p in parts) < min_n:
            continue
        trial_m = merged[:i] + parts + merged[i + 1:]
        trial_t = _name_themes(clauses, trial_m, c, Ec, mu, cfg, ctx)
        if not any(trial_t[j].get("name_mixed") for j in (i, i + 1)) and n_mixed(trial_t) < n_mixed(themes):
            detail(f"[themes] {themes[i]['theme_name']!r} fits none of its {len(merged[i])} clauses well: split into "
                  f"{trial_t[i]['theme_name']!r} ({len(parts[0])}) and {trial_t[i + 1]['theme_name']!r} ({len(parts[1])})")
            themes, merged, split = trial_t, trial_m, True
    if split and n_target and len(merged) > n_target:
        merged = _reduce(merged, Ec, n_target)
        themes = _name_themes(clauses, merged, c, Ec, mu, cfg, ctx)
    return themes, merged


def _shown_members(merged, Ec, gv, floor, iters, shown_floor=None, margin=0.1):
    """Which clauses each theme will show: the ones nearest its centre (above the entry bar) once the centres are pulled towards them,
    as the refinement at the end of discover() does. Wider than the clusters, which are only the dense cores of the topics.
    The General comment centre `gv`, when there is one, competes for clauses. A clause also counts as shown when it is only as close as
    `shown_floor` (assign.fallback_min_sim): the assignment gives such leftover clauses to their closest theme."""
    C = np.stack([_unit(Ec[m].mean(0)) for m in merged]).astype(np.float32)
    if gv is not None:
        C = np.vstack([C, gv[None].astype(np.float32)])
    for _ in range(iters):
        S = Ec @ C.T
        top, best = S.argmax(1), S.max(1)
        for j in range(len(C)):
            near = (top == j) & (best >= floor)
            if near.sum() >= 5:
                C[j] = _unit(C[j] + _unit(Ec[near].mean(0)))
    S = Ec @ C.T
    top, best = S.argmax(1), S.max(1)
    low = min(floor, shown_floor) if shown_floor is not None else floor
    shown = (top[:, None] == np.arange(len(C))[None]) & (best >= low)[:, None]
    if len(C) > 1:                                              # the second theme of a clause that is nearly as close to it (assign.secondary_margin)
        second = np.argsort(-S, axis=1)[:, 1]
        s2 = S[np.arange(len(S)), second]
        sec = (s2 >= floor) & (best - s2 <= margin)
        shown[np.arange(len(S))[sec], second[sec]] = True
    return [np.where(shown[:, j])[0] for j in range(len(merged))]


def _recheck_names(themes, shown, ctx, c):
    """A name that fits a theme's dense core can still fit little of what the theme shows. Check every found theme's name against the
    clauses it will show: below themes.name_min_coverage the next-ranked names are tried (not one another theme has), and when none
    passes the theme is marked mixed. A theme that passes is no longer marked."""
    index = ctx["index"]
    taken = {t["theme_name"] for t in themes}
    for t, m in zip(themes, shown):
        if t.get("required") or t.get("seeded") or t.get("general") or not len(m):
            continue
        aliases = t.get("name_aliases") or ()
        ok = _name_passes(index, t["theme_name"], m, c, aliases)
        if not ok:
            for alt in t.get("name_alternatives", []):
                if alt not in taken and _name_passes(index, alt, m, c):
                    detail(f"[themes] {t['theme_name']!r} does not fit the clauses it shows: renamed {alt!r}")
                    taken.discard(t["theme_name"])
                    taken.add(alt)
                    t["theme_name"], ok, aliases = alt, True, ()
                    t.pop("name_aliases", None)
                    break
        t["name_shown_coverage"] = round(quality.name_rates(index, t["theme_name"], m, aliases)[0], 3)
        if ok:
            t.pop("name_mixed", None)
        else:
            t["name_mixed"] = True


def _is_generic(t, m, Ec, gv, c):
    """Returns (generic?, closeness to General): no informative word of its own, or clauses close to the General comment seeds."""
    sim = float(np.median(Ec[m] @ gv))
    return t.get("n_informative", 99) < c.get("generic_min_terms", 1) or sim >= c.get("generic_theme_sim", 0.4), sim


def _protected_names(c):
    """Lower-case names of themes a person has a say in: the required ones (General comment included) and every theme the curation
    refers to. Discovered themes with these names are never merged."""
    names = set()
    for r in c.get("required") or []:
        names.add(str(r["name"] if isinstance(r, dict) else r).strip().lower())
    cur = c.get("curation") or {}
    for new, group in (cur.get("merge") or {}).items():
        names |= {str(n).strip().lower() for n in [new, *group]}
    for key in ("rename", "extra_keywords", "seeds", "add"):
        names |= {str(n).strip().lower() for n in (cur.get(key) or {})}
    names |= {str(n).strip().lower() for n in cur.get("drop") or []}
    for op in cur.get("ops") or []:
        names |= {str(n).strip().lower() for n in (op[1:] if op[0] != "merge" else [op[1], *op[2]]) if n}
    return names


def _duplicate_pairs(C, keywords, name_vecs, eligible, c):
    """Pairs of themes that look like one topic: [(strength, i, j, why)], strongest first. A pair qualifies when
      - the centres are at least themes.merge_sim close (cosine), or
      - their main words overlap by at least themes.merge_jaccard (Jaccard), or
      - their names mean the same: cosine of the (centred) name vectors at least themes.merge_name_sim.
    A signal that is not set is off. `eligible` is False for themes that must not be merged (yours, General comment, generic)."""
    if c.get("merge_k") is not None:
        return _outlier_pairs(C, keywords, name_vecs, eligible, c["merge_k"])
    sim, jac, nsim = c.get("merge_sim"), c.get("merge_jaccard"), c.get("merge_name_sim")
    out = []
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            if not (eligible[i] and eligible[j]):
                continue
            hits = {}
            if sim is not None and float(C[i] @ C[j]) >= sim:
                hits["centres"] = float(C[i] @ C[j]) / sim
            if jac is not None:
                v = quality.keyword_jaccard(keywords[i], keywords[j])
                if v >= jac:
                    hits["main words"] = v / jac
            if nsim is not None and name_vecs is not None and float(name_vecs[i] @ name_vecs[j]) >= nsim:
                hits["names"] = float(name_vecs[i] @ name_vecs[j]) / nsim
            if hits:
                out.append((max(hits.values()), i, j, "+".join(sorted(hits))))
    return sorted(out, key=lambda p: (-p[0], p[1], p[2]))


def _outlier_pairs(C, keywords, name_vecs, eligible, k):
    """Relative version of `_duplicate_pairs`, with no fixed similarity: a pair qualifies when its centre cosine is an OUTLIER among the cosines
    of all pairs of eligible themes of this survey (more than k standard deviations above their mean, themes.merge_k) AND another signal
    agrees: the main words overlap (a Jaccard that is itself an outlier and above zero) or the names mean the same (name similarity an
    outlier). Needs at least three pairs to have a distribution. Strongest (largest z) first."""
    idx = [i for i in range(len(C)) if eligible[i]]
    pairs = [(i, j) for a, i in enumerate(idx) for j in idx[a + 1:]]
    if len(pairs) < 3:
        return []

    def z(values):
        v = np.asarray(values, float)
        return (v - v.mean()) / v.std() if v.std() > 1e-9 else np.zeros(len(v))
    zc = z([float(C[i] @ C[j]) for i, j in pairs])
    jac = np.array([quality.keyword_jaccard(keywords[i], keywords[j]) for i, j in pairs])
    zj = z(jac)
    zn = z([float(name_vecs[i] @ name_vecs[j]) for i, j in pairs]) if name_vecs is not None else np.zeros(len(pairs))
    out = []
    for n, (i, j) in enumerate(pairs):
        if zc[n] < k:
            continue
        why = [w for w, ok in (("main words", jac[n] > 0 and zj[n] >= k), ("names", zn[n] >= k)) if ok]
        if why:
            out.append((float(zc[n]), i, j, "centres+" + "+".join(why)))
    return sorted(out, key=lambda p: (-p[0], p[1], p[2]))


def _split_oversized(merged, Ec, c):
    """A theme much bigger than the typical theme of this survey (themes.split_ratio x the median size) is split in two with ward linkage
    on its clause vectors, when it falls apart more clearly than this survey's themes usually do. Every theme is split in two on the side
    to measure how clearly (silhouette, cosine); a smooth blob splits with a silhouette like any other theme, two topics glued together
    stand out: the big theme's silhouette must be more than themes.split_k standard deviations above the mean of the survey's themes
    and at least themes.split_min_gain above their median. Both halves must have themes.min_theme_clauses. No random step."""
    ratio = c.get("split_ratio")
    if not ratio or len(merged) < 3:
        return merged
    min_n = c.get("min_theme_clauses", 8)

    def halves(m):
        lab = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(Ec[m])
        return lab, (float(silhouette_score(Ec[m], lab, metric="cosine")) if len(set(lab)) == 2 else -1.0)
    cache = {id(m): halves(m) for m in merged if len(m) >= 6}
    sil = np.array([v[1] for v in cache.values()])
    if len(sil) < 3 or sil.std() < 1e-9:
        return merged
    median_size = float(np.median([len(m) for m in merged]))
    out = []
    for m in merged:
        lab, s_m = cache.get(id(m), (None, None))
        sizes = np.bincount(lab, minlength=2) if lab is not None else (0, 0)
        if (lab is None or len(m) < ratio * median_size or min(sizes) < min_n
                or (s_m - sil.mean()) / sil.std() < c.get("split_k", 1.5) or s_m - np.median(sil) < c.get("split_min_gain", 0.1)):
            out.append(m)
            continue
        detail(f"[themes] a theme of {len(m)} clauses (median {median_size:.0f}) falls apart more clearly than the others: split into {sizes[0]} and {sizes[1]}")
        out.extend(m[lab == k] for k in (0, 1))
    return out


def _merge_duplicates(clauses, merged, c, Ec, mu, cfg, ctx, gv):
    """Merges candidate themes that are one topic split in pieces ("Packaging", "Label", "Wrapper"), BEFORE the n_themes largest are kept,
    so the pieces add up to one theme instead of dropping out one by one. Candidates are named first (their main words and names are
    the signals); each theme is merged with at most one partner per round, strongest pair first, and the merged themes are named
    again in the next round (at most three rounds), so a chain of look-alikes never collapses into one blob by transitivity.
    Never merged: themes with the name of a required theme or one the curation mentions, and generic ones (they go to General comment)."""
    if all(c.get(k) is None for k in ("merge_sim", "merge_jaccard", "merge_name_sim", "merge_k")):
        return merged
    protected = _protected_names(c)
    for _ in range(3):
        themes = _name_themes(clauses, merged, c, Ec, mu, cfg, ctx)
        eligible = [t["theme_name"].lower() not in protected and not (gv is not None and _is_generic(t, m, Ec, gv, c)[0])
                    for t, m in zip(themes, merged)]
        C = np.stack([_unit(Ec[m].mean(0)) for m in merged])
        name_vecs = None
        if (c.get("merge_name_sim") is not None or c.get("merge_k") is not None) and cfg is not None:
            from . import embed
            names = [t["theme_name"].lower() for t in themes]
            V = embed.encode_cached([sha("name:" + n) for n in names], names, cfg)
            V = center(V, mu, c.get("center_strength", 1.0), axis_of(cfg)) if c.get("center") else V
            name_vecs = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        taken, groups = set(), []
        for _, i, j, why in _duplicate_pairs(C, [t["keywords"] for t in themes], name_vecs, eligible, c):
            if i in taken or j in taken:
                continue
            taken |= {i, j}
            groups.append((i, j))
            detail(f"[themes] {themes[i]['theme_name']!r} ({len(merged[i])}) and {themes[j]['theme_name']!r} ({len(merged[j])}) look like one topic ({why}): merged")
        if not groups:
            break
        gone = {j for _, j in groups}
        pooled = {i: np.concatenate([merged[i], merged[j]]) for i, j in groups}
        merged = [pooled.get(k, m) for k, m in enumerate(merged) if k not in gone]
    return merged


def _fold_generic(themes, merged, Ec, gv, c):
    """A cluster of generic remarks ("easy drink, overall, balanced") is not a topic: it has no informative word of its own, or its
    clauses sit close to the General comment seeds. It is dropped, and its clauses are assigned afterwards, where the ones that
    name nothing specific end up in General comment."""
    keep = []
    for i, (t, m) in enumerate(zip(themes, merged)):
        generic, sim = _is_generic(t, m, Ec, gv, c)
        t["general_sim"] = round(sim, 3)
        if generic:
            detail(f"[themes] {t['theme_name']!r} is generic (informative words {t.get('n_informative')}, closeness to General {sim:.2f}): folded into General comment")
            continue
        keep.append(i)
    if len(keep) < max(2, c.get("min_themes", 2)):
        return themes, merged
    return [themes[i] for i in keep], [merged[i] for i in keep]


def _candidates(E, ok, c, seed):
    """Every theme the data supports: cluster, merge near-duplicates, drop tiny groups, split oversized ones."""
    lab, sub = _cluster(E[ok], c, seed)
    members = {l: ok[sub[lab == l]] for l in sorted(set(lab) - {-1})}
    ls = list(members)
    cents = np.stack([_unit(E[members[l]].mean(0)) for l in ls])
    # merge near-duplicate clusters (connected components over centroid cosine)
    _, comp = connected_components(csr_matrix(cents @ cents.T >= c["merge_cosine"]), directed=False)
    groups = {}
    for l, g in zip(ls, comp):
        groups.setdefault(g, []).extend(members[l].tolist())
    merged = [np.array(v) for v in groups.values() if len(v) >= c["min_theme_clauses"]]
    return _split_large(merged, E, len(ok), c, seed)


def _reduce(merged, Ec, n):
    """Keep the n largest themes; every smaller one is folded into its most similar kept theme, so its clauses are
    spread over the kept themes instead of being thrown away."""
    merged = sorted(merged, key=len, reverse=True)
    if not n or len(merged) <= n:
        return merged
    keep = [list(m) for m in merged[:n]]
    cents = np.stack([_unit(Ec[m].mean(0)) for m in merged[:n]])
    for m in merged[n:]:
        j = int(np.argmax(cents @ _unit(Ec[m].mean(0))))
        keep[j].extend(m.tolist())
    return sorted((np.array(k) for k in keep), key=len, reverse=True)


def _split_large(merged, E, n_usable, c, seed):
    """A theme holding more than themes.split_large_frac of all clauses (one "Taste" blob) is clustered again on its
    own, with a smaller minimum size, and replaced by its sub-themes."""
    frac = c.get("split_large_frac")
    if not frac:
        return merged
    out = []
    for m in merged:
        if len(m) <= frac * n_usable or len(m) < 4 * c["min_theme_clauses"]:
            out.append(m)
            continue
        sub = dict(c, min_cluster_size=max(4, c["min_cluster_size"] // 2), min_samples=max(2, c["min_samples"] - 1),
                   min_cluster_frac=0, min_themes=2)
        lab, idx = _cluster(E[m], sub, seed)
        parts = [m[idx[lab == l]] for l in sorted(set(lab) - {-1})]
        parts = [p for p in parts if len(p) >= c["min_theme_clauses"]]
        out.extend(parts if len(parts) >= 2 else [m])
    return out


def _curate(themes, merged, c):
    """Optional human edits from config (themes.curation): merge / rename / extra_keywords / drop, by theme name."""
    cur = c.get("curation") or {}

    def idx(n):                                              # case-insensitive, ignores themes already merged away or dropped
        for i, t in enumerate(themes):
            if not t.get("_dead") and t["theme_name"].lower() == str(n).strip().lower():
                return i
        alive = [t["theme_name"] for t in themes if not t.get("_dead")]
        raise ValueError(f"curation refers to unknown theme {n!r}. Themes: {alive}")

    def check_free(name, allowed):
        """A theme name must stay unique: two themes with one name would share an output column."""
        for i, t in enumerate(themes):
            if not t.get("_dead") and i not in allowed and t["theme_name"].lower() == str(name).strip().lower():
                raise ValueError(f"A theme called {t['theme_name']!r} already exists. Choose another name.")

    def merge(new, group):
        ids = list(dict.fromkeys(idx(n) for n in group))      # the same theme named twice counts once
        if len(ids) < 2:
            raise ValueError("Pick at least two different themes to merge.")
        if new:
            check_free(new, ids)
        keep = ids[0]
        themes[keep]["_parts"] = [p for i in ids for p in (themes[i].get("_parts") or [merged[i]])]     # assignment keeps everything each part covered
        merged[keep] = np.concatenate([merged[i] for i in ids])
        kws = list(dict.fromkeys(k for i in ids for k in themes[i]["keywords"]))
        match = list(dict.fromkeys(k for i in ids for k in themes[i].get("assign_keywords", [])))
        themes[keep].update(keywords=kws[:c["n_keywords"]], assign_keywords=match, frequency=int(len(merged[keep])))
        for i in ids[1:]:
            themes[i]["_dead"] = True
        themes[keep]["theme_name"] = new or themes[keep]["theme_name"]
        themes[keep].pop("name_mixed", None)
        themes[keep].pop("name_aliases", None)                  # merged themes are named by a person or by the first one's name: not "mixed"

    def soft(what, fn, *args):
        """Edits written in the config may name themes this data did not produce: warn and skip instead of failing."""
        try:
            fn(*args)
        except ValueError as e:
            print(f"WARNING: skipped curation ({what}): {e}")

    def set_name(old, new):
        i = idx(old)
        check_free(new, [i])
        themes[i]["theme_name"] = new
        themes[i].pop("name_mixed", None)
        themes[i].pop("name_aliases", None)                    # a name a person chose is not checked

    def set_keywords(n, kws):
        themes[idx(n)]["curated_keywords"] = list(kws)         # bypass the generic-keyword filter in assign

    def kill(n):
        themes[idx(n)]["_dead"] = True

    for new, group in (cur.get("merge") or {}).items():
        soft(f"merge into {new}", merge, new, group)
    for old, new in (cur.get("rename") or {}).items():
        soft(f"rename {old}", set_name, old, new)
    for n, kws in (cur.get("extra_keywords") or {}).items():
        soft(f"keywords for {n}", set_keywords, n, kws)
    for n in cur.get("drop") or []:
        soft(f"drop {n}", kill, n)
    for op in cur.get("ops") or []:                          # edits made one after another at the prompt, applied in order
        kind = op[0]
        if kind == "merge":                                  # ["merge", new_name_or_null, [names...]]
            merge(op[1], op[2])
        elif kind == "rename":                               # ["rename", old, new]
            set_name(op[1], op[2])
        elif kind == "drop":                                 # ["drop", name]
            themes[idx(op[1])]["_dead"] = True
        elif kind in ("drop_if_present", "rename_if_present"):    # an edit on a theme you added by name: the answers may also have produced one of that name
            if any(not t.get("_dead") and t["theme_name"].lower() == str(op[1]).strip().lower() for t in themes):
                if kind == "rename_if_present":
                    set_name(op[1], op[2])
                else:
                    kill(op[1])
    keep = [i for i, t in enumerate(themes) if not t.get("_dead")]
    if len(keep) < c["min_themes"] and not (c.get("required") and (cur.get("drop") or cur.get("ops"))):   # dropping every discovered theme is fine when the user's own themes fill the output
        raise ValueError(f"Only {len(keep)} themes left after curation.")
    return [themes[i] for i in keep], [merged[i] for i in keep]


def _expand(vec, Ec, clauses, c, words=(), cfg=None, mu=None):
    """A theme the user only named: find the clauses that mean the same thing (or use its words), pull the centre towards
    them, repeat so paraphrases are found too, and learn the words they use that the rest of the survey does not.
    Returns (centre, keywords, members); members are indices into `clauses`."""
    if Ec is None or clauses is None or not len(Ec):
        return vec, [], np.array([], int)
    texts = clauses["clause"].str.lower()
    lexical = np.zeros(len(Ec), bool)
    for w in words:                                          # clauses that literally use the name / the words given
        if len(w) >= 3:
            lexical |= texts.str.contains(r"\b" + re.escape(w.lower()), regex=True).to_numpy()
    thr = c.get("required_member_sim", 0.40)
    centre = vec
    for _ in range(c.get("required_rounds", 3)):
        sims = Ec @ centre
        strict = (sims >= thr) | (lexical & (sims >= 0.15))     # a word match alone is not enough: it must also be on topic
        if strict.sum() < c.get("required_min_members", 3):
            return vec, [], np.array([], int)                    # nothing in the answers is about it: do not drift towards unrelated clauses
        centre = _unit(vec + _unit(Ec[strict].mean(0))).astype(np.float32)
    sims = Ec @ centre
    m = np.where(strict)[0]
    m = m[np.argsort(-sims[m])][:c.get("required_max_members", 150)]
    take = len(m)
    try:
        cv = CountVectorizer(binary=True, stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-']{2,}\b")
        X = cv.fit_transform(texts)
    except ValueError:
        return centre, [], m
    terms = np.array(cv.get_feature_names_out())
    n_in = np.asarray(X[m].sum(0)).ravel()
    n_all = np.asarray(X.sum(0)).ravel()
    share_in, share_all = n_in / len(m), n_all / X.shape[0]
    good = (n_in >= 2) & (share_in >= c.get("keyword_min_share", 0.10)) & (share_in >= c.get("required_min_lift", 4.0) * share_all) & (share_all <= c.get("required_max_share_all", 0.05))
    rank = np.argsort(-(share_in * share_in / np.maximum(share_all, 1e-9)) * good)
    n_kw = c.get("n_keywords", 8)
    kws = [terms[i] for i in rank[:3 * n_kw] if good[i]]
    if kws and cfg is not None and mu is not None:
        # A frequent word is not a good keyword just because it sits in the theme's clauses ("favourite", "good" in a Price theme).
        # Keep a word only if its own MEANING is close to the theme's centre and it is not a plain evaluation word.
        from . import embed
        V0 = embed.encode_cached([sha("name:" + k) for k in kws], kws, cfg)
        ax = axis_of(cfg)
        V = center(V0, mu, c.get("center_strength", 1.0), ax) if c.get("center") else V0
        ok = (V @ centre) >= c.get("required_kw_min_sim", 0.35)
        if ax is not None:
            ok &= np.abs(V0 @ ax) < c.get("name_eval_max", 0.35)
        kws = [k for k, o in zip(kws, ok) if o]
    return centre, kws[:n_kw], m


def _auto_seeds(name):
    """Polarity-neutral example phrases for a theme the user only named."""
    return [name, f"the {name}", f"I like the {name}", f"the {name} is not good", f"{name} could be better"]


def _apply_required(themes, C, mu, c, cfg, Ec=None, clauses=None):
    """Themes the user says MUST be in the results (themes.required: names or {name, keywords, seeds}).
    If a discovered theme is the same topic it is renamed and sharpened; otherwise a new theme is added.
    Everything else that was discovered stays, so the user's list is added to, not replaced."""
    from . import embed
    for r in c.get("required") or []:
        r = {"name": r} if isinstance(r, str) else dict(r)
        name = r["name"].strip()
        if not name or r.get("hidden"):                        # hidden = removed by the user, but it still shapes the search (see overrides_for in app.py)
            continue
        base = (r.get("defined_by") or name).strip()           # a renamed theme is still defined by the name it was added with
        sem_only = bool(r.get("semantic_only"))                # no keywords at all, given or learned: matched by meaning only
        kws = [] if sem_only else list(r.get("keywords") or re.findall(r"[A-Za-z']{3,}", base.lower()))
        seeds = list(r.get("seeds") or _auto_seeds(base))
        V = embed.encode_cached([sha("seed:" + x.lower()) for x in seeds], seeds, cfg)
        V = center(V, mu, c.get("center_strength", 1.0), axis_of(cfg)) if c.get("center") else V
        vec = _unit(V.mean(0)).astype(np.float32)
        vec, learned, members = _expand(vec, Ec, clauses, c, words=kws, cfg=cfg, mu=mu)
        n_mem = len(members)
        # words learned from the data are frequent words that happen to sit in the theme's clauses ("bitter", "weak" for a
        # theme about aftertaste); themes.learn_keywords: false keeps only the words the user wrote
        kws = [] if sem_only else list(dict.fromkeys(kws + (learned if c.get("learn_keywords", True) else [])))
        flags = {"general": True} if r.get("general") else {}   # the theme for generic remarks ("it is good"): see assign.run
        have = [i for i, t in enumerate(themes) if t["theme_name"].lower() == name.lower()]
        sims = C @ vec
        if have:                                              # already there (discovered or curated): just sharpen
            j = have[0]
        elif len(sims) and float(sims.max()) >= c.get("required_merge_cos", 0.5):
            j = int(sims.argmax())                             # same topic as a discovered theme: take it over
            themes[j]["theme_name"] = name
            themes[j].pop("name_mixed", None)
            themes[j].pop("name_aliases", None)
        else:
            themes.append({"theme_name": name, "keywords": kws, "curated_keywords": kws, "frequency": n_mem,
                           "seeded": True, "required": True, **flags})
            C = np.vstack([C, vec[None]])
            continue
        C[j] = _unit(C[j] + vec)
        themes[j]["keywords"] = list(dict.fromkeys(kws + themes[j]["keywords"]))[:c.get("n_keywords", 8)]
        themes[j].update(curated_keywords=list(dict.fromkeys(themes[j].get("curated_keywords", []) + kws)), required=True, **flags)
    return themes, C


def _hand_made_members(clauses, Ec, mu, c, cfg):
    """Clauses that belong to themes the user wrote (themes.required / curation.add). Found before clustering, so the topics
    that are LEFT are clustered again without them and do not get one big blob that swallows the hand-made theme."""
    from . import embed
    specs = []
    for r in c.get("required") or []:
        r = {"name": r} if isinstance(r, str) else dict(r)
        if r.get("name", "").strip():
            name = (r.get("defined_by") or r["name"]).strip()
            specs.append((list(r.get("seeds") or _auto_seeds(name)),
                          [] if r.get("semantic_only") else list(r.get("keywords") or re.findall(r"[A-Za-z']{3,}", name.lower()))))
    for name, spec in ((c.get("curation") or {}).get("add") or {}).items():
        specs.append((spec["seeds"], list(spec.get("keywords", [])) or re.findall(r"[A-Za-z']{3,}", name.lower())))
    found = [np.array([], int)]
    for seeds, words in specs:
        V = embed.encode_cached([sha("seed:" + x.lower()) for x in seeds], seeds, cfg)
        V = center(V, mu, c.get("center_strength", 1.0), axis_of(cfg)) if c.get("center") else V
        found.append(_expand(_unit(V.mean(0)).astype(np.float32), Ec, clauses, c, words=words, cfg=cfg, mu=mu)[2])
    return np.unique(np.concatenate(found))


def _trim_to_total(themes, C, Ec, n_total, floor):
    """themes.n_total = how many themes the output has IN TOTAL (the user's own included). Drop the smallest beyond that: themes found in
    the data first (the likelier duplicates of a theme the user wrote), then the user's own, the general-comment theme last of all.
    Sizes are counted inside each group, so adding a theme of yours never changes which found theme is trimmed, and your own themes
    are ranked by their own size. The clauses of a dropped theme go to the nearest remaining one when they are assigned."""
    if len(themes) <= n_total:
        return themes, C
    mine = lambda t: bool(t.get("required") or t.get("seeded"))
    size = np.zeros(len(themes), int)
    for group in ([j for j, t in enumerate(themes) if not mine(t)], [j for j, t in enumerate(themes) if mine(t)]):
        if group:
            S = Ec @ C[group].T
            size[group] = np.bincount(S.argmax(1)[S.max(1) >= floor], minlength=len(group))
    order = sorted(range(len(themes)), key=lambda j: (bool(themes[j].get("general")), mine(themes[j]), size[j]))
    gone = set(order[:len(themes) - n_total])
    keep = [j for j in range(len(themes)) if j not in gone]
    return [themes[j] for j in keep], C[keep]


# Settings that act only AFTER the clusters are found (naming, your own themes, trimming). Changing them never needs a new clustering.
_AFTER_CLUSTERING = {"required", "curation", "n_total", "learn_keywords", "required_kw_min_sim", "required_merge_cos",
                     "floor_quantile", "floor_max", "refine_iters", "n_keywords", "rediscover", "freeze",
                     "name_min_coverage", "name_min_lift", "merge_k", "split_ratio", "split_k", "split_min_gain", "n_phrases", "n_examples", "phrase_mmr_lambda", "phrase_pool", "phrase_min_count", "phrase_lift_weight", "aspect_labels", "aspect_label_min_sim", "merge_sim", "merge_jaccard", "merge_name_sim"}                       # naming and splitting a mixed theme work on the saved clusters


def _clustering_code_sig() -> str:
    """Fingerprint of the code that finds the clusters, so clusters saved by older code are never reused after it changed."""
    fns = (_find_candidates, _aspect_candidates, _candidates, _cluster, _split_large, _scaled, _with_aspects, _aspect_text, _clause_terms,
           _noise_for, _opinion_words)
    return sha("".join(inspect.getsource(f) for f in fns), 8)


def _discovery_fingerprint(clauses, general, cfg) -> str:
    """What the clustering depends on: the clauses, the embedding model, the seed, the discovery settings and the clustering code.
    Not the themes you add."""
    c = cfg["themes"]
    settings = {k: v for k, v in c.items() if k not in _AFTER_CLUSTERING}
    gen = list(general[0].get("seeds") or _auto_seeds(general[0]["name"])) if general and c.get("general_exclude_sim") is not None else []
    keys = clauses["clause_key"].tolist() if "clause_key" in clauses else clauses["clause"].tolist()
    blob = json.dumps({"keys": sha("|".join(map(str, keys)), 16), "model": str(cfg["embed"]["model"]), "seed": cfg["seed"],
                       "settings": settings, "general": gen, "method": c.get("method", "aspects"), "aspect_weight": c.get("aspect_weight", 0.5), "algo": 2, "code": _clustering_code_sig(), "lists": sha(",".join(sorted(_NOISE | _GENERIC_NAMES)), 8), "axis": cfg["themes"].get("sentiment_axis")}, sort_keys=True, default=str)
    return sha(blob, 16)


def _frozen_path(cfg):
    return Path(cfg["run_dir"]) / "clusters.json"


def _load_frozen(cfg, fp):
    """The clusters saved for exactly this data and these discovery settings, so adding a theme or re-running cannot reshuffle them.
    themes.freeze: false or themes.rediscover: true (--rediscover) ignores the saved clusters and finds them again."""
    c = cfg["themes"]
    if not c.get("freeze", True) or c.get("rediscover"):
        return None
    try:
        saved = json.loads(_frozen_path(cfg).read_text()).get(fp)
    except (OSError, ValueError):
        return None
    return [np.array(m, int) for m in saved] if saved else None


def _save_frozen(cfg, fp, merged):
    if not cfg["themes"].get("freeze", True):
        return
    path = _frozen_path(cfg)
    try:
        allsaved = json.loads(path.read_text())
    except (OSError, ValueError):
        allsaved = {}
    allsaved.pop(fp, None)
    allsaved[fp] = [np.asarray(m).tolist() for m in merged]
    allsaved = dict(list(allsaved.items())[-8:])                    # keep the latest few (other scopes / settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(allsaved))


def _scaled(c, n_pool, n_target):
    """Size thresholds that fit a small survey: a theme needs at most its share of the clauses (half of an even split over the
    requested themes), never less than 3, so 150 answers asked for 12 themes are not held to a minimum meant for thousands."""
    share = n_pool / max(1, 2 * (n_target or 10))
    return dict(c, min_theme_clauses=int(max(3, min(c["min_theme_clauses"], share))))


def _find_candidates(clauses, Ecl, pool, c, n_target, noise, seed):
    """The candidate themes. The first try uses the configured thresholds. When that gives fewer themes than were asked for (and
    themes.expand is not false), looser attempts follow, each keeping the result with the most themes: smaller theme size, rarer
    words allowed as topic words, then dense regions of clauses with smaller clusters. Fewer than asked after all of it = a warning."""
    aspects = c.get("method", "aspects") == "aspects"
    want = n_target or c["min_themes"]
    best = []

    def take(m):
        nonlocal best
        if len(m) > len(best):
            best = m
        return len(best) >= want

    k = (n_target or 10) + 2                                        # slack: some groups are too small, merged or generic later
    if aspects:
        take(_aspect_candidates(clauses, Ecl, pool, c, k, noise))
    if len(best) < c["min_themes"] or not aspects:
        take(_candidates(Ecl, pool, c, seed))                         # no usable words (or the dense-region method): dense regions
    if len(best) >= want or not c.get("expand", True):
        return best
    cs = _scaled(c, len(pool), n_target)
    if aspects:
        for df in dict.fromkeys([cs.get("aspect_min_df", 4), 3, 2]):
            if take(_aspect_candidates(clauses, Ecl, pool, dict(cs, aspect_min_df=df), k, noise)):
                return best
    cc = dict(cs)
    while len(best) < want and cc["min_cluster_size"] > 4:
        cc = dict(cc, min_cluster_size=max(4, cc["min_cluster_size"] - 2), min_samples=max(2, cc["min_samples"] - 1), min_cluster_frac=0)
        take(_candidates(Ecl, pool, cc, seed))
    return best


def _split_to_count(merged, Ec, n):
    """Exactly n themes when the data has the clauses for it: the largest theme is split in two (ward linkage on its clause vectors, no
    random step) until there are n. The requested number is a target, not a ceiling: a theme that is only a little tighter than a
    neighbour is still reported separately."""
    merged = list(merged)
    while n and len(merged) < n:
        splittable = [i for i, m in enumerate(merged) if len(m) >= 2]
        if not splittable:
            detail(f"[themes] only {len(merged)} themes can be made from the clauses; {n} were asked for")
            break
        i = max(splittable, key=lambda i: (len(merged[i]), -i))
        m = merged[i]
        lab = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(Ec[m])
        parts = sorted((m[lab == k] for k in (0, 1)), key=lambda p: (-len(p), int(p.min())))
        detail(f"[themes] {len(merged)} themes found, {n} asked for: a theme of {len(m)} clauses is split into {len(parts[0])} and {len(parts[1])}")
        merged[i:i + 1] = parts
    return merged


def discover(clauses, E, cfg):
    """clauses must have a 0..n-1 index aligned with E. Returns (themes, C)."""
    c, seed = cfg["themes"], cfg["seed"]
    rng = np.random.default_rng(seed)
    ok = np.where(clauses["clause"].str.split().str.len() >= c["min_words"])[0]
    if len(ok) > c["max_fit_clauses"]:
        ok = rng.choice(ok, c["max_fit_clauses"], replace=False)
    if len(ok) < 2 * c["min_cluster_size"]:
        raise ValueError(f"Only {len(ok)} usable clauses; too few for theme discovery. "
                         "Use themes.scope: pooled or lower themes.min_cluster_size.")
    mu = E.mean(0, keepdims=True)
    Ec = center(E, mu, c.get("center_strength", 1.0), axis_of(cfg)) if c.get("center") else E        # centroids live in the same space assign.run scores in
    n_target = c.get("n_themes") or c.get("max_themes")
    axis = axis_of(cfg)
    Ecl = center(E, mu, 0.0, axis) if axis is not None else E     # clustering space: with a sentiment axis, praise / complaint wording is taken out
    pool = ok                                           # every clause: the themes found in the answers must not depend on the themes you add
    thr = c.get("general_exclude_sim")
    general = [r for r in (c.get("required") or []) if isinstance(r, dict) and r.get("general")]
    gv = None
    if general:
        from . import embed
        gseeds = list(general[0].get("seeds") or _auto_seeds(general[0]["name"]))
        gV = embed.encode_cached([sha("seed:" + x.lower()) for x in gseeds], gseeds, cfg)
        gv = _unit(center(gV, mu, c.get("center_strength", 1.0), axis_of(cfg)).mean(0))
    if thr is not None and general:
        # Generic remarks ("I like it", "Very good", "Everything is right") say nothing about a topic, but clustering still groups
        # them, and they end up as a mixed-up theme (a "Price" theme full of "nothing to change"). Clauses this close to the
        # general-comment theme are left out of the clustering; they are still assigned to a theme afterwards.
        from . import embed
        seeds = list(general[0].get("seeds") or _auto_seeds(general[0]["name"]))
        V = embed.encode_cached([sha("seed:" + x.lower()) for x in seeds], seeds, cfg)
        v = _unit(center(V, mu, c.get("center_strength", 1.0), axis_of(cfg)).mean(0))
        keep = (Ec[ok] @ v) < thr
        if keep.sum() >= 2 * c["min_cluster_size"]:
            pool = ok[keep]
    noise, generic = _noise_for(clauses, pool, c, cfg, mu)
    fp = _discovery_fingerprint(clauses, general, cfg)
    merged = _load_frozen(cfg, fp)
    if merged is None:
        Ecl = _with_aspects(clauses, Ecl, mu, axis, c, cfg, noise)
        merged = _find_candidates(clauses, Ecl, pool, c, n_target, noise, seed)
        _save_frozen(cfg, fp, merged)
    n_found = len(merged)
    ctx = _naming_context(clauses, c, noise, generic)
    merged = _merge_duplicates(clauses, merged, c, Ec, mu, cfg, ctx, gv)          # pooled sizes count for the cut below
    merged = _split_oversized(merged, Ec, c)
    merged = _reduce(merged, Ec, n_target)
    merged = _split_to_count(merged, Ec, n_target)
    if len(merged) < c["min_themes"]:
        raise ValueError(f"Only {len(merged)} themes found. Lower themes.min_cluster_size / min_theme_clauses.")
    themes, merged = _name_and_split(clauses, merged, c, Ec, mu, cfg, n_target, ctx)
    for t in themes:
        t["candidates_found"] = n_found
    if gv is not None and c.get("fold_generic", True):
        themes, merged = _fold_generic(themes, merged, Ec, gv, c)
        goal = (c.get("n_total") or n_target or 0) - sum(1 for r in general if not r.get("hidden"))     # found themes the output needs besides General comment
        if goal and len(merged) < goal:                              # folding generic themes left too few: split the largest again
            merged = _split_to_count(merged, Ec, goal)
            themes = _name_themes(clauses, merged, c, Ec, mu, cfg, ctx)
            for t in themes:
                t["candidates_found"] = n_found
    if ctx["index"] is not None:                                     # before curation, so edits written against the names see the final ones
        _recheck_names(themes, _shown_members(merged, Ec, gv, cfg["assign"]["min_sim"], c.get("refine_iters", 2),
                                              cfg["assign"].get("fallback_min_sim"), cfg["assign"].get("secondary_margin", 0.1)), ctx, c)
    themes, merged = _curate(themes, merged, c)
    C = np.stack([_unit(Ec[m].mean(0)) for m in merged]).astype(np.float32) if merged else np.zeros((0, Ec.shape[1]), np.float32)
    for t in themes:
        parts = t.pop("_parts", None)
        if parts:                                                   # a merged theme keeps a centre for each theme it was made of
            t["sub_centres"] = [np.round(_unit(Ec[p].mean(0)), 4).tolist() for p in parts]
    names = [t["theme_name"] for t in themes]
    for name, seeds in ((c.get("curation") or {}).get("seeds") or {}).items():
        # sharpen a broad discovered theme with example phrases: centroid = members' mean + seeds' mean
        from . import embed
        if name not in names:
            raise ValueError(f"curation.seeds refers to unknown theme {name!r}. Themes: {names}")
        V = embed.encode_cached([sha("seed:" + x.lower()) for x in seeds], seeds, cfg)
        V = center(V, mu, c.get("center_strength", 1.0), axis_of(cfg)) if c.get("center") else V
        j = names.index(name)
        C[j] = _unit(C[j] + _unit(V.mean(0)))
    extra = []
    for name, spec in ((c.get("curation") or {}).get("add") or {}).items():
        # a theme defined by hand: centroid = mean of its example phrases (embedded and centred like the clauses)
        from . import embed
        seeds = spec["seeds"]
        V = embed.encode_cached([sha("seed:" + x.lower()) for x in seeds], seeds, cfg)
        V = center(V, mu, c.get("center_strength", 1.0), axis_of(cfg)) if c.get("center") else V
        words = list(spec.get("keywords", [])) or re.findall(r"[A-Za-z']{3,}", name.lower())
        centre, learned, members = _expand(_unit(V.mean(0)).astype(np.float32), Ec, clauses, c, words=words, cfg=cfg, mu=mu)
        n_mem = len(members)
        kws = list(dict.fromkeys(list(spec.get("keywords", [])) + learned))
        have = [i for i, t in enumerate(themes) if t["theme_name"].lower() == name.lower()]
        if have:                                           # discovery already produced this theme: sharpen it, do not duplicate
            j = have[0]
            C[j] = _unit(C[j] + centre)
            themes[j]["curated_keywords"] = list(dict.fromkeys(themes[j].get("curated_keywords", []) + kws))
            continue
        extra.append(centre)
        themes.append({"theme_name": name, "keywords": kws, "curated_keywords": kws, "frequency": n_mem, "seeded": True})
    if extra:
        C = np.vstack([C, np.stack(extra).astype(np.float32)])
    themes, C = _apply_required(themes, C, mu, c, cfg, Ec=Ec, clauses=clauses)
    floor = cfg["assign"]["min_sim"] if cfg is not None else 0.3
    for _ in range(c.get("refine_iters", 2)):
        # Clusters are built from the dense core of each topic only. Pull every centre towards all the clauses that are now
        # closest to it, so borderline clauses get matched and the centres sit in the middle of what they describe.
        S = Ec @ C.T
        top, best = S.argmax(1), S.max(1)
        for j in range(len(C)):
            near = (top == j) & (best >= floor)
            if near.sum() >= 5:
                C[j] = _unit(C[j] + _unit(Ec[near].mean(0)))
    if c.get("n_total"):
        themes, C = _trim_to_total(themes, C, Ec, c["n_total"], floor)
    # sizes are recounted against the final centres: a theme added by hand takes the clauses closest to it from the others
    S = Ec @ C.T
    top, best = S.argmax(1), S.max(1)
    for j, t in enumerate(themes):
        near = (top == j) & (best >= floor)
        t["frequency"] = int(near.sum())
        # Each theme gets its own entry bar, learned from how close its clear members are to it: a tight theme (packaging)
        # is stricter than a loose one, so vague answers ("I enjoyed it") do not fall into a specific theme by accident.
        q = c.get("floor_quantile", 0.05)
        t["min_sim"] = max(floor, min(float(np.quantile(best[near], q)), c.get("floor_max", 0.0))) if near.sum() >= 8 else floor
        if c.get("n_examples"):                                     # the clauses that sit nearest the centre say best what the theme is
            t["representative_clauses"] = _representatives(clauses["clause"], S[:, j], top == j, c["n_examples"])
    return themes, C
