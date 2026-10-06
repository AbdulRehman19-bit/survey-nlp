import re
import numpy as np
import spacy
import umap
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from .util import center, sha


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


def _name_themes(clauses, merged, c, Ec=None, mu=None, cfg=None):
    nlp = spacy.load(c["spacy_model"], disable=["ner", "lemmatizer"])
    docs = [" ".join(clauses["clause"].iloc[m]) for m in merged]
    cv = CountVectorizer(ngram_range=(1, 2), stop_words="english",
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-']+\b")
    tf = cv.fit_transform(docs).toarray().astype(float)
    terms = np.array(cv.get_feature_names_out())
    w = tf / np.maximum(tf.sum(1, keepdims=True), 1)
    A = tf.sum() / len(docs)
    score = w * np.log(1 + A / np.maximum(tf.sum(0), 1))          # class-based TF-IDF
    out, used = [], set()
    all_texts = clauses["clause"].str.lower()
    for i, m in enumerate(merged):
        order = np.argsort(-score[i])
        kws = terms[order[:c["n_keywords"]]].tolist()
        name = None
        texts = clauses["clause"].iloc[m].str.lower()
        cands = []
        for t in terms[order[:40]]:                                 # POS-guarded: head must be a noun
            if len(t) < 4 or texts.str.contains(t, regex=False).mean() < c.get("name_min_share", 0.08):
                continue                                            # fragments ("don") and words few clauses in the cluster use
            last = nlp(t)[-1]
            if last.pos_ in ("NOUN", "PROPN") and t.title() not in used:
                cands.append(t)
                if len(cands) >= c.get("name_candidates", 8):
                    break
        name = None
        if cands and Ec is not None and cfg is not None and c.get("name_by_meaning", True):
            # Of the good-looking words, pick the one whose MEANING is closest to this theme's centre, so a theme about
            # "tastes nice / fruity / refreshing" is not named after one flavour that happens to be frequent.
            from . import embed
            V = embed.encode_cached([sha("name:" + t) for t in cands], cands, cfg)
            V = center(V, mu, c.get("center_strength", 1.0)) if c.get("center") else V
            centre = _unit(Ec[m].mean(0))
            sims = V @ centre - 0.01 * np.arange(len(cands))        # tiny preference for the better c-TF-IDF rank
            name = cands[int(np.argmax(sims))].title()
        elif cands:
            name = cands[0].title()
        name = name or kws[0].title()
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
        out.append({"theme_name": name, "keywords": kws, "assign_keywords": match_kws[:c["n_keywords"]],
                    "frequency": int(len(m))})
    return out


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

    def merge(new, group):
        ids = [idx(n) for n in group]
        keep = ids[0]
        merged[keep] = np.concatenate([merged[i] for i in ids])
        kws = list(dict.fromkeys(k for i in ids for k in themes[i]["keywords"]))
        match = list(dict.fromkeys(k for i in ids for k in themes[i].get("assign_keywords", [])))
        themes[keep].update(keywords=kws[:c["n_keywords"]], assign_keywords=match, frequency=int(len(merged[keep])))
        for i in ids[1:]:
            themes[i]["_dead"] = True
        themes[keep]["theme_name"] = new or themes[keep]["theme_name"]

    def soft(what, fn, *args):
        """Edits written in the config may name themes this data did not produce: warn and skip instead of failing."""
        try:
            fn(*args)
        except ValueError as e:
            print(f"WARNING: skipped curation ({what}): {e}")

    def set_name(old, new):
        themes[idx(old)]["theme_name"] = new

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
            themes[idx(op[1])]["theme_name"] = op[2]
        elif kind == "drop":                                 # ["drop", name]
            themes[idx(op[1])]["_dead"] = True
    keep = [i for i, t in enumerate(themes) if not t.get("_dead")]
    if len(keep) < c["min_themes"] and not (c.get("required") and (cur.get("drop") or cur.get("ops"))):   # dropping every discovered theme is fine when the user's own themes fill the output
        raise ValueError(f"Only {len(keep)} themes left after curation.")
    return [themes[i] for i in keep], [merged[i] for i in keep]


def _expand(vec, Ec, clauses, c, words=()):
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
            strict = np.zeros(len(Ec), bool)
            strict[np.argsort(-sims)[:c.get("required_min_members", 3)]] = True
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
    kws = [terms[i] for i in rank[:c.get("n_keywords", 8)] if good[i]]
    return centre, kws, m


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
        if not name:
            continue
        sem_only = bool(r.get("semantic_only"))                # no keywords at all, given or learned: matched by meaning only
        kws = [] if sem_only else list(r.get("keywords") or re.findall(r"[A-Za-z']{3,}", name.lower()))
        seeds = list(r.get("seeds") or _auto_seeds(name))
        V = embed.encode_cached([sha("seed:" + x.lower()) for x in seeds], seeds, cfg)
        V = center(V, mu, c.get("center_strength", 1.0)) if c.get("center") else V
        vec = _unit(V.mean(0)).astype(np.float32)
        vec, learned, members = _expand(vec, Ec, clauses, c, words=kws)
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
            name = r["name"].strip()
            specs.append((list(r.get("seeds") or _auto_seeds(name)),
                          [] if r.get("semantic_only") else list(r.get("keywords") or re.findall(r"[A-Za-z']{3,}", name.lower()))))
    for name, spec in ((c.get("curation") or {}).get("add") or {}).items():
        specs.append((spec["seeds"], list(spec.get("keywords", [])) or re.findall(r"[A-Za-z']{3,}", name.lower())))
    found = [np.array([], int)]
    for seeds, words in specs:
        V = embed.encode_cached([sha("seed:" + x.lower()) for x in seeds], seeds, cfg)
        V = center(V, mu, c.get("center_strength", 1.0)) if c.get("center") else V
        found.append(_expand(_unit(V.mean(0)).astype(np.float32), Ec, clauses, c, words=words)[2])
    return np.unique(np.concatenate(found))


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
    Ec = center(E, mu, c.get("center_strength", 1.0)) if c.get("center") else E        # centroids live in the same space assign.run scores in
    n_target = c.get("n_themes") or c.get("max_themes")
    hand = _hand_made_members(clauses, Ec, mu, c, cfg)
    pool = np.setdiff1d(ok, hand)                       # what the user's own themes cover is set aside, the rest is clustered
    if len(pool) < 2 * c["min_cluster_size"]:
        pool = ok
    cc = dict(c)
    merged = _candidates(E, pool, cc, seed)
    while c.get("expand", True) and n_target and len(merged) < n_target and cc["min_cluster_size"] > 4:
        # fewer candidates than requested: look for smaller topics before giving up
        cc = dict(cc, min_cluster_size=max(4, cc["min_cluster_size"] - 2), min_samples=max(2, cc["min_samples"] - 1),
                  min_cluster_frac=0)
        merged = _candidates(E, pool, cc, seed)
    n_found = len(merged)
    merged = _reduce(merged, Ec, n_target)
    if len(merged) < c["min_themes"]:
        raise ValueError(f"Only {len(merged)} themes found. Lower themes.min_cluster_size / min_theme_clauses.")
    themes = _name_themes(clauses, merged, c, Ec=Ec, mu=mu, cfg=cfg)
    for t in themes:
        t["candidates_found"] = n_found
    themes, merged = _curate(themes, merged, c)
    C = np.stack([_unit(Ec[m].mean(0)) for m in merged]).astype(np.float32) if merged else np.zeros((0, Ec.shape[1]), np.float32)
    names = [t["theme_name"] for t in themes]
    for name, seeds in ((c.get("curation") or {}).get("seeds") or {}).items():
        # sharpen a broad discovered theme with example phrases: centroid = members' mean + seeds' mean
        from . import embed
        if name not in names:
            raise ValueError(f"curation.seeds refers to unknown theme {name!r}. Themes: {names}")
        V = embed.encode_cached([sha("seed:" + x.lower()) for x in seeds], seeds, cfg)
        V = center(V, mu, c.get("center_strength", 1.0)) if c.get("center") else V
        j = names.index(name)
        C[j] = _unit(C[j] + _unit(V.mean(0)))
    extra = []
    for name, spec in ((c.get("curation") or {}).get("add") or {}).items():
        # a theme defined by hand: centroid = mean of its example phrases (embedded and centred like the clauses)
        from . import embed
        seeds = spec["seeds"]
        V = embed.encode_cached([sha("seed:" + x.lower()) for x in seeds], seeds, cfg)
        V = center(V, mu, c.get("center_strength", 1.0)) if c.get("center") else V
        words = list(spec.get("keywords", [])) or re.findall(r"[A-Za-z']{3,}", name.lower())
        centre, learned, members = _expand(_unit(V.mean(0)).astype(np.float32), Ec, clauses, c, words=words)
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
    n_total = c.get("n_total")
    if n_total and len(themes) > n_total:
        # themes.n_total = how many themes the output has IN TOTAL (the user's own included). Drop the smallest beyond that: themes
        # found in the data first (the likelier duplicates of a theme the user wrote), then the user's own. The clauses of a dropped
        # theme go to the nearest remaining one when they are assigned.
        S = Ec @ C.T
        size = np.bincount(S.argmax(1)[S.max(1) >= floor], minlength=len(themes))
        mine = lambda t: bool(t.get("required") or t.get("seeded"))
        order = sorted(range(len(themes)), key=lambda j: (mine(themes[j]), size[j]))
        gone = set(order[:len(themes) - n_total])
        keep = [j for j in range(len(themes)) if j not in gone]
        themes, C = [themes[j] for j in keep], C[keep]
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
    return themes, C
