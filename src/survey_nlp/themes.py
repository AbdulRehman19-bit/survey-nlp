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


def _name_themes(clauses, merged, c):
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
    for i, m in enumerate(merged):
        order = np.argsort(-score[i])
        kws = terms[order[:c["n_keywords"]]].tolist()
        name = None
        for t in terms[order[:40]]:                                 # POS-guarded: head must be a noun
            last = nlp(t)[-1]
            if last.pos_ in ("NOUN", "PROPN") and t.title() not in used:
                name = t.title()
                break
        name = name or kws[0].title()
        used.add(name)
        out.append({"theme_name": name, "keywords": kws, "frequency": int(len(m))})
    return out


def _curate(themes, merged, c):
    """Optional human edits from config (themes.curation): merge / rename / extra_keywords / drop, by theme name."""
    cur = c.get("curation") or {}
    names = [t["theme_name"] for t in themes]

    def idx(n):
        if n not in names:
            raise ValueError(f"curation refers to unknown theme {n!r}. Discovered themes: {names}")
        return names.index(n)

    for new, group in (cur.get("merge") or {}).items():
        ids = [idx(n) for n in group]
        keep = ids[0]
        merged[keep] = np.concatenate([merged[i] for i in ids])
        kws = list(dict.fromkeys(k for i in ids for k in themes[i]["keywords"]))
        themes[keep].update(keywords=kws[:c["n_keywords"]], frequency=int(len(merged[keep])))
        for i in ids[1:]:
            themes[i]["_dead"] = True
        themes[keep]["theme_name"] = new
        names = [t["theme_name"] for t in themes]
    for old, new in (cur.get("rename") or {}).items():
        themes[idx(old)]["theme_name"] = new
        names = [t["theme_name"] for t in themes]
    for n, kws in (cur.get("extra_keywords") or {}).items():
        themes[idx(n)]["curated_keywords"] = list(kws)       # bypass the generic-keyword filter in assign
    for n in cur.get("drop") or []:
        themes[idx(n)]["_dead"] = True
    keep = [i for i, t in enumerate(themes) if not t.get("_dead")]
    if len(keep) < c["min_themes"]:
        raise ValueError(f"Only {len(keep)} themes left after curation.")
    return [themes[i] for i in keep], [merged[i] for i in keep]


def _auto_seeds(name):
    """Polarity-neutral example phrases for a theme the user only named."""
    return [name, f"the {name}", f"I like the {name}", f"the {name} is not good", f"{name} could be better"]


def _apply_required(themes, C, mu, c, cfg):
    """Themes the user says MUST be in the results (themes.required: names or {name, keywords, seeds}).
    If a discovered theme is the same topic it is renamed and sharpened; otherwise a new theme is added.
    Everything else that was discovered stays, so the user's list is added to, not replaced."""
    from . import embed
    for r in c.get("required") or []:
        r = {"name": r} if isinstance(r, str) else dict(r)
        name = r["name"].strip()
        if not name:
            continue
        kws = list(r.get("keywords") or re.findall(r"[A-Za-z']{3,}", name.lower()))
        seeds = list(r.get("seeds") or _auto_seeds(name))
        V = embed.encode_cached([sha("seed:" + x.lower()) for x in seeds], seeds, cfg)
        V = center(V, mu) if c.get("center") else V
        vec = _unit(V.mean(0)).astype(np.float32)
        have = [i for i, t in enumerate(themes) if t["theme_name"].lower() == name.lower()]
        sims = C @ vec
        if have:                                              # already there (discovered or curated): just sharpen
            j = have[0]
        elif len(sims) and float(sims.max()) >= c.get("required_merge_cos", 0.5):
            j = int(sims.argmax())                             # same topic as a discovered theme: take it over
            themes[j]["theme_name"] = name
        else:
            themes.append({"theme_name": name, "keywords": kws, "curated_keywords": kws, "frequency": 0,
                           "seeded": True, "required": True})
            C = np.vstack([C, vec[None]])
            continue
        C[j] = _unit(C[j] + vec)
        themes[j].update(curated_keywords=list(dict.fromkeys(themes[j].get("curated_keywords", []) + kws)), required=True)
    return themes, C


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
    merged.sort(key=len, reverse=True)
    merged = merged[:c["max_themes"]]
    if len(merged) < c["min_themes"]:
        raise ValueError(f"Only {len(merged)} themes found. Lower themes.min_cluster_size / min_theme_clauses.")
    themes = _name_themes(clauses, merged, c)
    themes, merged = _curate(themes, merged, c)
    mu = E.mean(0, keepdims=True)
    Ec = center(E, mu) if c.get("center") else E        # centroids live in the same space assign.run scores in
    C = np.stack([_unit(Ec[m].mean(0)) for m in merged]).astype(np.float32)
    names = [t["theme_name"] for t in themes]
    for name, seeds in ((c.get("curation") or {}).get("seeds") or {}).items():
        # sharpen a broad discovered theme with example phrases: centroid = members' mean + seeds' mean
        from . import embed
        if name not in names:
            raise ValueError(f"curation.seeds refers to unknown theme {name!r}. Themes: {names}")
        V = embed.encode_cached([sha("seed:" + x.lower()) for x in seeds], seeds, cfg)
        V = center(V, mu) if c.get("center") else V
        j = names.index(name)
        C[j] = _unit(C[j] + _unit(V.mean(0)))
    extra = []
    for name, spec in ((c.get("curation") or {}).get("add") or {}).items():
        # a theme defined by hand: centroid = mean of its example phrases (embedded and centred like the clauses)
        from . import embed
        seeds = spec["seeds"]
        V = embed.encode_cached([sha("seed:" + x.lower()) for x in seeds], seeds, cfg)
        V = center(V, mu) if c.get("center") else V
        extra.append(_unit(V.mean(0)))
        kws = list(spec.get("keywords", []))
        themes.append({"theme_name": name, "keywords": kws, "curated_keywords": kws, "frequency": 0, "seeded": True})
    if extra:
        C = np.vstack([C, np.stack(extra).astype(np.float32)])
    themes, C = _apply_required(themes, C, mu, c, cfg)
    return themes, C
