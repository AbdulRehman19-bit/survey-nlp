import numpy as np
from survey_nlp import themes as th


def test_curate_merge_rename_drop_keywords():
    themes = [{"theme_name": n, "keywords": [n.lower()], "frequency": 3} for n in ("Milk", "Cold Milk", "Junk", "Smell")]
    merged = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5]), np.array([6, 7])]
    c = {"n_keywords": 8, "min_themes": 2, "curation": {
        "merge": {"Milk": ["Milk", "Cold Milk"]}, "rename": {"Smell": "Aroma"},
        "extra_keywords": {"Aroma": ["scent"]}, "drop": ["Junk"]}}
    t, m = th._curate(themes, merged, c)
    assert [x["theme_name"] for x in t] == ["Milk", "Aroma"]
    assert sorted(m[0].tolist()) == [0, 1, 2, 3, 4]
    assert t[1]["curated_keywords"] == ["scent"]


def test_typed_edit_with_unknown_name_is_an_error_but_config_edit_is_skipped():
    th_ = lambda: [{"theme_name": "A", "keywords": [], "frequency": 1}, {"theme_name": "B", "keywords": [], "frequency": 1}]
    m = lambda: [np.array([0]), np.array([1])]
    try:
        th._curate(th_(), m(), {"n_keywords": 8, "min_themes": 1, "curation": {"ops": [["drop", "Nope"]]}})
    except ValueError as e:
        assert "Nope" in str(e) and "A" in str(e)
    else:
        raise AssertionError("expected ValueError")
    t, _ = th._curate(th_(), m(), {"n_keywords": 8, "min_themes": 1, "curation": {"drop": ["Nope"]}})
    assert [x["theme_name"] for x in t] == ["A", "B"]                         # config edit: warned and skipped


def test_required_theme_added_or_merged(monkeypatch):
    # fake the embedder so the test needs no model: seeds map to fixed unit vectors by their text
    import survey_nlp.embed as embed
    vec = {"Price": np.array([0, 1], np.float32), "Flavour": np.array([1, 0], np.float32)}
    monkeypatch.setattr(embed, "encode_cached",
                        lambda keys, texts, cfg: np.stack([vec["Price" if "rice" in t else "Flavour"] for t in texts]))
    themes = [{"theme_name": "Taste", "keywords": ["taste"], "frequency": 5}]
    C = np.array([[1, 0]], np.float32)
    c = {"required": ["Flavour", {"name": "Price", "keywords": ["cost"]}], "required_merge_cos": 0.5}
    t, C2 = th._apply_required(themes, C, np.zeros((1, 2), np.float32), c, None)
    assert [x["theme_name"] for x in t] == ["Flavour", "Price"]      # Taste took over by meaning; Price is new
    assert t[1]["curated_keywords"] == ["cost"] and C2.shape == (2, 2)


def test_reduce_folds_small_themes_into_nearest_kept():
    Ec = np.array([[1, 0], [1, 0.1], [1, -0.1], [0, 1], [0.1, 1], [0.9, 0.2]], np.float32)
    merged = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5])]          # 3 themes; keep 2
    out = th._reduce(merged, Ec, 2)
    assert [len(m) for m in out] == [4, 2]                                    # the single clause joined the x-axis theme
    assert sorted(out[0].tolist()) == [0, 1, 2, 5]
    assert len(th._reduce(merged, Ec, 10)) == 3                               # nothing to reduce


def test_curate_ops_run_in_order_and_follow_renames():
    themes = [{"theme_name": n, "keywords": [n.lower()], "frequency": 1} for n in ("Easy", "Bottle", "Cough", "Taste")]
    merged = [np.array([0]), np.array([1]), np.array([2]), np.array([3])]
    c = {"n_keywords": 8, "min_themes": 2, "curation": {"ops": [
        ["merge", "Opening", ["Easy", "Bottle"]], ["rename", "opening", "Cap and bottle"], ["drop", "cough"]]}}
    t, m = th._curate(themes, merged, c)
    assert [x["theme_name"] for x in t] == ["Cap and bottle", "Taste"]
    assert sorted(m[0].tolist()) == [0, 1]


def test_theme_name_is_the_candidate_closest_in_meaning(monkeypatch):
    import pandas as pd
    import survey_nlp.embed as embed
    vec = {"battery": np.array([1, 0], np.float32), "screen": np.array([0, 1], np.float32)}
    monkeypatch.setattr(embed, "encode_cached", lambda keys, texts, cfg: np.stack([vec["screen" if "screen" in t else "battery"] for t in texts]))
    clauses = pd.DataFrame({"clause": ["battery lasts, nice screen", "battery died, screen fine", "good battery and screen"] * 4})
    Ec = np.tile(np.array([[0.2, 1.0]], np.float32), (12, 1))                  # this theme's centre points at "screen"
    c = {"spacy_model": "en_core_web_sm", "n_keywords": 4, "name_min_share": 0.0, "name_candidates": 8}
    out = th._name_themes(clauses, [np.arange(12)], c, Ec=Ec, mu=np.zeros((1, 2), np.float32), cfg=object())
    assert out[0]["theme_name"] == "Screen"


def test_curation_may_drop_every_discovered_theme_when_required_themes_fill_the_output():
    import numpy as np
    import pytest
    from survey_nlp.themes import _curate
    themes = [{"theme_name": "A", "keywords": []}, {"theme_name": "B", "keywords": []}]
    merged = [np.array([0]), np.array([1])]
    c = {"min_themes": 2, "n_keywords": 8, "curation": {"drop": ["A", "B"]}}
    with pytest.raises(ValueError):                                      # no required themes: still guarded
        _curate([dict(t) for t in themes], list(merged), c)
    kept, _ = _curate(themes, merged, {**c, "required": ["Price"]})
    assert kept == []


def test_cli_keeps_required_themes_from_the_config_unless_the_user_chose_some():
    from survey_nlp.cli import _overrides, _empty
    cfg = {"themes": {}}
    assert "required" not in _overrides(cfg, _empty())["themes"]
    e = {**_empty(), "required": [{"name": "Price"}]}
    assert _overrides(cfg, e)["themes"]["required"] == [{"name": "Price"}]


def test_merge_and_rename_keep_theme_names_unique_and_need_two_different_themes():
    import numpy as np
    import pytest
    from survey_nlp.themes import _curate
    c = {"min_themes": 2, "n_keywords": 8}

    def mk():
        th = [{"theme_name": n, "keywords": [], "frequency": 1} for n in ("A", "B", "C")]
        return th, [np.array([i]) for i in range(3)]

    for ops in ([["merge", "C", ["A", "B"]]], [["rename", "A", "c"]], [["merge", None, ["A", "A"]]]):
        th, m = mk()
        with pytest.raises(ValueError):
            _curate(th, m, dict(c, curation={"ops": ops}))
    th, m = mk()
    out, _ = _curate(th, m, dict(c, curation={"ops": [["merge", "A", ["A", "B"]]]}))      # keeping one of its own names is fine
    assert [t["theme_name"] for t in out] == ["A", "C"]


def test_theme_names_are_unique():
    import numpy as np, pandas as pd
    from survey_nlp.themes import _name_themes
    clauses = pd.DataFrame({"clause": ["sweet", "so sweet", "very sweet", "sweet sweet"] * 2})   # two clusters with the same only word
    merged = [np.arange(0, 4), np.arange(4, 8)]
    names = [t["theme_name"] for t in _name_themes(clauses, merged, {"spacy_model": "en_core_web_sm", "n_keywords": 8}, None, None, None)]
    assert len(set(names)) == 2, names


def _themes_and_centres(spec):
    """spec: [(name, kind, n_clauses)], kind 'found' | 'mine' | 'general'. One clause per direction, so each theme owns exactly n clauses."""
    import numpy as np
    d = len(spec)
    C = np.eye(d, dtype=np.float32)
    rows = [np.eye(d, dtype=np.float32)[j] for j, (_, _, n) in enumerate(spec) for _ in range(n)]
    themes = [{"theme_name": nm, **({"required": True} if k in ("mine", "general") else {}), **({"general": True} if k == "general" else {})}
              for nm, k, _ in spec]
    return themes, C, np.stack(rows)


def test_trim_drops_found_themes_first_then_the_smallest_of_your_own_and_the_general_theme_last():
    from survey_nlp.themes import _trim_to_total
    themes, C, Ec = _themes_and_centres([("Big", "mine", 30), ("Mid", "mine", 12), ("Tiny", "mine", 3), ("Gen", "general", 1),
                                         ("FoundA", "found", 20), ("FoundB", "found", 5)])
    kept, Ck = _trim_to_total(themes, C, Ec, 4, 0.5)
    assert [t["theme_name"] for t in kept] == ["Big", "Mid", "Tiny", "Gen"]               # both found themes go before any of yours
    assert Ck.shape[0] == 4
    kept, _ = _trim_to_total(themes, C, Ec, 3, 0.5)
    assert [t["theme_name"] for t in kept] == ["Big", "Mid", "Gen"]                       # then the SMALLEST of yours (not the first in the list)
    kept, _ = _trim_to_total(themes, C, Ec, 8, 0.5)
    assert len(kept) == 6                                                                  # nothing to trim


def test_trim_of_found_themes_does_not_depend_on_the_themes_you_added():
    from survey_nlp.themes import _trim_to_total
    a, Ca, Ea = _themes_and_centres([("FoundA", "found", 20), ("FoundB", "found", 5), ("Mine", "mine", 10)])
    kept, _ = _trim_to_total(a, Ca, Ea, 2, 0.5)
    assert [t["theme_name"] for t in kept] == ["FoundA", "Mine"]


def test_cache_keys_know_what_a_stage_really_depends_on():
    from survey_nlp import pipeline
    plan = {n: s for n, _, s in pipeline.PLAN}
    assert "assign.min_sim" in plan["themes"]                                              # discover() uses it as its floor
    from survey_nlp.ingest import protected_words
    cfg = type("C", (), {"raw": {"themes": {"required": [{"name": "Price", "keywords": ["cheap", "re:cost"]}, "Colour"],
                                           "curation": {"extra_keywords": {"X": ["fizzy"]}}}}})()
    cfg.__class__.__getitem__ = lambda self, k: self.raw[k]
    assert protected_words(cfg) == sorted({"Price", "cheap", "cost", "Colour", "fizzy"})
    assert pipeline._code_sig(*pipeline._code_for("themes")) != pipeline._code_sig(*pipeline._code_for("assign"))


def test_saved_clusters_are_reused_and_follow_the_data_and_discovery_settings(tmp_path):
    cfg = {"run_dir": str(tmp_path), "seed": 1, "embed": {"model": "m"}, "themes": {"n_themes": 5, "min_cluster_size": 10}}
    clauses = __import__("pandas").DataFrame({"clause_key": ["a", "b", "c"], "clause": ["a", "b", "c"]})
    fp = th._discovery_fingerprint(clauses, [], cfg)
    assert th._load_frozen(cfg, fp) is None
    th._save_frozen(cfg, fp, [np.array([0, 1]), np.array([2])])
    assert [m.tolist() for m in th._load_frozen(cfg, fp)] == [[0, 1], [2]]
    cfg["themes"]["required"] = [{"name": "Price"}]                  # your own themes do not change the clustering
    cfg["themes"]["curation"] = {"drop": ["X"]}
    assert th._discovery_fingerprint(clauses, [], cfg) == fp
    cfg["themes"]["min_cluster_size"] = 5                              # a discovery setting does
    assert th._discovery_fingerprint(clauses, [], cfg) != fp
    cfg["themes"]["min_cluster_size"] = 10
    cfg["themes"]["rediscover"] = True                                 # --rediscover ignores the saved clusters
    assert th._load_frozen(cfg, fp) is None


def test_name_candidate_pattern_counts_word_forms_but_not_other_words():
    import re
    rx = re.compile(th._stem_rx("crunch"))
    assert rx.search("so crunchy") and rx.search("a nice crunch")
    assert not rx.search("crunchiness is lost")
    assert re.compile(th._stem_rx("pieces")).search("the piece is big")


def test_clean_terms_drop_judging_words_repeats_and_overlapping_phrases():
    import pandas as pd
    texts = pd.Series(["easy drink overall", "the aftertaste lingers", "packaging is nice", "nice packaging", "easy drink",
                       "buy buy now", "white chocolate pieces", "white chocolate"] * 3)
    all_texts = pd.concat([texts, pd.Series(["some other answer here"] * 40)])
    pos = {"aftertaste": (1.0, 0.0), "lingers": (0.0, 0.0), "packaging": (1.0, 0.0), "chocolate": (1.0, 0.0), "white": (0.0, 1.0),
           "pieces": (1.0, 0.0), "drink": (1.0, 0.0), "buy": (0.0, 0.0), "overall": (0.0, 0.0), "easy": (0.0, 1.0)}
    ranked = ["easy drink", "overall", "aftertaste lingers", "lingers", "buy buy", "packaging", "white chocolate", "chocolate", "aftertaste"]
    got = [t for t, _, _ in th._clean_terms(ranked, texts, all_texts, pos, {}, limit=10)]
    assert got == ["packaging", "white chocolate", "aftertaste"]            # no "easy drink", "overall", verb "lingers", "buy buy" or the lone "chocolate"


def test_a_generic_cluster_is_folded_into_the_general_theme_but_a_topic_is_kept():
    gv = np.array([1, 0], np.float32)
    Ec = np.array([[0.95, 0.3], [0.9, 0.4], [0.1, 0.99], [0.2, 0.97]], np.float32)
    themes = [{"theme_name": "Easy", "n_informative": 3}, {"theme_name": "Packaging", "n_informative": 3},
              {"theme_name": "Pad", "n_informative": 0}, {"theme_name": "Price", "n_informative": 2}]
    merged = [np.array([0, 1]), np.array([2, 3]), np.array([2, 3]), np.array([2, 3])]
    keep, m = th._fold_generic(themes, merged, Ec, gv, {"generic_theme_sim": 0.5, "min_themes": 1})
    assert [t["theme_name"] for t in keep] == ["Packaging", "Price"]                  # close to the General seeds / no informative word: gone


def test_aspect_text_keeps_topic_words_and_drops_judging_and_function_words():
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "parser"])
    assert th._aspect_text(nlp("A really sweet flavour")) == "sweet flavour"
    assert "love" not in th._aspect_text(nlp("I love the packaging")).split()
    assert "packaging" in th._aspect_text(nlp("I love the packaging")).split()


def test_aspect_themes_group_topic_words_that_belong_together_without_randomness():
    import pandas as pd
    a = ["the bottle looks great", "nice label design", "the packaging is smart", "lovely bottle and label", "packaging design is good"] * 3
    b = ["the price is fair", "good value for money", "price feels high", "worth the price", "value is poor"] * 3
    clauses = pd.DataFrame({"clause": a + b})
    E = np.array([[1, 0.05]] * len(a) + [[0.05, 1]] * len(b), np.float32)
    c = {"spacy_model": "en_core_web_sm", "min_theme_clauses": 5, "split_large_frac": 0.9}
    first = th._aspect_candidates(clauses, E, np.arange(len(clauses)), c, 4)
    again = th._aspect_candidates(clauses, E, np.arange(len(clauses)), c, 4)
    groups = sorted(sorted(g.tolist()) for g in first)
    assert groups and all(all(i < 15 for i in g) or all(i >= 15 for i in g) for g in groups)   # no theme mixes packaging and price answers
    assert groups == sorted(sorted(g.tolist()) for g in again)


def _fake_topic_embedder(monkeypatch, topics):
    """Every text embeds to the unit vector of the first topic word it contains (no model needed)."""
    import survey_nlp.embed as embed
    eye = np.eye(len(topics), dtype=np.float32)
    monkeypatch.setattr(embed, "encode_cached", lambda keys, texts, cfg: np.stack(
        [eye[next((i for i, w in enumerate(topics) if w in t), 0)] for t in texts]))
    return eye


def test_a_name_used_by_too_few_clauses_gives_way_to_the_next_candidate_or_is_marked_mixed(monkeypatch):
    import pandas as pd
    _fake_topic_embedder(monkeypatch, ["screen", "battery"])
    clauses = pd.DataFrame({"clause": ["the battery lasts long", "battery died fast", "the battery is big", "love the battery life"] * 3
                            + ["the screen is sharp", "nice screen", "screen was bright", "a sharp screen", "the screen cracks"]
                            + ["the box was small", "a small box", "a plain box"]})
    Ec = np.tile(np.array([[1.0, 0.1]], np.float32), (len(clauses), 1))                 # the theme's centre points at "screen"
    base = {"spacy_model": "en_core_web_sm", "n_keywords": 6, "name_min_share": 0.0, "name_candidates": 8}
    name = lambda **kw: _only(th._name_themes(clauses, [np.arange(len(clauses))], dict(base, **kw), Ec=Ec, mu=np.zeros((1, 2), np.float32), cfg=object()))
    t = name()
    assert t["theme_name"] == "Screen" and "name_mixed" not in t                         # without the check: the closest in meaning
    t = name(name_min_coverage=0.5)
    assert t["theme_name"] == "Battery" and t["name_coverage"] >= 0.5 and "name_mixed" not in t     # screen covers 5/20, battery 12/20
    t = name(name_min_coverage=0.9)
    assert t["theme_name"] == "Screen" and t["name_mixed"] is True                       # none passes: best name kept, flagged


def _only(themes):
    assert len(themes) == 1
    return themes[0]


def test_a_mixed_theme_is_split_in_two_only_when_both_halves_fit_a_name_and_are_big_enough(monkeypatch):
    import pandas as pd
    topics = ["battery", "screen", "price", "camera", "speaker"]
    eye = _fake_topic_embedder(monkeypatch, topics)
    forms = ["the {} is here", "{} again", "about the {}", "{} was there", "my {}", "that {}", "this {}", "{} too"]
    clauses = pd.DataFrame({"clause": [f.format(t) for t in topics for f in forms]})
    Ec = np.stack([eye[i] for i in range(5) for _ in forms])
    base = {"spacy_model": "en_core_web_sm", "n_keywords": 6, "name_min_share": 0.0, "name_candidates": 12, "name_min_coverage": 0.25,
            "min_theme_clauses": 8}
    one = [np.arange(len(clauses))]                                                      # five topics in one theme: every name covers 1/5 of it
    themes, merged = th._name_and_split(clauses, one, base, Ec, np.zeros((1, 5), np.float32), object())
    assert len(themes) == 2 and sorted(np.concatenate(merged).tolist()) == list(range(40))
    assert not any(t.get("name_mixed") for t in themes) and all(t["name_coverage"] >= 0.25 for t in themes)
    again, merged2 = th._name_and_split(clauses, one, base, Ec, np.zeros((1, 5), np.float32), object())
    assert [t["theme_name"] for t in again] == [t["theme_name"] for t in themes]       # deterministic
    assert [m.tolist() for m in merged2] == [m.tolist() for m in merged]
    themes, merged = th._name_and_split(clauses, one, dict(base, min_theme_clauses=30), Ec, np.zeros((1, 5), np.float32), object())
    assert len(themes) == 1 and themes[0]["name_mixed"] is True                          # halves too small: kept whole, best name, flagged
    themes, merged = th._name_and_split(clauses, one, dict(base, name_min_coverage=0.0), Ec, np.zeros((1, 5), np.float32), object())
    assert len(themes) == 1 and "name_mixed" not in themes[0]                            # the check is off


def test_duplicate_pairs_use_centres_main_words_or_names_and_skip_themes_that_may_not_merge():
    C = np.array([[1, 0, 0], [0.95, 0.31, 0], [0, 1, 0], [0, 0.2, 0.98]], np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    kws = [["packaging", "label"], ["design"], ["bottle", "label", "packaging"], ["price"]]
    ok = [True] * 4
    pairs = lambda c, names=None, elig=ok: [(i, j) for _, i, j, _ in th._duplicate_pairs(C, kws, names, elig, c)]
    assert pairs({"merge_sim": 0.9}) == [(0, 1)]                                           # centres
    assert pairs({"merge_jaccard": 0.5}) == [(0, 2)]                                       # main words: 2 of 3 shared
    assert pairs({}) == []                                                                 # every signal off
    names = np.array([[1, 0], [0, 1], [0, 1], [0.1, 0.99]], np.float32)
    assert pairs({"merge_name_sim": 0.95}, names) == [(1, 2), (1, 3), (2, 3)]                # names that mean the same
    both = th._duplicate_pairs(C, kws, None, ok, {"merge_sim": 0.9, "merge_jaccard": 0.5})
    assert {(i, j) for _, i, j, _ in both} == {(0, 1), (0, 2)} and {w for *_, w in both} == {"centres", "main words"}
    assert pairs({"merge_sim": 0.9}, elig=[True, False, True, True]) == []                 # a protected theme is never a partner


def test_protected_names_cover_required_and_curated_themes():
    c = {"required": ["Price", {"name": "General comment"}], "curation": {"merge": {"Taste": ["Flavour", "Sweet"]}, "rename": {"Smell": "Aroma"},
         "drop": ["Cough"], "ops": [["merge", "Opening", ["Easy", "Bottle"]], ["rename", "Pack", "Box"], ["drop", "Junk"]]}}
    assert th._protected_names(c) >= {"price", "general comment", "taste", "flavour", "sweet", "smell", "cough", "opening", "easy", "bottle", "pack", "junk"}


def _merge_setup(monkeypatch):
    import pandas as pd
    _fake_topic_embedder(monkeypatch, ["packaging", "wrapper", "price"])
    forms = ["the {} is here", "{} again", "about the {}", "{} was there", "my {}", "that {}", "this {}", "{} too", "nice {}", "some {}"]
    clauses = pd.DataFrame({"clause": [f.format(t) for t in ("packaging", "wrapper", "price") for f in forms]})
    # packaging and wrapper point almost the same way; price is elsewhere
    Ec = np.array([[1, 0.02, 0]] * 10 + [[1, 0.0, 0.02]] * 10 + [[0, 0, 1]] * 10, np.float32)
    Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
    groups = [np.arange(0, 10), np.arange(10, 20), np.arange(20, 30)]
    c = {"spacy_model": "en_core_web_sm", "n_keywords": 6, "name_min_share": 0.0, "name_candidates": 8, "merge_sim": 0.95}
    return clauses, groups, Ec, c


def test_look_alike_candidates_are_merged_before_the_cut_but_not_when_protected_or_generic(monkeypatch):
    clauses, groups, Ec, c = _merge_setup(monkeypatch)
    mu = np.zeros((1, 3), np.float32)
    ctx = th._naming_context(clauses, c)
    out = th._merge_duplicates(clauses, groups, c, Ec, mu, object(), ctx, None)
    assert sorted(len(m) for m in out) == [10, 20] and sorted(np.concatenate(out).tolist()) == list(range(30))      # sizes pooled, nothing lost
    again = th._merge_duplicates(clauses, groups, c, Ec, mu, object(), ctx, None)
    assert [m.tolist() for m in out] == [m.tolist() for m in again]                                                # deterministic
    kept = th._merge_duplicates(clauses, groups, dict(c, required=["Wrapper"]), Ec, mu, object(), ctx, None)
    assert sorted(len(m) for m in kept) == [10, 10, 10]                                                           # a required theme's name: left alone
    kept = th._merge_duplicates(clauses, groups, dict(c, curation={"rename": {"Packaging": "Pack"}}), Ec, mu, object(), ctx, None)
    assert sorted(len(m) for m in kept) == [10, 10, 10]                                                           # a theme the curation mentions
    kept = th._merge_duplicates(clauses, groups, c, Ec, mu, object(), ctx, np.array([1, 0.01, 0], np.float32))     # General comment's direction
    assert sorted(len(m) for m in kept) == [10, 10, 10]                                                           # generic candidates are not rescued by a merge
    assert th._merge_duplicates(clauses, groups, {k: v for k, v in c.items() if k != "merge_sim"}, Ec, mu, object(), ctx, None) is groups   # off


def test_noun_head_check_rejects_describing_words_and_non_nouns_even_when_the_tagger_says_noun():
    from survey_nlp import quality
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    texts = ["too greasy", "it is greasy", "very greasy", "so greasy", "really greasy", "a bit greasy", "nice texture", "love the texture",
             "away from home", "take it away", "going away", "far away", "put away", "texture again"]
    usage = quality.WordUsage([t.lower() for t in texts])
    pos = {"greasy": (1.0, 0.0), "texture": (1.0, 0.0), "away": (0.0, 0.0), "crunchy": (0.1, 0.8)}
    assert quality.is_noun_head("texture", pos, usage) and not quality.is_noun_head("greasy", pos, usage)
    assert not quality.is_noun_head("away", pos, usage)                                  # the answers never use it as a noun
    assert not quality.is_noun_head("crunchy", pos, usage)                               # the learned table says adjective
    assert quality.is_noun_head("mouth", pos, usage, "NOUN") and not quality.is_noun_head("mouth", pos, usage, "ADJ")   # unseen word: the tag decides


def _aspect_setup(monkeypatch, labels_vec):
    import pandas as pd
    import survey_nlp.embed as embed
    vec = {"greasy": [1, 0, 0], "mouth": [0.6, 0.8, 0], "texture": [0.9, 0.1, 0.1], "price": [0, 0, 1]}
    def enc(keys, texts, cfg):
        return np.stack([np.array(vec.get(t.split()[0], [0, 0, 0]), np.float32) if not t.startswith("zzz") else labels_vec for t in texts])
    monkeypatch.setattr(embed, "encode_cached", enc)
    clauses = pd.DataFrame({"clause": ["too greasy", "it is greasy", "very greasy", "so greasy", "a greasy mouth", "greasy mouth feel", "really greasy",
                                       "greasy again", "that greasy mouth", "is greasy", "greasy mouth", "my mouth"] * 2})
    Ec = np.tile(np.array([[1.0, 0.0, 0.0]], np.float32), (len(clauses), 1))
    c = {"spacy_model": "en_core_web_sm", "n_keywords": 6, "name_min_share": 0.0, "name_candidates": 8, "aspect_labels": ["Texture", "Price"],
         "aspect_label_min_sim": 0.45}
    return clauses, Ec, c


def test_a_describing_word_name_becomes_the_closest_aspect_label_only_when_close_enough(monkeypatch):
    clauses, Ec, c = _aspect_setup(monkeypatch, None)
    mu = np.zeros((1, 3), np.float32)
    name = lambda **kw: _only(th._name_themes(clauses, [np.arange(len(clauses))], dict(c, **kw), Ec=Ec, mu=mu, cfg=object()))
    t = name(aspect_labels=[])
    assert t["theme_name"] == "Greasy"                                                     # off: the describing word, as before
    t = name()
    assert t["theme_name"] == "Texture" and t["name_aliases"] == ["greasy"]                 # centre [1,0,0] is 0.9 from "texture"
    t = name(aspect_labels=["Price"])
    assert t["theme_name"] == "Greasy Mouth" and "name_aliases" not in t                    # no label close enough: the best candidate headed by a noun
    t = name(aspect_labels=["Texture"], aspect_label_min_sim=0.999)
    assert t["theme_name"] == "Greasy Mouth" and "name_aliases" not in t


def test_an_aspect_label_is_covered_by_the_descriptor_it_replaced():
    from survey_nlp import quality
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    ix = quality.ClauseIndex(["too greasy", "greasy again", "nice price"], nlp)
    assert quality.name_coverage(ix, "Texture", [0, 1, 2]) == 0.0
    assert quality.name_coverage(ix, "Texture", [0, 1, 2], ["greasy"]) == 2 / 3


def test_phrase_candidates_are_noun_chunks_and_runs_of_content_words():
    import spacy
    nlp = spacy.load("en_core_web_sm")
    got = th._clause_phrases(nlp("The white chocolate pieces are really crunchy, but the box is too small."))
    assert "white chocolate pieces" in got and "chocolate pieces" in got and "white chocolate" in got       # chunk and its runs
    assert "box" in got and "the box" not in got                                                              # articles trimmed
    assert not any(p.startswith(("the ", "is ", "are ")) for p in got)
    assert "crunchy box" not in got                                                                           # a comma / function words end a run


def test_repeats_of_a_better_phrase_are_dropped_and_the_rest_is_varied_by_mmr():
    assert th._phrase_norm("White Pieces".lower()) == "white piece"
    terms = ["too sweet", "sweet", "sweetness", "white pieces", "piece", "packaging", "pieces of box"]
    assert [terms[i] for i in th._drop_variants(terms)] == ["too sweet", "white pieces", "packaging"]
    V = np.array([[1, 0, 0], [0.99, 0.14, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    rel = [1.0, 0.95, 0.9, 0.85]
    assert th._mmr(rel, V, 3, 1.0) == [0, 1, 2]                                  # relevance only: the near-copy comes second
    assert th._mmr(rel, V, 2, 0.6) == [0, 2]                                     # with variety: the copy gives way to a different topic
    assert th._mmr(rel, V, 9, 0.6) == th._mmr(rel, V, 9, 0.6) and len(th._mmr(rel, V, 9, 0.6)) == 4


def test_theme_phrases_end_to_end_have_no_repeats_and_representatives_are_nearest_and_unique(monkeypatch):
    import pandas as pd
    _fake_topic_embedder(monkeypatch, ["packaging", "label", "price"])
    clauses = pd.DataFrame({"clause": ["the box packaging is smart", "smart box packaging", "love the label design", "label design is neat",
                                       "the box label looks good", "box packaging again", "neat label design", "smart box packaging here"] * 2})
    Ec = np.tile(np.array([[1.0, 0.2, 0.0]], np.float32), (len(clauses), 1))
    c = {"spacy_model": "en_core_web_sm", "n_keywords": 6, "name_min_share": 0.0, "name_candidates": 8, "n_phrases": 4, "phrase_min_count": 2}
    t = _only(th._name_themes(clauses, [np.arange(len(clauses))], c, Ec=Ec, mu=np.zeros((1, 3), np.float32), cfg=object()))
    kws = t["keywords"]
    assert 1 <= len(kws) <= 4 and len(set(kws)) == len(kws)
    assert not any(a != b and (th._phrase_norm(a) in th._phrase_norm(b)) for a in kws for b in kws)           # no phrase inside another
    assert any(" " in k for k in kws)                                                                          # real phrases, not only words
    texts = pd.Series(["far", "near", "near", "mid", "nearer"])
    sims = np.array([0.1, 0.9, 0.9, 0.5, 0.95])
    assert th._representatives(texts, sims, np.ones(5, bool), 3) == ["nearer", "near", "mid"]                  # nearest first, each text once
    assert th._representatives(texts, sims, np.array([1, 0, 0, 0, 0], bool), 3) == ["far"]                     # only what the theme owns
    assert th._representatives(texts, sims, np.zeros(5, bool), 2) == ["nearer", "near"]                        # owns nothing: closest overall


def test_opinion_words_are_found_from_the_data_as_outliers_on_the_sentiment_axis(monkeypatch):
    import survey_nlp.embed as embed
    axis = np.array([1, 0, 0], np.float32)
    monkeypatch.setattr(th, "axis_of", lambda cfg: axis)
    judging = {"lovely", "dreadful"}
    neutral = [f"topic{chr(97 + i)}" for i in range(20)]
    vec = lambda w: np.array([1.0 if w in judging else 0.05 * (hash(w) % 3), 0.5, 0.5], np.float32)
    monkeypatch.setattr(embed, "encode_cached", lambda keys, texts, cfg: np.stack([vec(t) for t in texts]))
    rows = [[w] for w in neutral + sorted(judging)] * 3
    c = {"filler_k": 2.0}
    got = th._opinion_words(rows, None, c, object())
    assert {"lovely", "dreadful"} <= got and not any(w in got for w in neutral)
    assert "lovelys" in got                                                            # plural forms too, so n-gram tokens match
    assert th._opinion_words(rows, None, {"filler_k": 50.0}, object()) == set()      # a stricter k finds none: relative, not a fixed distance
    monkeypatch.setattr(th, "axis_of", lambda cfg: None)
    assert th._opinion_words(rows, None, c, object()) == set()                         # no sentiment axis: nothing to find


def test_filler_modes_choose_the_word_lists_and_language_mode_has_no_food_or_drink_words(monkeypatch):
    import pandas as pd
    clauses = pd.DataFrame({"clause": ["the drink is good"] * 4})
    noise, generic = th._noise_for(clauses, np.arange(4), {"filler_mode": "fixed"}, None)
    assert noise is th._NOISE and "drink" in noise and "good" in noise
    noise, generic = th._noise_for(clauses, np.arange(4), {"filler_mode": "language"}, None)
    assert "good" in noise and "very" in noise and "thing" in generic                 # general English judging / filler words remain
    assert not (th._DOMAIN_FLAVOURED & noise)                                         # no "drink", "meal", "buy", "day", "people"
    assert th._noise_for(clauses, np.arange(4), {}, None)[0] is th._NOISE            # no setting: the original behaviour


def test_the_saved_clusters_fingerprint_follows_the_filler_settings_and_the_word_lists(monkeypatch):
    import pandas as pd
    cfg = {"run_dir": "x", "seed": 1, "embed": {"model": "m"}, "themes": {"n_themes": 5, "filler_mode": "language"}}
    clauses = pd.DataFrame({"clause_key": ["a", "b"], "clause": ["a", "b"]})
    fp = th._discovery_fingerprint(clauses, [], cfg)
    cfg["themes"]["filler_mode"] = "data"
    assert th._discovery_fingerprint(clauses, [], cfg) != fp                          # a discovery setting
    cfg["themes"]["filler_mode"] = "language"
    assert th._discovery_fingerprint(clauses, [], cfg) == fp
    monkeypatch.setattr(th, "_NOISE", th._NOISE | {"extra"})
    assert th._discovery_fingerprint(clauses, [], cfg) != fp                          # editing the built-in list re-discovers


def _unit_rows(a):
    a = np.asarray(a, np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_merge_needs_a_similarity_that_is_an_outlier_among_this_surveys_pairs_and_a_second_signal():
    rng = np.random.default_rng(0)
    base = np.eye(6, 8, dtype=np.float32) + 0.05 * rng.standard_normal((6, 8)).astype(np.float32)
    C = _unit_rows(np.vstack([base, base[0] + 0.1 * rng.standard_normal(8).astype(np.float32)]))     # theme 6 is a copy of theme 0
    kws = [["packaging", "label"], ["price"], ["taste"], ["crunch"], ["milk"], ["colour"], ["packaging", "label", "bottle"]]
    ok = [True] * 7
    got = th._duplicate_pairs(C, kws, None, ok, {"merge_k": 1.5})
    assert [(i, j) for _, i, j, _ in got] == [(0, 6)] and "main words" in got[0][3]
    kws_other = [list(k) for k in kws]
    kws_other[6] = ["bottle"]
    assert th._duplicate_pairs(C, kws_other, None, ok, {"merge_k": 1.5}) == []              # similar centres alone are not enough
    assert th._duplicate_pairs(C, kws, None, ok, {"merge_k": 8.0}) == []                    # a stricter k: nothing stands out that far
    assert th._duplicate_pairs(C, kws, None, [True, False, True, True, True, True, True], {"merge_k": 1.5}) != []
    assert th._duplicate_pairs(C, kws, None, [False] + [True] * 5 + [False], {"merge_k": 1.5}) == []      # a protected theme is never a partner
    assert th._duplicate_pairs(C[:2], kws[:2], None, ok[:2], {"merge_k": 0.0}) == []                      # one pair: no distribution to be an outlier in
    assert th._duplicate_pairs(C[:3], kws[:3], None, ok[:3], {"merge_k": 1.5}) == []                      # three pairs: none can be 1.5 std out
    uniform = _unit_rows(np.tile([[1, 1, 1, 1]], (5, 1)) + np.eye(5, 4))
    assert th._duplicate_pairs(uniform, [["a"]] * 5, None, [True] * 5, {"merge_k": 1.5}) is not None   # no crash on near-identical centres
    names = _unit_rows(np.vstack([np.eye(6, 8), np.eye(6, 8)[0] + 0.01]))
    by_name = th._duplicate_pairs(C, kws_other, names, ok, {"merge_k": 1.5})
    assert [(i, j) for _, i, j, _ in by_name] == [(0, 6)] and "names" in by_name[0][3]                # names that mean the same count as the second signal
    assert th._duplicate_pairs(C, kws, None, ok, {"merge_sim": 0.99}) == []                            # the older absolute settings still work without merge_k


def test_a_theme_far_bigger_than_the_median_is_split_only_when_it_falls_apart_more_clearly_than_the_others():
    rng = np.random.default_rng(1)
    blob = lambda centre, n: np.array(centre, np.float32) + 0.05 * rng.standard_normal((n, 4)).astype(np.float32)
    small = [blob(v, 12) for v in ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0])]
    two = np.vstack([blob([0, 0, 0, 1], 25), blob([1, 1, 1, 1], 25)])                      # two clear topics glued together
    one = blob([1, 0, 1, 1], 50)                                                          # one smooth topic, just big
    c = {"split_ratio": 2.0, "split_k": 1.5, "split_min_gain": 0.1, "min_theme_clauses": 8}

    def run(big, cfg=c):
        Ec = _unit_rows(np.vstack(small + [big]))
        merged = [np.arange(12 * k, 12 * k + 12) for k in range(5)] + [np.arange(60, 110)]
        return [len(m) for m in th._split_oversized(merged, Ec, cfg)], merged
    assert run(two)[0] == [12] * 5 + [25, 25]                                             # falls apart far more clearly than the others: split
    assert run(one)[0] == [12] * 5 + [50]                                                 # a big smooth theme is left whole
    assert run(two, dict(c, split_ratio=5.0))[0] == [12] * 5 + [50]                       # not big enough relative to the median
    assert run(two, dict(c, min_theme_clauses=30))[0] == [12] * 5 + [50]                  # halves too small
    assert run(two, dict(c, split_k=50.0))[0] == [12] * 5 + [50]                          # stricter: not that far out
    Ec = _unit_rows(np.vstack(small + [two]))
    merged = [np.arange(12 * k, 12 * k + 12) for k in range(5)] + [np.arange(60, 110)]
    assert [m.tolist() for m in th._split_oversized(merged, Ec, c)] == [m.tolist() for m in th._split_oversized(merged, Ec, c)]     # deterministic
    assert th._split_oversized(merged, Ec, {}) is merged                                  # off without themes.split_ratio


def test_a_name_is_judged_by_how_much_more_the_theme_uses_it_than_the_rest_of_the_survey():
    import spacy
    from survey_nlp import quality
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    theme = ["adverts everywhere", "too many adverts"] + ["slow login", "login fails", "login again", "cannot login", "login code",
                                                            "login screen", "login help", "login error"]
    rest = [f"the taste is fine {i}" for i in range(20)] + ["login problem"] * 2
    index = quality.ClauseIndex(theme + rest, nlp)
    members = np.arange(10)
    inside, outside = quality.name_rates(index, "Adverts", members)
    assert inside == 0.2 and outside == 0.0 and quality.name_lift(inside, outside) == 10.0
    assert quality.name_ok(inside, outside, 0.10, 2.0) and not quality.name_ok(inside, outside, 0.25, 2.0)    # rare but distinctive: fine above the floor
    inside, outside = quality.name_rates(index, "Login", members)
    assert inside == 0.8 and outside < 0.1
    assert not quality.name_ok(0.40, 0.35, 0.10, 2.0) and quality.name_ok(0.40, 0.35, 0.10)                    # common everywhere: tells nothing; no min_lift = floor only
    c = {"name_min_coverage": 0.10, "name_min_lift": 2.0}
    assert th._name_passes(index, "Adverts", members, c) and not th._name_passes(index, "Taste", members, c)
    assert th._name_passes(index, "Login", members, c)


def test_edits_on_a_theme_you_added_reach_a_found_theme_of_that_name_only_when_there_is_one():
    mk = lambda: ([{"theme_name": n, "keywords": [], "frequency": 1} for n in ("Bottle", "Water")], [np.array([0]), np.array([1])])
    c = {"n_keywords": 8, "min_themes": 1, "curation": {"ops": [["rename_if_present", "Bottle", "Packaging"], ["drop_if_present", "Price"]]}}
    t, _ = th._curate(*mk(), c)
    assert [x["theme_name"] for x in t] == ["Packaging", "Water"]            # "Price" is not there: nothing to do, no error
    c["curation"]["ops"] = [["drop_if_present", "bottle"]]
    t, _ = th._curate(*mk(), c)
    assert [x["theme_name"] for x in t] == ["Water"]


def test_a_merged_theme_remembers_the_centres_of_its_parts_and_a_rename_keeps_the_match_name():
    Ec = np.array([[1, 0], [1, 0.1], [0, 1], [0.1, 1], [-1, 0]], np.float32)
    themes = [{"theme_name": n, "match_name": n, "keywords": [], "frequency": 2} for n in ("A", "B", "C")]
    merged = [np.array([0, 1]), np.array([2, 3]), np.array([4])]
    c = {"n_keywords": 8, "min_themes": 1, "curation": {"ops": [["merge", "AB", ["A", "B"]], ["merge", "ABC", ["AB", "C"]], ["rename", "ABC", "All"]]}}
    t, m = th._curate(themes, merged, c)
    assert len(t) == 1 and t[0]["theme_name"] == "All" and t[0]["match_name"] == "A"
    assert [p.tolist() for p in t[0]["_parts"]] == [[0, 1], [2, 3], [4]]     # parts of parts are flattened, not nested
