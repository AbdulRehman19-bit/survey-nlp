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
