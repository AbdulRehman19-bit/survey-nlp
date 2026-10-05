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


def test_curate_unknown_name_is_explained():
    try:
        th._curate([{"theme_name": "A", "keywords": []}], [np.array([0])],
                   {"n_keywords": 8, "min_themes": 1, "curation": {"drop": ["Nope"]}})
    except ValueError as e:
        assert "Nope" in str(e) and "A" in str(e)
    else:
        raise AssertionError("expected ValueError")


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
