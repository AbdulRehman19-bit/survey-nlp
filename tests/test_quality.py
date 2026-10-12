import numpy as np
import pandas as pd
import pytest
import spacy
from survey_nlp import quality


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm", disable=["ner", "parser"])


def test_name_coverage_counts_word_forms_and_phrases_but_not_other_words(nlp):
    texts = ["the packaging is nice", "packaging was dull", "crunchy and fresh", "a nice crunch", "pieces of white chocolate",
             "one piece was white", "too sweet"]
    ix = quality.ClauseIndex(texts, nlp)
    allc = np.arange(len(texts))
    assert quality.name_coverage(ix, "Packaging", allc) == pytest.approx(2 / 7)
    assert quality.name_coverage(ix, "Crunch", [2, 3, 6]) == pytest.approx(2 / 3)          # "crunchy" and "crunch"
    assert quality.name_coverage(ix, "Pieces", [4, 5]) == 1.0                                # lemma: "piece" = "pieces"
    assert quality.name_coverage(ix, "White Chocolate", [4, 5]) == pytest.approx(0.5)       # a phrase needs both words together
    assert quality.name_coverage(ix, "Packaging & Sweetness", [0, 6]) == pytest.approx(0.5)  # a joined name: either part counts
    assert quality.name_coverage(ix, "Sweet 2", [6]) == 1.0                                  # the "2" of a repeated name is ignored
    assert quality.name_coverage(ix, "Packaging", []) == 0.0


def test_npmi_is_high_for_words_that_occur_together_and_low_for_words_that_never_do():
    a = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)
    b = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)
    c = np.array([0, 0, 0, 1, 1, 1, 0, 0], bool)
    assert quality.npmi([a, b]) == pytest.approx(1.0)
    assert quality.npmi([a, c]) == pytest.approx(-1.0)
    assert quality.npmi([a, np.zeros(8, bool)]) is None                                      # a word that occurs nowhere is left out
    assert quality.npmi([a, b, c]) < quality.npmi([a, b])


def test_gini_is_zero_for_equal_sizes_and_grows_with_imbalance():
    assert quality.gini([5, 5, 5, 5]) == 0.0
    assert quality.gini([1, 1, 1, 97]) > quality.gini([20, 25, 25, 30]) > 0.0
    assert quality.gini([7]) == 0.0


def test_a_describing_word_is_recognised_by_how_the_answers_use_it_not_by_the_tagger_alone(nlp):
    texts = ["too greasy", "it is greasy", "very greasy", "so greasy", "really greasy", "a bit greasy", "the texture is odd",
             "nice texture", "love the texture", "texture is ok", "great texture", "texture again"]
    ix = quality.ClauseIndex(texts, nlp)
    assert quality.is_adjective("greasy", {"greasy": (1.0, 0.0)}, ix)           # the tagger calls it a noun, the answers use it as a describing word
    assert not quality.is_adjective("texture", {"texture": (1.0, 0.0)}, ix)
    assert quality.is_adjective("crunchy", {"crunchy": (0.1, 0.8)}, ix)         # a rare word: the learned table decides
    assert quality._is_adjective("Greasy", {}, ix) and not quality._is_adjective("Texture", {}, ix)
    assert not quality._is_adjective("Greasy & Texture", {}, ix)                # a joined name counts only if every part is one


def test_keyword_jaccard_folds_plurals_and_splits_phrases():
    assert quality.keyword_jaccard(["packaging", "label", "bottle"], ["bottles", "packaging", "label"]) == 1.0
    assert quality.keyword_jaccard(["packaging", "white chocolate"], ["price", "value"]) == 0.0


def test_report_covers_size_spread_general_share_low_coverage_names_and_duplicates(nlp):
    clauses = pd.DataFrame({"clause_id": range(10), "clause": [
        "the packaging is nice", "packaging looks smart", "great packaging", "the price is high", "fair price",
        "price feels fine", "crunchy bite", "too sweet", "i like it", "it is good"]})
    assign = pd.DataFrame({"clause_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6], "scope": "ALL",
                           "theme_idx": [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 0]})
    themes = [{"theme_name": "Packaging", "keywords": ["packaging", "smart"]}, {"theme_name": "Price", "keywords": ["price"]},
              {"theme_name": "Crunchy", "keywords": ["bite", "sweet"]}, {"theme_name": "General comment", "keywords": [], "general": True, "required": True}]
    C = np.array([[1, 0], [0, 1], [0.6, 0.8], [0.5, 0.5]], np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    art = {"segment": clauses, "assign": assign, "themes": {"ALL": (themes, C)}}
    cfg = {"themes": {"spacy_model": "en_core_web_sm", "quality_bar": 0.5}}
    r = quality._scope_report("ALL", clauses, assign, themes, C, cfg, quality.ClauseIndex(clauses["clause"], nlp),
                              {"crunchy": (0.0, 1.0), "packaging": (1.0, 0.0), "price": (1.0, 0.0)})
    s = r["summary"]
    assert s["n_themes"] == 3 and s["n_clauses"] == 10
    assert s["general_share"] == 0.2 and s["unassigned_share"] == 0.0
    assert s["largest_theme"] == "Packaging" and s["largest_share"] == 0.4                   # clause 6 is in two themes
    by = {t["theme"]: t for t in r["themes"]}
    assert by["Packaging"]["name_coverage"] == 0.75 and by["Price"]["name_coverage"] == 1.0
    assert by["Crunchy"]["name_coverage"] == 0.5                                              # "crunchy bite" uses the name, "too sweet" does not
    assert s["name_coverage_min"] == 0.5 and s["n_below_bar"] == 0                           # exactly at the bar is not below it
    assert by["Crunchy"]["adjective_name"] and not by["Price"]["adjective_name"]
    assert s["n_adjective_names"] == 1
    # raising the bar flags the weak name, unless the theme is marked mixed
    cfg["themes"]["quality_bar"] = 0.6
    r2 = quality._scope_report("ALL", clauses, assign, themes, C, cfg, quality.ClauseIndex(clauses["clause"], nlp), {})
    assert r2["summary"]["n_below_bar"] == 1
    themes[2]["name_mixed"] = True
    r3 = quality._scope_report("ALL", clauses, assign, themes, C, cfg, quality.ClauseIndex(clauses["clause"], nlp), {})
    assert r3["summary"]["n_below_bar"] == 0 and r3["summary"]["n_mixed"] == 1
    assert r3["near_duplicates"] == []                                                        # Packaging / Crunchy centres are not close enough
    C[2] = C[0]                                                                                # a third theme on the same centre
    r4 = quality._scope_report("ALL", clauses, assign, themes, C, cfg, quality.ClauseIndex(clauses["clause"], nlp), {})
    assert [(p["a"], p["b"]) for p in r4["near_duplicates"]] == [("Packaging", "Crunchy")]
    line = quality.summary_line({"scopes": {"ALL": r4}})
    assert "3 themes" in line and "1 near-duplicate" in line
