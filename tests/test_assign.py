import numpy as np, pandas as pd
from survey_nlp.assign import run

def test_keyword_is_whole_word():
    clauses = pd.DataFrame({"clause_id": [0, 1], "clause": ["I like tea", "great steak"]})
    E = np.array([[1, 0], [1, 0]], np.float32)
    C = np.array([[0, 1]], np.float32)                      # orthogonal: semantic never fires
    themes = [{"theme_name": "Tea", "keywords": ["tea"]}]
    cfg = {"assign": {"min_sim": 0.9, "secondary_margin": 0.03, "keyword_min_len": 3}}
    out = run(clauses, E, themes, C, cfg)
    assert out.clause_id.tolist() == [0]                     # "steak" must not match "tea"


def test_curated_keywords_replace_discovered_and_support_regex():
    clauses = pd.DataFrame({"clause_id": [0, 1, 2], "clause": ["milk chocolate bits", "gets soggy in milk", "good and nice"]})
    E = np.array([[1, 0]] * 3, np.float32)
    C = np.array([[0, 1]], np.float32)
    themes = [{"theme_name": "Milk", "keywords": ["good"], "curated_keywords": ["re:milk(?! chocolate)", "soggy"]}]
    cfg = {"assign": {"min_sim": 0.9, "secondary_margin": 0.03, "keyword_min_len": 3}}
    assert run(clauses, E, themes, C, cfg).clause_id.tolist() == [1]      # "good" ignored, "milk chocolate" excluded
