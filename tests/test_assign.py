import numpy as np, pandas as pd
from survey_nlp.assign import run

def test_keyword_is_whole_word():
    clauses = pd.DataFrame({"clause_id": [0, 1], "clause": ["I like tea", "great steak"]})
    E = np.array([[1, 0], [1, 0]], np.float32)
    C = np.array([[0, 1]], np.float32)                      # orthogonal: semantic never fires
    themes = [{"theme_name": "Tea", "keywords": ["tea"], "curated_keywords": ["tea"]}]
    cfg = {"assign": {"min_sim": 0.9, "secondary_margin": 0.03, "keyword_min_len": 3}}
    out = run(clauses, E, themes, C, cfg)
    assert out.clause_id.tolist() == [0]                     # "steak" must not match "tea"


def test_semantic_top1_and_near_tie_top2():
    clauses = pd.DataFrame({"clause_id": [0, 1, 2], "clause": ["a", "b", "c"]})
    E = np.array([[1, 0], [0.7, 0.7], [0.2, 0.1]], np.float32)
    C = np.array([[1, 0], [0, 1]], np.float32)
    themes = [{"theme_name": "A", "keywords": []}, {"theme_name": "B", "keywords": []}]
    cfg = {"assign": {"min_sim": 0.5, "secondary_margin": 0.03, "keyword_min_len": 3}}
    out = run(clauses, E, themes, C, cfg)
    assert out[out.clause_id == 0].theme_idx.tolist() == [0]                 # clear top-1
    assert sorted(out[out.clause_id == 1].theme_idx.tolist()) == [0, 1]      # near tie: both themes
    assert 2 not in out.clause_id.tolist()                                   # below min_sim, no keyword: unassigned


def test_discovered_keywords_do_not_match_unless_enabled_and_regex_keywords_work():
    clauses = pd.DataFrame({"clause_id": [0, 1, 2], "clause": ["milk chocolate bits", "gets soggy in milk", "good and nice"]})
    E = np.array([[1, 0]] * 3, np.float32)
    C = np.array([[0, 1]], np.float32)
    base = {"min_sim": 0.9, "secondary_margin": 0.03, "keyword_min_len": 3}
    discovered = [{"theme_name": "Milk", "keywords": ["good"]}]
    assert run(clauses, E, discovered, C, {"assign": base}).empty                                      # discovered words ignored
    assert run(clauses, E, discovered, C, {"assign": {**base, "use_discovered_keywords": True}}).clause_id.tolist() == [2]
    hand = [{"theme_name": "Milk", "keywords": ["good"], "curated_keywords": ["re:milk(?! chocolate)", "soggy"]}]
    assert run(clauses, E, hand, C, {"assign": base}).clause_id.tolist() == [1]                         # hand-written replace discovered


def _three(general_flag):
    clauses = pd.DataFrame({"clause_id": [0, 1, 2], "clause": ["red box cheap", "xyz", "abc"]})
    E = np.array([[1, 0, 0], [0.3, 0.1, 0], [0, 0, 0.01]], np.float32)
    C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    themes = [{"theme_name": "A", "keywords": []}, {"theme_name": "B", "keywords": []},
              {"theme_name": "G", "keywords": [], **({"general": True} if general_flag else {})}]
    cfg = {"assign": {"min_sim": 0.9, "secondary_margin": 0.03, "keyword_min_len": 3, "fallback_min_sim": 0.2}}
    return clauses, E, C, themes, cfg


def test_fallback_joins_the_closest_theme_when_close_enough():
    out = run(*(lambda c, E, C, t, cfg: (c, E, t, C, cfg))(*_three(False)))
    assert out[out.clause_id == 1].via.tolist() == ["fallback"]            # sim 0.3 >= 0.2
    assert 2 not in out.clause_id.tolist()                                  # nothing close: still unassigned


def test_general_theme_takes_every_clause_nothing_else_matched():
    clauses, E, C, themes, cfg = _three(True)
    out = run(clauses, E, themes, C, cfg)
    assert sorted(out.clause_id.tolist()) == [0, 1, 2]                      # every clause has a theme
    assert out[out.clause_id == 2].theme_idx.tolist() == [2]                # the unmatched one went to the general theme
    assert out[out.clause_id == 1].theme_idx.tolist() == [0]                # a close theme still wins over the general one
    cfg["assign"]["general_catch_all"] = False
    assert 2 not in run(clauses, E, themes, C, cfg).clause_id.tolist()


def test_cap_keeps_the_strongest_themes_and_ranks_the_general_theme_last():
    clauses = pd.DataFrame({"clause_id": [0], "clause": ["x"]})
    E = np.array([[0.6, 0.6, 0.5]], np.float32)
    C = np.eye(3, dtype=np.float32)
    themes = [{"theme_name": "A", "keywords": []}, {"theme_name": "B", "keywords": []}, {"theme_name": "G", "keywords": [], "general": True}]
    cfg = {"assign": {"min_sim": 0.4, "secondary_margin": 0.2, "keyword_min_len": 3, "max_themes_per_clause": 2}}
    assert sorted(run(clauses, E, themes, C, cfg).theme_idx.tolist()) == [0, 1]


def test_general_theme_is_dropped_when_a_specific_theme_took_the_clause():
    clauses = pd.DataFrame({"clause_id": [0, 1], "clause": ["good box", "it is good"]})
    E = np.array([[0.8, 0, 0.8], [0, 0, 1]], np.float32)                       # clause 0 is close to A and to the general theme G
    C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    themes = [{"theme_name": "A", "keywords": []}, {"theme_name": "B", "keywords": []},
              {"theme_name": "G", "keywords": [], "general": True}]
    cfg = {"assign": {"min_sim": 0.5, "secondary_margin": 0.5, "keyword_min_len": 3, "fallback_min_sim": 0.2, "max_themes_per_clause": 2}}
    out = run(clauses, E, themes, C, cfg)
    assert out[out.clause_id == 0].theme_idx.tolist() == [0]                   # A only: G is not added next to a specific theme
    assert out[out.clause_id == 1].theme_idx.tolist() == [2]                   # nothing specific: G


def test_hand_written_keyword_match_is_not_removed_by_the_cap():
    clauses = pd.DataFrame({"clause_id": [0], "clause": ["a refreshing and fruity flavour"]})
    E = np.array([[0.7, 0.7, 0.05]], np.float32)                              # close to A and B, far from R
    C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    themes = [{"theme_name": "A", "keywords": []}, {"theme_name": "B", "keywords": []},
              {"theme_name": "R", "keywords": ["refreshing"], "curated_keywords": ["refreshing"]}]
    cfg = {"assign": {"min_sim": 0.4, "secondary_margin": 0.5, "keyword_min_len": 3, "max_themes_per_clause": 2}}
    out = run(clauses, E, themes, C, cfg)
    assert 2 in out.theme_idx.tolist()                                         # R is kept although A and B are stronger


def test_theme_name_word_matches_by_keyword_but_a_shared_name_word_does_not():
    clauses = pd.DataFrame({"clause_id": [0, 1], "clause": ["it feels refreshing to drink", "a nice taste"]})
    E = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5]], np.float32)                  # below min_sim for every theme: only a keyword can match
    C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    themes = [{"theme_name": "Refreshing Taste", "keywords": [], "assign_keywords": ["refreshing", "taste"]},
              {"theme_name": "Taste", "keywords": [], "assign_keywords": ["taste"]},
              {"theme_name": "Other", "keywords": []}]
    cfg = {"assign": {"min_sim": 0.9, "secondary_margin": 0.03, "keyword_min_len": 3}}
    out = run(clauses, E, themes, C, cfg)
    assert out[out.clause_id == 0].theme_idx.tolist() == [0]                   # "refreshing" is in the name of theme 0
    assert out[out.clause_id == 1].theme_idx.tolist() == [1]                   # "taste" belongs to the theme called Taste, not to Refreshing Taste
