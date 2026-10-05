import pandas as pd
from survey_nlp.sentiment import decide, aggregate

CODES = {"neutral": 0, "positive": 1, "negative": 2}

def test_mixed_fires_both():
    pos, neg, neu = decide(pd.Series([0.92]), pd.Series([0.90]), 0.60, 0.05)
    assert (pos.iloc[0], neg.iloc[0], neu.iloc[0]) == (1, 1, 0)

def test_neutral_when_neither():
    pos, neg, neu = decide(pd.Series([0.2]), pd.Series([0.1]), 0.60, 0.05)
    assert (pos.iloc[0], neg.iloc[0], neu.iloc[0]) == (0, 0, 1)

def _scored(pp, pn):
    return pd.DataFrame({"text_key": ["a"], "scope": ["ALL"], "theme_idx": [0],
                         "p_pos": [pp], "p_neg": [pn], "p_neu": [0.0]})

def _cfg(policy):
    return {"absa": {"evidence_threshold": 0.6, "allowance": 0.05},
            "output": {"codes": CODES, "mixed_policy": policy}}

def test_codes():
    assert aggregate(_scored(0.9, 0.05), _cfg("stronger")).code.iloc[0] == 1     # positive
    assert aggregate(_scored(0.05, 0.9), _cfg("stronger")).code.iloc[0] == 2     # negative
    assert aggregate(_scored(0.2, 0.1), _cfg("stronger")).code.iloc[0] == 0      # neutral

def test_mixed_policy():
    assert aggregate(_scored(0.92, 0.90), _cfg("stronger")).code.iloc[0] == 1    # 0.92 > 0.90
    assert aggregate(_scored(0.92, 0.90), _cfg("negative")).code.iloc[0] == 2
    assert aggregate(_scored(0.92, 0.90), _cfg("neutral")).code.iloc[0] == 0


def test_infer_polarity_from_question_name():
    from survey_nlp.export import infer_polarity
    assert infer_polarity("Like")["neutral_as"] == "positive"
    assert infer_polarity("What did you LIKE about it?")["neutral_as"] == "positive"
    assert infer_polarity("Dislike")["neutral_as"] == "negative"
    assert infer_polarity("What did you NOT LIKE?")["neutral_as"] == "negative"
    assert infer_polarity("Any other comments") is None


def test_short_names_for_long_questions():
    from survey_nlp.export import short_names
    n = short_names(["What did you LIKE about this product overall? - value", "What did you NOT LIKE about this product overall? - value", "Like"])
    assert n["What did you LIKE about this product overall? - value"] == "Likes"
    assert n["What did you NOT LIKE about this product overall? - value"] == "Dislikes"
    assert n["Like"] == "Like"


def _export_themes(pol, p_pos, p_neg, p_neu=0.0):
    import tempfile, pathlib
    import numpy as np
    from survey_nlp import export
    long = pd.DataFrame({"row_id": [0], "respondent_id": [0], "question": ["Q"], "text": ["not too sweet"],
                         "text_clean": ["not too sweet"], "valid": [True], "text_key": ["a"]})
    base = pd.DataFrame({"row_id": [0], "respondent_id": [0]})
    ov = pd.DataFrame({"text_key": ["a"], "p_neg": [0.1], "p_neu": [0.1], "p_pos": [0.8], "label": ["positive"], "confidence": [0.8]})
    scored = pd.DataFrame({"text_key": ["a"], "scope": ["ALL"], "theme_idx": [0], "clause_key": ["x"], "clause": ["c"], "theme": ["T"],
                           "sim": [0.5], "via": ["both"], "p_pos": [p_pos], "p_neg": [p_neg], "p_neu": [p_neu]})
    d = pathlib.Path(tempfile.mkdtemp())
    cfg = {"run_dir": str(d), "themes": {"scope": "pooled"}, "absa": {"evidence_threshold": 0.5, "allowance": 0.05},
           "output": {"codes": CODES, "mixed_policy": "stronger", "include_overall": True, "not_discussed_value": None,
                      "question_polarity": {"Q": pol}}}
    art = {"ingest": (long, base), "themes": {"ALL": ([{"theme_name": "T", "keywords": ["t"], "frequency": 1}], np.zeros((1, 2)))},
           "absa": scored, "overall": ov}
    export.run(art, type("C", (), {"__getitem__": lambda s, k: cfg[k]})(), {})
    return pd.read_excel(d / "aspect_sentiment_results.xlsx", sheet_name="results")["T"].iloc[0]


def test_weak_opposite_theme_rating_follows_the_question_but_strong_one_stays():
    like = {"neutral_as": "positive", "themes_opposite_min_conf": 0.7}
    assert _export_themes(like, p_pos=0.05, p_neg=0.62) == 1       # weak negative in a Like question -> Positive
    assert _export_themes(like, p_pos=0.01, p_neg=0.97) == 2       # confident negative stays Negative
    dislike = {"neutral_as": "negative", "themes_opposite_min_conf": 0.9}
    assert _export_themes(dislike, p_pos=0.85, p_neg=0.05) == 2    # positive in a Dislike question -> Negative
