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
