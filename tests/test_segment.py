import pandas as pd
from survey_nlp.config import Cfg
from survey_nlp import segment

CFG = Cfg({"segment": {"contrast_words": ["but", "however", "although", "though", "yet", "whereas"], "min_words": 2}})

def _run(texts):
    u = pd.DataFrame({"text_key": [f"k{i}" for i in range(len(texts))], "text_clean": texts})
    return segment.run(u, CFG)

def test_contrast_split():
    out = _run(["The packaging is attractive but the flavour feels artificial."])
    assert out.clause.tolist() == ["The packaging is attractive", "the flavour feels artificial"]

def test_short_text_kept():
    assert _run(["Good"]).clause.tolist() == ["Good"]
