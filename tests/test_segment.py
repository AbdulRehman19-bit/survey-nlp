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


KEEP = Cfg({"segment": {"contrast_words": ["but", "however", "although", "though", "yet", "whereas"], "min_words": 2,
                        "keep_short_next_to_contrast": True, "short_not_opinion": ["nothing", "none"]}})


def _run_keep(texts):
    u = pd.DataFrame({"text_key": [f"k{i}" for i in range(len(texts))], "text_clean": texts})
    return segment.run(u, KEEP)


def test_one_word_beside_contrast_kept():
    assert _run_keep(["No smell, but delicious"]).clause.tolist() == ["No smell", "delicious"]
    assert _run(["No smell, but delicious"]).clause.tolist() == ["No smell"]            # off by default


def test_filler_word_beside_contrast_dropped():
    assert _run_keep(["Nothing but does make me want actual chocolate!"]).clause.tolist() == ["does make me want actual chocolate"]


def test_split_at_and():
    cfg = Cfg({"segment": {"contrast_words": ["but"], "min_words": 2, "split_and_min_words": 4}})
    run = lambda t: segment.run(pd.DataFrame({"text_key": ["k"], "text_clean": [t]}), cfg).clause.tolist()
    assert run("Stays crispy for ages in milk and the two chocolates go together brilliantly") == ["Stays crispy for ages in milk", "the two chocolates go together brilliantly"]
    assert run("I like the mix of white and chocolate pieces") == ["I like the mix of white and chocolate pieces"]       # a side is too short: one topic
    assert run("sweet and sour taste") == ["sweet and sour taste"]
