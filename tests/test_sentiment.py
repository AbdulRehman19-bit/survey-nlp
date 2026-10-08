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


def test_infer_polarity_from_the_answers_not_the_question_wording():
    from survey_nlp.export import infer_polarity
    codes = {"neutral": 0, "positive": 1, "negative": 2}
    assert infer_polarity([1] * 18 + [2, 0], codes)["neutral_as"] == "positive"
    assert infer_polarity([2] * 18 + [1, 0], codes)["neutral_as"] == "negative"
    assert infer_polarity([1] * 10 + [2] * 8 + [0] * 2, codes) is None          # mixed answers: no polarity
    assert infer_polarity([1] * 3, codes) is None                               # too few answers to tell


def test_short_names_for_long_questions():
    from survey_nlp.export import short_names
    a, b = "What did you LIKE about this product overall? - value", "What did you NOT LIKE about this product overall? - value"
    n = short_names([a, b, "Like"])
    assert n[a] == "LIKE" and n[b] == "NOT LIKE" and n["Like"] == "Like"


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


def test_evaluative_flags_find_only_the_learned_words():
    from survey_nlp.sentiment import evaluative
    pos, neg = evaluative(pd.Series(["Smell bad", "It tastes a bit sweet", "love it"]), {"bad": -1, "love": 1})
    assert neg.tolist() == [True, False, False] and pos.tolist() == [False, False, True]


V = {"artificial": -1, "bad": -1, "good": 1, "tasty": 1}


def test_negated_negative_word_is_positive_and_negated_positive_is_negative():
    from survey_nlp.sentiment import evaluative
    t = ["not so artificial, like the mix feeling", "It wasn't as artificial as other sport drinks", "it was not too artificial",
         "no artificial taste", "zero artificiality", "not bad", "It isn\u2019t good", "It tastes artificial", "no doubt it is good",
         "not sweet, artificial"]
    pos, neg = evaluative(t, V)
    assert pos.tolist() == [True, True, True, True, True, True, False, False, True, False]
    assert neg.tolist() == [False, False, False, False, False, False, True, True, False, True]


def test_less_and_lack_turn_a_word_into_its_opposite():
    from survey_nlp.sentiment import evaluative
    pos, neg = evaluative(["It feels less good than others", "It is less artificial than others", "It lacks good flavour", "It is good"], V)
    assert neg.tolist() == [True, False, True, False]                    # "less good", "lacks good": a shortfall
    assert pos.tolist() == [False, True, False, True]                    # "less artificial": better than the rest


def test_negation_reach_stops_at_a_comma_or_a_long_gap():
    from survey_nlp.sentiment import evaluative
    pos, neg = evaluative(["I don't like that it's artificial", "not tasty and artificial"], V)
    assert neg.tolist() == [True, True]                                  # "artificial" is still a plain complaint there


def test_lexical_fix_corrects_the_model_only_when_a_negation_contradicts_it():
    from survey_nlp.sentiment import lexical_flags, lexical_fix
    t = ["not so artificial", "It tastes artificial", "not good", "not artificial but really bad"]
    code, hit = lexical_fix([2, 2, 1, 1], lexical_flags(t, V), CODES)
    assert code.tolist() == [1, 2, 2, 1] and hit.tolist() == [True, False, True, False]


def test_aggregate_applies_the_negation_fix():
    s = _scored(0.02, 0.96).assign(clause=["not so artificial"])         # the model said Negative for aspect "Artificial"
    g = aggregate(s, _cfg("stronger"), V)
    assert g.code.iloc[0] == 1 and bool(g.lex_fix.iloc[0])


def test_firm_text_skips_bare_mentions_and_mild_remarks():
    from survey_nlp.sentiment import firm_text
    t = ["Its taste", "It tastes a bit sweet", "It is a bit too sweet", "It is too sweet not fizzy enough", "Expensive but so watery"]
    assert firm_text(t).tolist() == [False, False, True, True, True]


def test_firm_negative_clause_is_flagged_for_like_questions():
    s = _scored(0.01, 0.98).assign(clause=["It is too sweet not fizzy enough"])
    assert bool(aggregate(s, _cfg("stronger")).firm_neg.iloc[0])
    assert not bool(aggregate(_scored(0.01, 0.98).assign(clause=["Its taste"]), _cfg("stronger")).firm_neg.iloc[0])


def test_like_question_keeps_a_firm_complaint_but_a_dislike_question_does_not_flip_praise():
    import tempfile, pathlib
    import numpy as np
    from survey_nlp import export
    def run(pol, text, p_pos, p_neg):
        long = pd.DataFrame({"row_id": [0], "respondent_id": [0], "question": ["Q"], "text": [text], "text_clean": [text], "valid": [True], "text_key": ["a"]})
        ov = pd.DataFrame({"text_key": ["a"], "p_neg": [0.1], "p_neu": [0.1], "p_pos": [0.8], "label": ["positive"], "confidence": [0.8]})
        sc = pd.DataFrame({"text_key": ["a"], "scope": ["ALL"], "theme_idx": [0], "clause_key": ["x"], "clause": [text], "theme": ["T"],
                           "sim": [0.5], "via": ["both"], "p_pos": [p_pos], "p_neg": [p_neg], "p_neu": [0.0]})
        d = pathlib.Path(tempfile.mkdtemp())
        cfg = {"run_dir": str(d), "themes": {"scope": "pooled"}, "absa": {"evidence_threshold": 0.5, "allowance": 0.05},
               "output": {"codes": CODES, "mixed_policy": "stronger", "include_overall": True, "not_discussed_value": None, "question_polarity": {"Q": pol}}}
        art = {"ingest": (long, pd.DataFrame({"row_id": [0], "respondent_id": [0]})),
               "themes": {"ALL": ([{"theme_name": "T", "keywords": ["t"], "frequency": 1}], np.zeros((1, 2)))}, "absa": sc, "overall": ov}
        export.run(art, type("C", (), {"__getitem__": lambda s, k: cfg[k]})(), {})
        return pd.read_excel(d / "aspect_sentiment_results.xlsx", sheet_name="results")["T"].iloc[0]
    like = {"neutral_as": "positive", "themes_follow_question": True}
    assert run(like, "It is too sweet not fizzy enough", 0.01, 0.98) == 2          # firm complaint in a Like question
    assert run(like, "Its taste", 0.01, 0.98) == 1                                 # bare mention stays praise
    dislike = {"neutral_as": "negative", "themes_follow_question": True}
    assert run(dislike, "It is not something I enjoy at all", 0.98, 0.01) == 2     # model error on negation, stays a complaint


def test_negation_fix_wins_over_the_firm_complaint_rule():
    s = _scored(0.01, 0.98).assign(clause=["not so artificial, like the mix feeling"])
    g = aggregate(s, _cfg("stronger"), V)
    assert g.code.iloc[0] == 1 and not bool(g.firm_neg.iloc[0])


def test_spec_mode_skips_the_later_rules():
    s = _scored(0.02, 0.96).assign(clause=["not so artificial, like the mix feeling"])
    spec = dict(_cfg("stronger"), mode="spec")
    g = aggregate(s, spec, V)                                    # even if a word list were passed, the plain rule decides
    assert g.code.iloc[0] == 2 and not bool(g.firm_neg.iloc[0])  # model said Negative: spec keeps it
    assert aggregate(s, _cfg("stronger"), V).code.iloc[0] == 1   # without mode the negation fix still applies


NEUTRAL = [r"\b(?:ok|okay)\b", r"\bnothing special\b", r"\bpleasant enough\b"]


def test_lukewarm_needs_a_phrase_and_no_clearly_evaluative_word_left():
    from survey_nlp.sentiment import lukewarm
    t = ["It is okay", "Pleasant enough", "Okay but the smell is awful", "I love it", "It tastes of orange"]
    assert lukewarm(t, NEUTRAL, {"awful": -1, "pleasant": 1, "love": 1}).tolist() == [True, True, False, False, False]
    assert lukewarm(t, [], {}).tolist() == [False] * 5                         # no list in the config: the rule is off


def test_lukewarm_blockers_stop_a_complaint_that_contains_a_lukewarm_word():
    from survey_nlp.sentiment import lukewarm
    t = ["Okay taste, too much fizz", "Okay taste"]
    assert lukewarm(t, NEUTRAL, {}, [r"\btoo\b"]).tolist() == [False, True]


def test_lukewarm_answers_are_neutral_in_overall_and_themes_but_other_answers_keep_the_question_polarity():
    import tempfile, pathlib
    import numpy as np
    from survey_nlp import export
    def run(text):
        long = pd.DataFrame({"row_id": [0], "respondent_id": [0], "question": ["Q"], "text": [text], "text_clean": [text], "valid": [True], "text_key": ["a"]})
        ov = pd.DataFrame({"text_key": ["a"], "p_neg": [0.1], "p_neu": [0.1], "p_pos": [0.8], "label": ["positive"], "confidence": [0.8]})
        sc = pd.DataFrame({"text_key": ["a"], "scope": ["ALL"], "theme_idx": [0], "clause_key": ["x"], "clause": [text], "theme": ["T"],
                           "sim": [0.5], "via": ["both"], "p_pos": [0.9], "p_neg": [0.05], "p_neu": [0.0]})
        d = pathlib.Path(tempfile.mkdtemp())
        cfg = {"run_dir": str(d), "themes": {"scope": "pooled"}, "absa": {"evidence_threshold": 0.5, "allowance": 0.05},
               "output": {"codes": CODES, "mixed_policy": "stronger", "include_overall": True, "not_discussed_value": None,
                          "neutral_phrases": NEUTRAL, "question_polarity": {"Q": {"neutral_as": "positive", "themes_follow_question": True}}}}
        art = {"ingest": (long, pd.DataFrame({"row_id": [0], "respondent_id": [0]})),
               "themes": {"ALL": ([{"theme_name": "T", "keywords": ["t"], "frequency": 1}], np.zeros((1, 2)))}, "absa": sc, "overall": ov}
        export.run(art, type("C", (), {"__getitem__": lambda s, k: cfg[k]})(), {})
        r = pd.read_excel(d / "aspect_sentiment_results.xlsx", sheet_name="results")
        return r["Overall"].iloc[0], r["T"].iloc[0]
    assert run("The sweetness is okay") == (0, 0)
    assert run("The sweetness is great") == (1, 1)


def test_infer_polarity_ignores_neutral_answers_when_the_rest_clearly_take_a_side():
    from survey_nlp.export import infer_polarity
    codes = {"neutral": 0, "positive": 1, "negative": 2}
    assert infer_polarity([2] * 8 + [0] * 10 + [1] * 2, codes)["neutral_as"] == "negative"      # complaints the model called Neutral
    assert infer_polarity([0] * 18 + [2, 1], codes) is None                                      # almost nobody took a side


def test_infer_polarity_falls_back_to_the_wording_of_the_question():
    from survey_nlp.export import infer_polarity
    codes = {"neutral": 0, "positive": 1, "negative": 2}
    mixed = [1] * 10 + [2] * 8 + [0] * 2
    assert infer_polarity(mixed, codes, name="What did you NOT LIKE about it?")["neutral_as"] == "negative"
    assert infer_polarity(mixed, codes, name="Dislikes")["neutral_as"] == "negative"
    assert infer_polarity(mixed, codes, name="Likes")["neutral_as"] == "positive"
    assert infer_polarity(mixed, codes, name="Comments") is None
