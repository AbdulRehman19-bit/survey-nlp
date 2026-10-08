from survey_nlp.spelling import apply_corrections, find_corrections

TEXTS = ["The artifical taste is bad", "artifical flavour", "Tastes artificial and sweet", "I dont like the artificial taste",
         "The Cadbury flavour is good", "good colour", "the colour is nice", "Very sweeet", "sweet and nice", "taste is sweet"]


def test_typos_are_repaired_and_names_british_spellings_and_codes_are_not():
    fx = find_corrections(TEXTS + ["Cadbury taste", "flavours are nice", "aaab xzq"], {"spell_min_count": 3})
    assert fx["artifical"] == "artificial" and fx["sweeet"] == "sweet"
    assert fx["dont"] == "don't"                                   # a lost apostrophe is repaired so the negation is seen
    for kept in ("cadbury", "flavours", "colour", "aaab", "xzq"):
        assert kept not in fx                                      # a name, a British spelling, a code: left alone


def test_words_you_use_in_your_themes_and_frequent_words_are_protected():
    assert "sweeet" not in find_corrections(TEXTS, {}, protect=["sweeet"])
    assert "sweeet" not in find_corrections(TEXTS + ["sweeet"] * 5, {"spell_min_count": 3})      # many people wrote it: not a typo


def test_corrections_keep_the_capital_letter_and_other_words():
    assert apply_corrections("Artifical taste, ARTIFICAL, nice", {"artifical": "artificial"}) == "Artificial taste, ARTIFICIAL, nice"
