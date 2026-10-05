from survey_nlp import cli

NAMES = ["Flavour", "Easy", "Bottle", "Cough"]


def test_commands_edit_the_choices():
    e = cli._empty()
    assert cli._apply("n 8", e, NAMES) is None and e["n_themes"] == 8
    assert cli._apply("add Price: cost, cheap", e, NAMES) is None
    assert e["required"] == [{"name": "Price", "keywords": ["cost", "cheap"]}]
    assert cli._apply("merge easy + Bottle = Opening", e, NAMES) is None
    assert cli._apply("remove cough", e, NAMES) is None
    assert cli._apply("remove Price", e, NAMES) is None and e["required"] == []     # removing an added theme un-adds it
    assert e["ops"] == [["merge", "Opening", ["Easy", "Bottle"]], ["drop", "Cough"]]


def test_bad_commands_are_explained_and_change_nothing():
    e = cli._empty()
    assert "Current themes" in cli._apply("remove Nope", e, NAMES)
    assert cli._apply("merge Easy", e, NAMES) is not None
    assert cli._apply("n 1", e, NAMES) is not None
    assert e == cli._empty()
