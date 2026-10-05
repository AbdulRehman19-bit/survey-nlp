import pandas as pd
from survey_nlp import ingest

def test_detect_picks_open_ended_only():
    df = pd.DataFrame({
        "id": range(40),
        "gender": ["M", "F"] * 20,
        "likes": [f"I really like the taste number {i} a lot" for i in range(40)],
    })
    cols = ingest.detect_text_cols(df, {"id_col": "id", "auto_detect": {"min_mean_words": 4, "min_unique_ratio": 0.3}})
    assert cols == ["likes"]


def test_detect_skips_json_and_sparse_columns():
    n = 40
    df = pd.DataFrame({
        "likes": [f"I really like the taste number {i} a lot" for i in range(n)],
        "pairs": ['[{"winner":"Light","loser":"Milky"}]' + str(i) for i in range(n)],
        "other": [f"some long other text here {i}" if i < 2 else None for i in range(n)],
    })
    cols = ingest.detect_text_cols(df, {"auto_detect": {"min_mean_words": 4, "min_unique_ratio": 0.3}})
    assert cols == ["likes"]


def test_layout_long_vs_wide():
    n = 60
    text = [f"this is comment number {i} about the product" for i in range(n)]
    long_df = pd.DataFrame({"Source": ["Like", "Dislike"] * (n // 2), "Text": text})
    assert ingest.detect_layout(long_df, {})["layout"] == "long"
    assert ingest.detect_layout(long_df, {})["question_col"] == "Source"
    # one open-ended column next to a demographic: stays wide (the gender column is not a question name)
    wide_df = pd.DataFrame({"id": range(n), "gender": ["M", "F"] * (n // 2), "age": ["18-24", "25-34", "35-44"] * (n // 3),
                            "likes": text})
    assert ingest.detect_layout(wide_df, {})["layout"] == "wide"
    two_q = pd.DataFrame({"likes": text, "dislikes": [f"I did not enjoy aspect number {i} at all" for i in range(n)]})
    assert ingest.detect_layout(two_q, {})["text_cols"] == ["likes", "dislikes"]


def test_header_row_is_the_row_with_most_distinct_labels():
    raw = pd.DataFrame([["survey_id", "abc", None, None], ["Section A", "Section A", "Section B", "Section B"],
                        ["Age", "City", "Likes", "Dislikes"], ["25", "Leeds", "taste is good", "too sweet"]])
    assert ingest.guess_header_row(raw) == 2
    plain = pd.DataFrame([["Source", "Text"], ["Like", "tastes nice"], ["Dislike", "too sweet"]])
    assert ingest.guess_header_row(plain) == 0


def test_question_id_row_under_the_header_is_dropped():
    df = pd.DataFrame({"likes": ["q-id-6abbd3547745bd2c18acca88-value", "tastes great", "too sweet for me"]})
    assert ingest._drop_id_row(df, ["likes"]).likes.tolist() == ["tastes great", "too sweet for me"]
    ok = pd.DataFrame({"likes": ["tastes great", "too sweet for me"]})
    assert len(ingest._drop_id_row(ok, ["likes"])) == 2
