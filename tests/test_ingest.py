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
