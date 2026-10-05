import re
import pandas as pd
from .util import sha


def detect_text_cols(df: pd.DataFrame, c: dict) -> list:
    """Heuristic: string columns with longish, mostly-unique answers are open-ended."""
    d = c.get("auto_detect", {})
    cols = []
    for col in df.columns:
        if col == c.get("id_col"):
            continue
        s = df[col].dropna()
        if s.empty or s.map(lambda x: isinstance(x, str)).mean() < 0.9:
            continue
        if len(s) < d.get("min_fill", 0.2) * len(df):          # mostly empty: "other, please specify" style fields
            continue
        if s.astype(str).str.lstrip().str[:1].isin(["[", "{"]).mean() >= 0.5:   # JSON / structured exports, not prose
            continue
        mean_words = s.astype(str).str.split().str.len().mean()
        uniq = s.nunique() / len(s)
        if mean_words >= d.get("min_mean_words", 4) and uniq >= d.get("min_unique_ratio", 0.3):
            cols.append(col)
    return cols


def guess_header_row(raw: pd.DataFrame, max_rows: int = 15) -> int:
    """The header is the earliest row (of the first few) that holds the most distinct text cells. Exports often start
    with metadata rows (survey id, section names) that have fewer distinct labels than the real column-name row."""
    best, best_n = 0, -1
    for i in range(min(max_rows, len(raw))):
        row = raw.iloc[i].dropna()
        if row.empty or row.map(lambda x: isinstance(x, str)).mean() < 0.8:
            continue
        n = row.astype(str).str.strip().nunique()
        if n > best_n:
            best, best_n = i, n
    return best


_QUESTION_NAME = r"source|question|type|categor|sentiment|label|like|polarity|group|field|topic|section|prompt"


def detect_layout(df: pd.DataFrame, c: dict) -> dict:
    """Is each row one respondent with a column per question ("wide"), or one comment per row with a column saying which
    question it answers ("long")? Long means: exactly one open-ended text column and one short, repeated, label-like column."""
    text_cols = detect_text_cols(df, c)
    cats = []
    for col in df.columns:
        if col in text_cols:
            continue
        s = df[col].dropna()
        if len(s) < 0.8 * len(df) or s.map(lambda x: isinstance(x, str)).mean() < 0.9:
            continue
        counts = s.astype(str).value_counts()
        if 2 <= len(counts) <= 12 and counts.min() >= max(3, 0.02 * len(s)) and s.astype(str).str.split().str.len().max() <= 6:
            cats.append(col)
    if len(text_cols) == 1 and cats:
        hinted = [x for x in cats if re.search(_QUESTION_NAME, str(x), re.I)]
        pick = hinted[0] if hinted else (cats[0] if len(df.columns) == 2 else None)
        if pick is not None:
            return {"layout": "long", "question_col": pick, "text_col": text_cols[0], "text_cols": []}
    return {"layout": "wide", "question_col": None, "text_col": None, "text_cols": text_cols}


def read_table(c: dict) -> pd.DataFrame:
    """Read the file. input.header_row may be a number or "auto"."""
    p = c["path"]
    if not p.lower().endswith((".xlsx", ".xls")):
        return pd.read_csv(p)
    hdr = c.get("header_row", 0)
    if hdr == "auto":
        hdr = guess_header_row(pd.read_excel(p, sheet_name=c.get("sheet", 0), header=None, nrows=15))
    skip = [hdr + 1 + i for i in range(c.get("skip_rows", 0))]   # junk rows right under the header
    return pd.read_excel(p, sheet_name=c.get("sheet", 0), header=hdr, skiprows=skip)


def _drop_id_row(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    """Some exports put a row of question IDs (one token, with digits: "q-id-6abb…-value") right under the header."""
    if df.empty or not text_cols:
        return df
    first = [df.iloc[0][k] for k in text_cols if pd.notna(df.iloc[0][k])]
    if first and all(isinstance(v, str) and len(v.split()) == 1 and len(v) >= 8 and any(ch.isdigit() for ch in v) for v in first):
        return df.iloc[1:].reset_index(drop=True)
    return df


def run(cfg):
    c = dict(cfg["input"])
    df = read_table(c)
    layout = c.get("layout", "auto")
    explicit = bool(c.get("question_col") and c.get("text_col")) or c.get("text_cols") not in ("auto", None, [])
    if layout == "auto" and not explicit:
        found = detect_layout(df, c)
        print(f"Detected layout: {found['layout']}" +
              (f" (questions in '{found['question_col']}', text in '{found['text_col']}')" if found["layout"] == "long"
               else f" (open-ended columns: {found['text_cols']})"))
        c.update({k: v for k, v in found.items() if k != "layout"})
    elif layout == "wide":
        c["question_col"] = c["text_col"] = None
    qcol, tcol = c.get("question_col"), c.get("text_col")
    long_format = bool(qcol and tcol)           # file is already "one row per comment": a question column + a text column
    if long_format:
        missing = [x for x in (qcol, tcol) if x not in df.columns]
        if missing:
            raise ValueError(f"question_col/text_col not found in file: {missing}. Available: {list(df.columns)}")
        print(f"Long-format input: questions in '{qcol}' = {list(dict.fromkeys(df[qcol].dropna().astype(str)))}, text in '{tcol}'")
    else:
        cols = c.get("text_cols", "auto")
        if cols in ("auto", None, []):
            cols = detect_text_cols(df, c)
        missing = [x for x in cols if x not in df.columns]
        if missing:
            raise ValueError(f"text_cols not found in file: {missing}. Available: {list(df.columns)}")
        if not cols:
            raise ValueError("No open-ended columns found. Set input.text_cols explicitly.")
        print("Open-ended columns analysed:", cols)

    df = _drop_id_row(df, [tcol] if long_format else cols)
    ids = df[c["id_col"]].to_numpy() if c.get("id_col") else range(len(df))
    base = pd.DataFrame({"row_id": range(len(df)), "respondent_id": ids})
    for k in c.get("keep_cols", []):
        base[k] = df[k].to_numpy()

    if long_format:                              # every row is one comment under one question
        long = pd.DataFrame({"row_id": range(len(df)), "respondent_id": ids,
                             "question": df[qcol].astype(str).to_numpy(), "text": df[tcol].to_numpy()})
    else:
        parts = [pd.DataFrame({"row_id": range(len(df)), "respondent_id": ids,
                               "question": q, "text": df[q].to_numpy()}) for q in cols]
        long = pd.concat(parts, ignore_index=True)
    # An apostrophe lost to a bad text encoding in the export shows up as U+FFFD ("don?t"). Restore it so words stay whole.
    clean = (long.text.fillna("").astype(str).str.replace("�", "'", regex=False)
             .str.replace(r"\s+", " ", regex=True).str.strip())
    valid = clean.str.len() >= c["min_chars"]
    if c.get("non_answer_regex"):
        valid &= ~clean.str.lower().str.fullmatch(c["non_answer_regex"])
    ex = c.get("non_answer_examples")
    if ex and valid.any():              # meaning-based non-answers ("I didn't find any negative"): no regex needed
        from . import embed
        texts = clean[valid].drop_duplicates().tolist()
        V = embed.encode_cached([sha(t.lower()) for t in texts], texts, cfg)
        P = embed.encode_cached([sha(e.lower()) for e in ex], list(ex), cfg)
        na = dict(zip(texts, (V @ P.T).max(1) >= c["non_answer_min_sim"]))
        valid &= ~clean.map(na).fillna(False).astype(bool)
    long["text_clean"] = clean
    long["valid"] = valid
    long["text_key"] = [sha(t.lower()) if v else "" for t, v in zip(clean, valid)]
    return long, base
