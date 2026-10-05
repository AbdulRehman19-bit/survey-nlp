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


def run(cfg):
    c = cfg["input"]
    p = c["path"]
    if p.lower().endswith((".xlsx", ".xls")):
        hdr = c.get("header_row", 0)
        skip = [hdr + 1 + i for i in range(c.get("skip_rows", 0))]   # junk rows right under the header
        df = pd.read_excel(p, sheet_name=c.get("sheet", 0), header=hdr, skiprows=skip)
    else:
        df = pd.read_csv(p)
    cols = c.get("text_cols", "auto")
    if cols in ("auto", None, []):
        cols = detect_text_cols(df, c)
    missing = [x for x in cols if x not in df.columns]
    if missing:
        raise ValueError(f"text_cols not found in file: {missing}. Available: {list(df.columns)}")
    if not cols:
        raise ValueError("No open-ended columns found. Set input.text_cols explicitly.")
    print("Open-ended columns analysed:", cols)

    ids = df[c["id_col"]].to_numpy() if c.get("id_col") else range(len(df))
    base = pd.DataFrame({"row_id": range(len(df)), "respondent_id": ids})
    for k in c.get("keep_cols", []):
        base[k] = df[k].to_numpy()

    parts = [pd.DataFrame({"row_id": range(len(df)), "respondent_id": ids,
                           "question": q, "text": df[q].to_numpy()}) for q in cols]
    long = pd.concat(parts, ignore_index=True)
    clean = long.text.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
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
