import json
import time
from pathlib import Path
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from . import sentiment

GREEN, RED, YELLOW = "C6EFCE", "FFC7CE", "FFEB9C"          # positive / negative / neutral cell colours


def _style(ws, widths):
    """Header look, column widths, frozen header, wrapped text."""
    ws.freeze_panes = "A2"
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="305496")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for j, h in enumerate([c.value for c in ws[1]], 1):
        ws.column_dimensions[get_column_letter(j)].width = widths(str(h))
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")


def _colour(ws, colour_of):
    for row in ws.iter_rows(min_row=2):
        for c in row:
            col = colour_of(c.value)
            if col:
                c.fill = PatternFill("solid", fgColor=col)


def run(art, cfg, timings):
    rd = Path(cfg["run_dir"])
    o = cfg["output"]
    codes = o["codes"]
    long, base = art["ingest"]
    T = art["themes"]                                   # scope -> (themes, C)
    g = sentiment.aggregate(art["absa"], cfg)
    L = long.merge(art["overall"], on="text_key", how="left")
    L["overall_code"] = L["label"].map(codes)
    questions = list(long.question.unique())
    multi = len(questions) > 1

    ql = o.get("question_labels") or {}                 # optional short names for the long question texts
    out = base.copy()
    detail_rows = []
    qcols = {}                                          # sheet name -> [(column in `out`, header on the question's sheet)]
    for q in questions:
        Lq = L[L.question == q].set_index("row_id")
        name = ql.get(q, q)
        pre = f"{name} | " if multi else ""
        out[name] = out.row_id.map(Lq["text"])
        qcols[name] = [(name, "Answer")]
        pol = (o.get("question_polarity") or {}).get(name)
        P = codes[pol["neutral_as"]] if pol else None          # what a bare mention means in this question
        if pol:
            # In a "what did you LIKE" question a neutral mention is a positive one, and a weakly-negative
            # overall score is almost always the model tripping on negation ("isn't overly sweet").
            opp = codes["negative"] if P == codes["positive"] else codes["positive"]
            p_opp = Lq["p_neg"] if P == codes["positive"] else Lq["p_pos"]
            oc = Lq["overall_code"].astype(float)
            oc = oc.where(oc != codes["neutral"], P)
            oc = oc.where(~((oc == opp) & (p_opp < pol.get("opposite_min_conf", 0.8))), P)
            Lq = Lq.assign(overall_code=oc.where(Lq["valid"]))
        if o["include_overall"]:
            out[f"{pre}Overall"] = out.row_id.map(Lq["overall_code"]).astype("Int64")
            qcols[name].append((f"{pre}Overall", "Overall"))
        scope = "ALL" if cfg["themes"]["scope"] == "pooled" else q
        themes, _ = T[scope]
        gs = g[g.scope == scope]
        for j, t in enumerate(themes):
            sub = gs[gs.theme_idx == j].set_index("text_key")
            col = Lq["text_key"].map(sub["code"])       # NaN = theme not mentioned
            if pol:
                col = col.where(col != codes["neutral"], P)
            if o["not_discussed_value"] is not None:
                col = col.where(~(col.isna() & Lq["valid"]), o["not_discussed_value"])
            out[f"{pre}{t['theme_name']}"] = out.row_id.map(col).astype("Int64")
            qcols[name].append((f"{pre}{t['theme_name']}", t["theme_name"]))
            d = Lq.reset_index()[["row_id", "respondent_id", "text_key"]].merge(sub.reset_index(), on="text_key")
            d["question"], d["theme"] = name, t["theme_name"]
            d["model_code"] = d["code"]                  # what the model said, before the question-polarity rule
            if pol:
                d["code"] = d["code"].where(d["code"] != codes["neutral"], P)
            detail_rows.append(d[["row_id", "respondent_id", "question", "theme", "p_pos", "p_neg", "p_neu", "mixed", "model_code", "code"]])

    detail = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    pairs = art["absa"][["text_key", "scope", "theme", "clause", "sim", "via", "p_neg", "p_neu", "p_pos"]]
    theme_tbl = pd.concat([pd.DataFrame(t).assign(scope=s) for s, (t, _) in T.items()], ignore_index=True)
    # readable version: Positive / Negative / Neutral instead of the numeric codes (the coded sheet is kept)
    words = out
    if o.get("labels"):
        inv = {codes[k]: v for k, v in o["labels"].items()}
        words = out.copy()
        for col in out.columns:
            if col.endswith("Overall") or col in {t["theme_name"] for ts, _ in T.values() for t in ts}                     or (multi and " | " in col):
                words[col] = out[col].map(lambda v: inv.get(v) if pd.notna(v) else None)
    path = rd / "aspect_sentiment_results.xlsx"
    try:
        open(path, "ab").close()                        # Excel keeps an open workbook locked on Windows
    except PermissionError:
        path = rd / f"aspect_sentiment_results_{time.strftime('%H%M%S')}.xlsx"
        print(f"WARNING: aspect_sentiment_results.xlsx is open in Excel and cannot be replaced. "
              f"Saved to {path.name} instead. Close the old file and run again to refresh it.")
    label = {codes[k]: v for k, v in (o.get("labels") or {}).items()}      # code -> word shown in cells
    shade = {label.get(codes["positive"], codes["positive"]): GREEN,
             label.get(codes["negative"], codes["negative"]): RED,
             label.get(codes["neutral"], codes["neutral"]): YELLOW}
    colour_of = lambda v: shade.get(v) if isinstance(v, (str, int)) and not isinstance(v, bool) else None
    other = [c for c in base.columns if c != "row_id"]                       # respondent id + keep_cols
    reserved = {"results", "results_codes", "detail", "pairs", "themes", "timings"}

    def sheet_name(n):
        n = "".join(ch for ch in n if ch not in '[]:*?/\\')[:31] or "Question"     # characters Excel forbids in sheet names
        return n + " (Q)" if n.lower() in reserved else n

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, cols in qcols.items():                                    # one sheet per open-ended question
            qdf = words[other + [c for c, _ in cols]].copy()
            qdf.columns = other + [h for _, h in cols]
            sn = sheet_name(name)
            qdf.to_excel(xw, sheet_name=sn, index=False)
            ws = xw.sheets[sn]
            _style(ws, lambda h: 60 if h == "Answer" else 14)
            _colour(ws, colour_of)
        words.to_excel(xw, sheet_name="results", index=False)               # every question side by side
        _style(xw.sheets["results"], lambda h: 50 if h in qcols else 16)
        _colour(xw.sheets["results"], colour_of)
        if words is not out:
            out.to_excel(xw, sheet_name="results_codes", index=False)
        detail.to_excel(xw, sheet_name="detail", index=False)
        pairs.to_excel(xw, sheet_name="pairs", index=False)
        theme_tbl.to_excel(xw, sheet_name="themes", index=False)
        pd.DataFrame([{"stage": k, "seconds": v} for k, v in timings.items()]).to_excel(xw, sheet_name="timings", index=False)
    (rd / "discovered_themes.json").write_text(json.dumps({s: t for s, (t, _) in T.items()}, indent=2))
    (rd / "timings.json").write_text(json.dumps(timings, indent=2))
    return str(path)
