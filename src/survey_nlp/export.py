import json
import re
import time
from pathlib import Path
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from . import sentiment
from .util import is_spec

COMBINED = "Combined"                                      # sheet with every question's answers one under the other
GREEN, RED, YELLOW ="C6EFCE", "FFC7CE", "FFEB9C"          # positive / negative / neutral cell colours


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


_ASKS_COMPLAINT = re.compile(r"\b(?:not like|dislike[ds]?|didn.?t like|don.?t like|negative|worst|improve|complain\w*|wrong|problems?|issues?|hate[ds]?)\b", re.I)
_ASKS_PRAISE = re.compile(r"\b(?:like[ds]?|enjoy\w*|love[ds]?|best|positive|favou?rite|good)\b", re.I)


def infer_polarity(codes_of_answers, codes, min_share=0.6, ratio=3.0, name=None, min_signal=0.3):
    """What does this question ask for? Read it from the answers: among the answers that took a side, if praise outweighs complaints
    by `ratio` the question asks for praise, if complaints outweigh praise it asks for complaints. Models often call a complaint
    Neutral ("a bit too sweet"), so the Neutral ones are left out of the comparison, but at least `min_signal` of all answers must
    have taken a side. When the answers do not decide, the question's own wording is used ("What did you NOT LIKE", "Dislikes").
    Mixed answers and no wording clue give no polarity, and the sentiment model decides every rating."""
    v = pd.Series(codes_of_answers).dropna()
    if len(v) < 10:
        return None
    pos, neg = (v == codes["positive"]).mean(), (v == codes["negative"]).mean()
    if pos + neg >= min_signal:
        if pos >= ratio * neg and pos >= min_share * (pos + neg):
            return {"neutral_as": "positive", "overall": "short", "themes_follow_question": True}
        if neg >= ratio * pos and neg >= min_share * (pos + neg):
            return {"neutral_as": "negative", "overall": "short", "themes_follow_question": True}
    if name:
        if _ASKS_COMPLAINT.search(str(name)):
            return {"neutral_as": "negative", "overall": "short", "themes_follow_question": True}
        if _ASKS_PRAISE.search(str(name)):
            return {"neutral_as": "positive", "overall": "short", "themes_follow_question": True}
    return None


def short_names(questions, limit=28):
    """Readable sheet/column names for long question texts: the words the questions do not share
    ("What did you LIKE about it" / "What did you NOT LIKE about it" -> "LIKE" / "NOT LIKE")."""
    out, used = {}, set()
    for q in questions:
        if len(q) <= limit:
            out[q] = q
            used.add(q)
    words = {q: q.split() for q in questions}
    pre = suf = 0
    long_q = [q for q in questions if q not in out]
    if len(long_q) > 1:
        ws = [words[q] for q in long_q]
        m = min(map(len, ws))
        while pre < m and len({w[pre] for w in ws}) == 1:
            pre += 1
        while suf < m - pre - 1 and len({w[len(w) - 1 - suf] for w in ws}) == 1:
            suf += 1
    for q in questions:
        if q in out:
            continue
        w = words[q]
        core = " ".join(w[pre:len(w) - suf if suf else len(w)]).strip(" ?-:")
        base = (core or q)[:limit].strip()
        name, k = base, 2
        while name in used:
            name, k = f"{base[:limit - 3].strip()} {k}", k + 1
        out[q] = name
        used.add(name)
    return out


def run(art, cfg, timings):
    rd = Path(cfg["run_dir"])
    o = cfg["output"]
    codes = o["codes"]
    long, base = art["ingest"]
    T = art["themes"]                                   # scope -> (themes, C)
    spec = is_spec(cfg)                                  # spec mode: ratings are the model's, with no later rules on top
    valence = None if spec else sentiment.word_valence(art["absa"], long[long.valid][["text_key", "question"]], cfg,
                                                       min_conf=o.get("evaluative_min_conf", 0.65))   # clearly evaluative words
    g = sentiment.aggregate(art["absa"], cfg, valence)
    L = long.merge(art["overall"], on="text_key", how="left")
    L["overall_code"] = L["label"].map(codes)
    if not spec:
        fixed, hit = sentiment.lexical_fix(L["overall_code"], sentiment.lexical_flags(L["text_clean"], valence), codes)
        L["overall_code"] = pd.Series(fixed, index=L.index).where(L["valid"] & (L["overall_code"].notna() | hit))   # "not so artificial" is not Negative
    questions = list(long.question.unique())
    multi = len(questions) > 1

    ql = dict(o.get("question_labels") or {})           # optional short names for the long question texts
    ql.update({q: n for q, n in short_names(questions).items() if q not in ql})
    out = base.copy()
    detail_rows = []
    qcols = {}                                          # sheet name -> [(column in `out`, header on the question's sheet)]
    for q in questions:
        Lq = L[L.question == q].set_index("row_id")
        name = ql.get(q, q)
        pre = f"{name} | " if multi else ""
        out[name] = out.row_id.map(Lq["text"])
        qcols[name] = [(name, "Answer")]
        pol = (o.get("question_polarity") or {}).get(name) or (
            infer_polarity(Lq["overall_code"].where(Lq["valid"]), codes, o.get("polarity_min_share", 0.6),
                           o.get("polarity_ratio", 3.0), name=name) if o.get("auto_polarity") else None)
        if spec:
            pol = None
        if pol and pol.get("off"):                              # the user said this question has no polarity
            pol = None
        P = codes[pol["neutral_as"]] if pol else None          # what a bare mention means in this question
        if pol:
            # In a "what did you LIKE" question merely mentioning a theme is a positive act, and in a "what did you NOT like"
            # question it is a complaint, so a neutral aspect rating takes the question's polarity.
            oc = Lq["overall_code"].astype(float)
            mode = pol.get("overall", "all")                    # whole-answer rating: all | short (bare mentions only) | none
            if mode != "none":
                bare = Lq["text_clean"].str.split().str.len() <= pol.get("short_words", 4)
                oc = oc.where(~((oc == codes["neutral"]) & (bare if mode == "short" else True)), P)
            omin = pol.get("opposite_min_conf")                 # optional: a weak rating AGAINST the question's polarity is
            if omin is not None:                                # usually the model tripping on negation ("isn't overly sweet")
                opp = codes["negative"] if P == codes["positive"] else codes["positive"]
                p_opp = Lq["p_neg"] if P == codes["positive"] else Lq["p_pos"]
                oc = oc.where(~((oc == opp) & (p_opp < omin)), P)
            if pol.get("themes_follow_question"):
                # The whole answer is praise in a "like" question and a complaint in a "dislike" one, unless the answer itself
                # says the opposite with a clearly evaluative word ("smell bad" in a "like" question).
                opp = codes["negative"] if P == codes["positive"] else codes["positive"]
                ev_p, ev_n = sentiment.evaluative(Lq["text_clean"], valence)
                real = (Lq["overall_code"] == opp) & (ev_n if opp == codes["negative"] else ev_p)
                fc = pol.get("firm_negative_conf", o.get("firm_negative_conf", 0.95))       # a long, clearly negative answer in a "like" question is a complaint
                if P == codes["positive"] and fc:
                    real |= (Lq["p_neg"] >= fc) & sentiment.firm_text(Lq["text_clean"])
                if P == codes["positive"] and o.get("complaint_phrases"):
                    real |= sentiment.phrase_flag(Lq["text_clean"], o["complaint_phrases"]).astype(bool)
                oc = pd.Series(float(P), index=oc.index).where(~real, float(opp))
            Lq = Lq.assign(overall_code=oc.where(Lq["valid"]))
        lw = pd.Series(False, index=Lq.index)
        if not spec and o.get("neutral_phrases"):          # lukewarm answers ("okay", "nothing special") are Neutral, whatever the question
            lw = sentiment.lukewarm(Lq["text_clean"], o["neutral_phrases"], valence, o.get("neutral_blockers")).astype(bool)
            Lq = Lq.assign(overall_code=Lq["overall_code"].astype(float).where(~lw, float(codes["neutral"])))
        if o["include_overall"]:
            out[f"{pre}Overall"] = out.row_id.map(Lq["overall_code"]).astype("Int64")
            qcols[name].append((f"{pre}Overall", "Overall"))
        scope = "ALL" if cfg["themes"]["scope"] == "pooled" else q
        themes, _ = T[scope]
        theme_cols = []
        gs = g[g.scope == scope]
        for j, t in enumerate(themes):
            sub = gs[gs.theme_idx == j].set_index("text_key")
            col = Lq["text_key"].map(sub["code"])       # NaN = theme not mentioned
            if pol:
                col = col.where(col != codes["neutral"], P)
                if pol.get("themes_follow_question"):
                    # a theme mentioned in a "like" answer is praise, in a "dislike" answer a complaint, unless the clause
                    # about it uses a clearly evaluative word that says otherwise ("smell bad")
                    opp_f = codes["negative"] if P == codes["positive"] else codes["positive"]
                    real = (col == opp_f) & Lq["text_key"].map(sub["ev_neg" if opp_f == codes["negative"] else "ev_pos"]).fillna(False).astype(bool)
                    if P == codes["positive"] and pol.get("firm_negative_conf", 0.95):
                        real |= Lq["text_key"].map(sub["firm_neg"]).fillna(False).astype(bool)
                    if P == codes["positive"] and "complaint" in sub:
                        real |= Lq["text_key"].map(sub["complaint"]).fillna(False).astype(bool) & col.notna()
                    col = col.where(col.isna(), P).where(~real, opp_f)
                tmin = pol.get("themes_opposite_min_conf")
                if tmin is not None:                             # weak rating against the question's polarity: see infer_polarity
                    opp_ = codes["negative"] if P == codes["positive"] else codes["positive"]
                    p_opp_ = Lq["text_key"].map(sub["p_neg" if P == codes["positive"] else "p_pos"])
                    col = col.where(~((col == opp_) & (p_opp_ < tmin)), P)
            col = col.where(~lw | col.isna(), codes["neutral"])
            if o["not_discussed_value"] is not None:
                col = col.where(~(col.isna() & Lq["valid"]), o["not_discussed_value"])
            out[f"{pre}{t['theme_name']}"] = out.row_id.map(col).astype("Int64")
            qcols[name].append((f"{pre}{t['theme_name']}", t["theme_name"]))
            theme_cols.append(f"{pre}{t['theme_name']}")
            d = Lq.reset_index()[["row_id", "respondent_id", "text_key"]].merge(sub.reset_index(), on="text_key")
            d["question"], d["theme"] = name, t["theme_name"]
            d["model_code"] = d["code"]                  # what the model said (after the negation fix), before the question-polarity rule
            if pol:
                d["code"] = d["code"].where(d["code"] != codes["neutral"], P)
                if pol.get("themes_follow_question"):
                    opp_f = codes["negative"] if P == codes["positive"] else codes["positive"]
                    real = (d["code"] == opp_f) & d["ev_neg" if opp_f == codes["negative"] else "ev_pos"].fillna(False).astype(bool)
                    if P == codes["positive"] and pol.get("firm_negative_conf", 0.95):
                        real |= d["firm_neg"].fillna(False).astype(bool)
                    if P == codes["positive"] and "complaint" in d:
                        real |= d["complaint"].fillna(False).astype(bool)
                    d["code"] = pd.Series(P, index=d.index).where(~real, opp_f)
                if pol.get("themes_opposite_min_conf") is not None:
                    opp_ = codes["negative"] if P == codes["positive"] else codes["positive"]
                    p_opp_ = d["p_neg"] if P == codes["positive"] else d["p_pos"]
                    d["code"] = d["code"].where(~((d["code"] == opp_) & (p_opp_ < pol["themes_opposite_min_conf"])), P)
            d["code"] = d["code"].where(~d["row_id"].map(lw).fillna(False).astype(bool), codes["neutral"])
            detail_rows.append(d[["row_id", "respondent_id", "question", "theme", "p_pos", "p_neg", "p_neu", "mixed", "model_code", "code", "lex_fix"]])

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
    reserved = {"results", "results_codes", "detail", "pairs", "themes", "timings", "spelling", COMBINED.lower()}

    def sheet_name(n):
        n = "".join(ch for ch in n if ch not in '[]:*?/\\')[:31] or "Question"     # characters Excel forbids in sheet names
        return n + " (Q)" if n.lower() in reserved else n

    try:
        long_mode = bool(cfg["input"].get("question_col"))                   # one comment per row: questions do not share rows
    except KeyError:
        long_mode = False
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        stacked = []
        for name, cols in qcols.items():                                    # one sheet per open-ended question
            qdf = words[other + [c for c, _ in cols]].copy()
            qdf.columns = other + [h for _, h in cols]
            if long_mode:
                qdf = qdf[qdf["Answer"].notna()]                            # only this question's own rows
            stacked.append(qdf[qdf["Answer"].notna()].assign(Question=name))
            sn = sheet_name(name)
            qdf.to_excel(xw, sheet_name=sn, index=False)
            ws = xw.sheets[sn]
            _style(ws, lambda h: 60 if h == "Answer" else 14)
            _colour(ws, colour_of)
        if len(stacked) > 1:                                                # all questions one under the other (Likes and Dislikes together)
            comb = pd.concat(stacked, ignore_index=True)
            comb = comb[other + ["Question"] + [c for c in comb.columns if c not in other + ["Question"]]]
            comb.to_excel(xw, sheet_name=COMBINED, index=False)
            _style(xw.sheets[COMBINED], lambda h: 60 if h == "Answer" else 14)
            _colour(xw.sheets[COMBINED], colour_of)
        if not long_mode:                                                   # side-by-side only makes sense when respondents answer every question
            words.to_excel(xw, sheet_name="results", index=False)
            _style(xw.sheets["results"], lambda h: 50 if h in qcols else 16)
            _colour(xw.sheets["results"], colour_of)
        if words is not out and not long_mode:
            out.to_excel(xw, sheet_name="results_codes", index=False)
        detail.to_excel(xw, sheet_name="detail", index=False)
        pairs.to_excel(xw, sheet_name="pairs", index=False)
        theme_tbl.to_excel(xw, sheet_name="themes", index=False)
        sp_file = rd / "spelling_corrections.json"
        if sp_file.exists() and json.loads(sp_file.read_text()):                 # typos repaired before the models read the answers
            pd.DataFrame(json.loads(sp_file.read_text())).to_excel(xw, sheet_name="spelling", index=False)
        pd.DataFrame([{"stage": k, "seconds": v} for k, v in timings.items()]).to_excel(xw, sheet_name="timings", index=False)
    (rd / "discovered_themes.json").write_text(json.dumps({s: t for s, (t, _) in T.items()}, indent=2))
    (rd / "timings.json").write_text(json.dumps(timings, indent=2))
    return str(path)
