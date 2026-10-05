"""Streamlit front end for the survey NLP pipeline.   Run:  streamlit run app.py

1  Data      pick a file, check the detected layout and columns
2  Themes    find themes from the answers, then rename / merge / remove / add / change how many
3  Analyse   run sentiment for every theme
4  Results   coloured tables per question, charts, download
"""
import contextlib
import copy
import glob
import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).parent
DEFAULTS = ROOT / "config_default.yaml"
APP_RUNS = ROOT / "runs" / "app"
GREEN, YELLOW, RED = "#C6EFCE", "#FFEB9C", "#FFC7CE"
INTERNAL_SHEETS = {"results", "results_codes", "detail", "pairs", "themes", "timings"}

st.set_page_config(page_title="Survey themes and sentiment", layout="wide")


# ----------------------------------------------------------------------------- helpers
class LiveLog(io.TextIOBase):
    """Shows what the pipeline prints (stage names and timings) while it runs."""

    def __init__(self, box):
        self.box, self.buf = box, ""

    def write(self, s):
        self.buf += s
        self.box.code(self.buf[-3000:] or " ", language=None)
        return len(s)


def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def empty_edits():
    return {"n_themes": 10, "required": [], "ops": []}


@st.cache_data(show_spinner=False)
def sheet_names(path: str, mtime: float):
    return pd.ExcelFile(path).sheet_names


@st.cache_data(show_spinner=False)
def load_preview(path: str, mtime: float, sheet, header):
    c = {"path": path, "sheet": sheet, "header_row": header}
    df = ingest.read_table(c)
    return df


def save_upload(upload) -> str:
    folder = APP_RUNS / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / upload.name
    target.write_bytes(upload.getbuffer())
    return str(target)


def parse_required(text: str):
    """One theme per line:  Price: cost, expensive, cheap   (words after the colon are optional)."""
    out = []
    for line in text.splitlines():
        name, _, kw = line.partition(":")
        if name.strip():
            item = {"name": name.strip()}
            if kw.strip():
                item["keywords"] = [k.strip() for k in kw.split(",") if k.strip()]
            out.append(item)
    return out


def build_config(s: dict) -> dict:
    """Generic defaults + whatever the user chose on the Data step."""
    cfg = yaml.safe_load(DEFAULTS.read_text(encoding="utf-8"))
    cfg["run_dir"] = s["run_dir"]
    i = cfg["input"]
    i.update(path=s["path"], sheet=s["sheet"], header_row=s["header"], layout=s["layout"])
    for k in ("question_col", "text_col", "text_cols"):
        i.pop(k, None)
    if s["layout"] == "long":
        i.update(question_col=s["question_col"], text_col=s["text_col"])
    else:
        i["text_cols"] = s["text_cols"]
    cfg["assign"].update(min_sim=s["min_sim"], fallback_min_sim=s["fallback"])
    cfg["absa"].update(evidence_threshold=s["evidence"])
    cfg["themes"]["n_themes"] = s["edits"]["n_themes"]
    pol = {}
    for q, choice in s["polarity"].items():
        if choice == "Positive (a 'like' question)":
            pol[q] = {"neutral_as": "positive", "overall": "short"}
        elif choice == "Negative (a 'dislike' question)":
            pol[q] = {"neutral_as": "negative", "overall": "short"}
        elif choice == "None":
            pol[q] = {"off": True}
    cfg["output"]["question_polarity"] = pol
    cfg["output"]["auto_polarity"] = True
    return cfg


def overrides_for(edits: dict) -> dict:
    return {"themes": {"n_themes": edits["n_themes"], "required": edits["required"],
                       "curation": {"ops": list(edits["ops"])}}}


def run_stage(cfg: dict, edits: dict, until: str, force=False):
    """Write the config, run the pipeline up to `until`, show the log live. Returns the stage outputs."""
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with st.status(f"Running up to '{until}'…", expanded=True) as status:
        log = LiveLog(st.empty())
        try:
            with contextlib.redirect_stdout(log):
                art = pipeline.run(str(cfg_path), until, force, overrides=overrides_for(edits))
        except ValueError as e:
            status.update(label="That could not be applied", state="error")
            raise
        status.update(label="Done", state="complete", expanded=False)
    return art


def themes_table(art) -> pd.DataFrame:
    rows = []
    for scope, (themes, _) in art["themes"].items():
        for t in themes:
            rows.append({"Keep": True, "Theme": t["theme_name"], "Clauses": t["frequency"] or None,
                         "Main words": ", ".join((t.get("curated_keywords") or t["keywords"])[:6]), "_orig": t["theme_name"]})
    return pd.DataFrame(rows)


def latest_results(run_dir: str):
    files = glob.glob(os.path.join(run_dir, "aspect_sentiment_results*.xlsx"))
    return max(files, key=os.path.getmtime) if files else None


def colour_cell(v):
    return {"Positive": f"background-color:{GREEN};color:#111", "Negative": f"background-color:{RED};color:#111",
            "Neutral": f"background-color:{YELLOW};color:#111"}.get(v, "")


# ----------------------------------------------------------------------------- page
st.title("Survey themes and sentiment")
st.caption("Open-ended answers in, themes and a Positive / Negative / Neutral rating per theme out.")


@st.cache_resource(show_spinner="Loading the language tools. This takes about 30 seconds the first time the app starts.")
def backend():
    from survey_nlp import export, ingest, pipeline      # heavy imports (torch, transformers): load once, after the page shows
    return export, ingest, pipeline


export, ingest, pipeline = backend()

tab_data, tab_themes, tab_results = st.tabs(["1 · Data", "2 · Themes and analysis", "3 · Results"])

# ----------------------------------------------------------------------------- 1. data
with tab_data:
    left, right = st.columns([1, 2])
    with left:
        source = st.radio("Where is the file?", ["Upload a file", "Pick from the data folder"], horizontal=True)
        path = None
        if source == "Upload a file":
            up = st.file_uploader("Excel or CSV", type=["xlsx", "xls", "csv"])
            if up is not None:
                path = save_upload(up)
        else:
            found = sorted(glob.glob(str(ROOT / "data" / "*.xlsx")) + glob.glob(str(ROOT / "data" / "*.xls"))
                           + glob.glob(str(ROOT / "data" / "*.csv")))
            if found:
                path = st.selectbox("File", found, format_func=lambda p: Path(p).name)
            else:
                st.info("No files in the data folder yet.")
        sheet, header = 0, "auto"
        if path and path.lower().endswith((".xlsx", ".xls")):
            names = sheet_names(path, os.path.getmtime(path))
            sheet = st.selectbox("Sheet", names) if len(names) > 1 else names[0]
        if path:
            auto = st.checkbox("Find the header row automatically", value=True)
            header = "auto" if auto else st.number_input("Header row (0 = first row)", 0, 50, 0)

    if not path:
        st.stop()

    df = load_preview(path, os.path.getmtime(path), sheet, header)
    c_in = {"path": path, "header_row": header, "sheet": sheet}
    found_layout = ingest.detect_layout(df, c_in)
    with right:
        st.write(f"**{len(df):,} rows, {df.shape[1]} columns**")
        st.dataframe(df.head(8).astype(str), width="stretch", height=250)

    st.subheader("What is in the file")
    c1, c2 = st.columns(2)
    layout_label = {"long": "One comment per row (a column says which question it answers)",
                    "wide": "One column per question (one row per respondent)"}
    layout = c1.radio("Layout", ["long", "wide"], index=0 if found_layout["layout"] == "long" else 1,
                      format_func=layout_label.get, help="Detected automatically. Change it if it looks wrong.")
    cols = list(df.columns)
    question_col = text_col = None
    text_cols = []
    if layout == "long":
        qd = found_layout["question_col"] or cols[0]
        td = found_layout["text_col"] or cols[-1]
        question_col = c2.selectbox("Column that names the question", cols, index=cols.index(qd) if qd in cols else 0)
        text_col = c2.selectbox("Column with the comments", cols, index=cols.index(td) if td in cols else 0)
        questions = [str(x) for x in df[question_col].dropna().astype(str).unique()]
        c2.write("Questions found:  " + ",  ".join(f"**{q}** ({(df[question_col].astype(str) == q).sum()})" for q in questions))
    else:
        text_cols = c2.multiselect("Open-ended columns", cols, default=[x for x in found_layout["text_cols"] if x in cols])
        questions = [str(x) for x in text_cols]
    if not questions:
        st.warning("No open-ended questions found yet. Pick the columns that hold the written answers.")
        st.stop()

    st.subheader("Settings")
    s1, s2 = st.columns(2)
    edits = ss("edits", empty_edits())
    n_themes = s1.slider("How many themes to keep", 2, 30, edits["n_themes"],
                         help="More themes are found first; the smaller ones are merged into the nearest kept theme.")
    req_text = s1.text_area("Themes that must be included (one per line, optional words after a colon)",
                            value="\n".join(r["name"] + (": " + ", ".join(r["keywords"]) if r.get("keywords") else "")
                                            for r in edits["required"]),
                            placeholder="Price: cost, expensive, cheap\nPackaging", height=110)
    short = export.short_names(questions)
    s2.write("**What does each question ask?** A bare mention is read as praise or as a complaint.")
    polarity = {}
    choices = ["Auto", "Positive (a 'like' question)", "Negative (a 'dislike' question)", "None"]
    for q in questions:
        polarity[short[q]] = s2.selectbox(f"{short[q]}", choices, key=f"pol_{q}")
    with st.expander("Advanced"):
        a1, a2, a3 = st.columns(3)
        min_sim = a1.slider("Match strength to a theme", 0.10, 0.70, 0.30, 0.01,
                            help="Lower assigns more answers to themes, higher is stricter.")
        fallback = a2.slider("Last-resort match", 0.05, 0.50, 0.25, 0.01,
                             help="An answer nothing else matched joins its closest theme if at least this close.")
        evidence = a3.slider("Sentiment confidence needed", 0.40, 0.90, 0.60, 0.05)

    new_required = parse_required(req_text)
    if n_themes != edits["n_themes"] or new_required != edits["required"]:
        st.session_state.edits = {**edits, "n_themes": n_themes, "required": new_required}
        st.session_state.pop("art", None)
    stem = Path(path).stem
    settings = dict(path=path, sheet=sheet, header=header, layout=layout, question_col=question_col, text_col=text_col,
                    text_cols=text_cols, polarity=polarity, min_sim=min_sim, fallback=fallback, evidence=evidence,
                    run_dir=str(APP_RUNS / stem), edits=st.session_state.edits)
    st.session_state.settings = settings
    st.success("Data ready. Go to the **Themes and analysis** tab.")

# ----------------------------------------------------------------------------- 2. themes + analysis
with tab_themes:
    s = ss("settings", None)
    if not s:
        st.info("Choose a file in the first tab.")
        st.stop()
    edits = st.session_state.edits
    cfg = build_config(s)

    go1, go2 = st.columns([1, 3])
    if go1.button("Find themes", type="primary") or "art" not in st.session_state and st.session_state.get("auto_find"):
        st.session_state.auto_find = True
        try:
            st.session_state.art = run_stage(cfg, edits, "themes")
        except ValueError as e:
            st.error(str(e))
    go2.caption("Themes are found from the answers themselves. Edit them below, then run the analysis.")

    art = st.session_state.get("art")
    if art and "themes" in art:
        table = themes_table(art)
        found_n = next(iter(art["themes"].values()))[0][0].get("candidates_found") if art["themes"] else None
        if found_n:
            st.write(f"**{found_n} topics were found; the {len(table)} below are kept.** The rest were merged into the nearest kept theme.")
        edited = st.data_editor(
            table.drop(columns="_orig"), key="theme_table", hide_index=True, width="stretch",
            column_config={"Keep": st.column_config.CheckboxColumn("Keep", width="small"),
                           "Theme": st.column_config.TextColumn("Theme (click to rename)"),
                           "Clauses": st.column_config.NumberColumn("Answers in theme", disabled=True),
                           "Main words": st.column_config.TextColumn("Main words", disabled=True)})

        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown("**Rename or remove**")
            st.caption("Rename in the table, untick Keep to remove, then apply.")
            if st.button("Apply table changes"):
                new_ops = []
                for old, row in zip(table["_orig"], edited.itertuples()):
                    if not row.Keep:
                        new_ops.append(["drop", old])
                    elif row.Theme.strip() and row.Theme != old:
                        new_ops.append(["rename", old, row.Theme.strip()])
                if new_ops:
                    edits["ops"] += new_ops
                    try:
                        st.session_state.art = run_stage(cfg, edits, "themes")
                        st.rerun()
                    except ValueError as e:
                        edits["ops"] = edits["ops"][:-len(new_ops)]
                        st.error(str(e))
        with e2:
            st.markdown("**Merge themes**")
            names = list(table["Theme"])
            pick = st.multiselect("Themes to merge", names, key="merge_pick")
            merged_name = st.text_input("Name of the merged theme", key="merge_name")
            if st.button("Merge") and len(pick) >= 2:
                edits["ops"].append(["merge", merged_name.strip() or None, pick])
                try:
                    st.session_state.art = run_stage(cfg, edits, "themes")
                    st.rerun()
                except ValueError as e:
                    edits["ops"].pop()
                    st.error(str(e))
        with e3:
            st.markdown("**Add a theme**")
            add_name = st.text_input("Theme name", key="add_name")
            add_kw = st.text_input("Words that signal it (optional, comma-separated)", key="add_kw")
            if st.button("Add") and add_name.strip():
                item = {"name": add_name.strip()}
                if add_kw.strip():
                    item["keywords"] = [k.strip() for k in add_kw.split(",") if k.strip()]
                edits["required"] = [r for r in edits["required"] if r["name"].lower() != item["name"].lower()] + [item]
                try:
                    st.session_state.art = run_stage(cfg, edits, "themes")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        if st.button("Undo all my edits"):
            st.session_state.edits = {**empty_edits(), "n_themes": edits["n_themes"]}
            st.session_state.pop("art", None)
            st.rerun()
        if edits["ops"]:
            with st.expander(f"Edits so far ({len(edits['ops'])})"):
                for op in edits["ops"]:
                    st.write("• " + {"drop": "Removed {1}", "rename": "Renamed {1} to {2}",
                                     "merge": "Merged {2} into {1}"}[op[0]].format(*op).replace("[", "").replace("]", "").replace("'", ""))

        st.divider()
        if st.button("Run the analysis", type="primary"):
            try:
                st.session_state.final = run_stage(cfg, edits, "export")
                st.session_state.result_dir = cfg["run_dir"]
                st.success("Finished. Open the **Results** tab.")
            except ValueError as e:
                st.error(str(e))

# ----------------------------------------------------------------------------- 3. results
with tab_results:
    s = st.session_state.get("settings")
    run_dir = st.session_state.get("result_dir") or (s["run_dir"] if s else None)
    f = latest_results(run_dir) if run_dir else None
    if not f:
        st.info("Run the analysis in the previous tab to see results here.")
        st.stop()
    sheets = pd.read_excel(f, sheet_name=None)
    question_sheets = [n for n in sheets if n not in INTERNAL_SHEETS]          # one sheet per open-ended question
    st.caption(f"Results file: {f}")
    if Path(f).name != "aspect_sentiment_results.xlsx":
        st.warning("The standard results file is open in Excel, so this run was saved under a different name.")

    # headline numbers
    m = st.columns(4)
    n_answers = sum(len(sheets[q]) for q in question_sheets)
    themed = sum(int((sheets[q][[c for c in sheets[q].columns if c not in ("respondent_id", "Answer", "Overall")]].notna().any(axis=1)).sum())
                 for q in question_sheets)
    m[0].metric("Answers", f"{n_answers:,}")
    m[1].metric("With at least one theme", f"{themed / max(n_answers, 1):.0%}")
    m[2].metric("Questions", len(question_sheets))
    m[3].metric("Themes", len(sheets["themes"]))

    qtabs = st.tabs(question_sheets)
    for qt, qname in zip(qtabs, question_sheets):
        with qt:
            d = sheets[qname]
            theme_cols = [c for c in d.columns if c not in ("respondent_id", "Answer", "Overall")]
            counts = pd.DataFrame({c: d[c].value_counts() for c in theme_cols}).T.reindex(
                columns=["Positive", "Neutral", "Negative"]).fillna(0).astype(int)
            counts = counts.assign(total=counts.sum(axis=1)).sort_values("total", ascending=False).drop(columns="total")
            ov = d["Overall"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).astype(int)
            g1, g2 = st.columns([2, 1])
            with g1:
                st.markdown("**How each theme is rated**")
                st.bar_chart(counts, color=["#43a047", "#f9a825", "#e53935"],
                             horizontal=True, height=max(220, 34 * len(counts)))
            with g2:
                st.markdown("**Whole answers**")
                st.bar_chart(ov.to_frame("answers"), color="#5c6bc0", height=260)
            f1, f2 = st.columns(2)
            only = f1.multiselect("Show answers about", theme_cols, key=f"flt_{qname}")
            word = f2.text_input("Search the answer text", key=f"srch_{qname}")
            view = d
            if only:
                view = view[view[only].notna().any(axis=1)]
            if word:
                view = view[view["Answer"].astype(str).str.contains(word, case=False, na=False)]
            st.write(f"{len(view):,} answers")
            st.dataframe(view.style.map(colour_cell), width="stretch", height=520, hide_index=True)

    st.divider()
    d1, d2 = st.columns(2)
    d1.download_button("Download the Excel workbook", data=Path(f).read_bytes(), file_name="survey_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    with d2.expander("Run times"):
        st.dataframe(sheets["timings"], hide_index=True)
