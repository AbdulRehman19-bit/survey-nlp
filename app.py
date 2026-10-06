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
PRESETS = sorted(p.name for p in ROOT.glob("config*.yaml"))                 # settings presets: every config*.yaml next to the app
DEFAULT_PRESET = DEFAULTS.name                                                # generic: themes come from the answers; other presets are optional
APP_RUNS = ROOT / "runs" / "app"
MIN_SIM, FALLBACK_SIM, EVIDENCE = 0.40, 0.30, 0.60      # matching strictness: fixed defaults, not shown on the page
GREEN, YELLOW, RED = "#C6EFCE", "#FFEB9C", "#FFC7CE"
NOT_THEMES = ("respondent_id", "Answer", "Overall", "No theme")      # columns on a question sheet that are not themes
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


def load_preset(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def polarity_choice(entry) -> str:
    """The question-polarity choice a preset already makes for a question (the starting value of the selector)."""
    if not entry:
        return "Auto"
    if entry.get("off"):
        return "None"
    return {"positive": "Positive (a 'like' question)", "negative": "Negative (a 'dislike' question)"}.get(entry.get("neutral_as"), "Auto")


def build_config(s: dict) -> dict:
    """The chosen settings preset (a config*.yaml) + whatever the user chose on the Data step."""
    cfg = load_preset(s["preset"])
    cfg["run_dir"] = s["run_dir"]
    i = cfg["input"]
    i.update(path=s["path"], sheet=s["sheet"], header_row=s["header"], layout=s["layout"])
    for k in ("question_col", "text_col", "text_cols"):
        i.pop(k, None)
    if s["layout"] == "long":
        i.update(question_col=s["question_col"], text_col=s["text_col"])
    else:
        i["text_cols"] = s["text_cols"]
    cfg["assign"].setdefault("min_sim", s["min_sim"])                  # the preset's values win; these are only fallbacks
    cfg["assign"].setdefault("fallback_min_sim", s["fallback"])
    cfg["absa"].setdefault("evidence_threshold", s["evidence"])
    cfg["themes"]["n_themes"] = cfg["themes"]["n_total"] = s["edits"]["n_themes"]      # the number chosen is the total, fixed themes included
    base = dict(cfg["output"].get("question_polarity") or {})           # the preset's own choices (with their finer settings) stay
    pol = {}
    for q, choice in s["polarity"].items():
        if choice == "Positive (a 'like' question)":
            keep = base.get(q) if (base.get(q) or {}).get("neutral_as") == "positive" else None
            pol[q] = keep or {"neutral_as": "positive", "overall": "short", "themes_follow_question": True}
        elif choice == "Negative (a 'dislike' question)":
            keep = base.get(q) if (base.get(q) or {}).get("neutral_as") == "negative" else None
            pol[q] = keep or {"neutral_as": "negative", "overall": "short", "themes_follow_question": True}
        elif choice == "None":
            pol[q] = {"off": True}
    cfg["output"]["question_polarity"] = pol                            # "Auto" = no entry: the polarity is read from the answers
    cfg["output"].setdefault("auto_polarity", True)
    return cfg


def overrides_for(cfg: dict, edits: dict) -> dict:
    """Themes the preset requires + the ones added here. Removing or renaming a required theme in the table changes this list
    (the pipeline's own edit operations only know discovered themes); other edits go on as operations."""
    req = [dict(r) if isinstance(r, dict) else {"name": r} for r in (cfg["themes"].get("required") or [])]
    mine = {r["name"].lower() for r in edits["required"]}
    req = [r for r in req if r["name"].lower() not in mine] + [dict(r) for r in edits["required"]]
    ops = []
    for op in edits["ops"]:
        hit = [r for r in req if op[0] in ("drop", "rename") and r["name"].lower() == str(op[1]).lower()]
        if not hit:
            ops.append(op)
        elif op[0] == "drop":
            req.remove(hit[0])
        else:
            hit[0]["name"] = op[2]
    return {"themes": {"n_themes": edits["n_themes"], "required": req, "curation": {"ops": ops}}}


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
                art = pipeline.run(str(cfg_path), until, force, overrides=overrides_for(cfg, edits))
        except Exception as e:                  # shown as a message on the page instead of crashing the app
            status.update(label="That could not be applied", state="error")
            raise ValueError(str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}") from e
        status.update(label="Done", state="complete", expanded=False)
    if until == "themes":                       # the table of themes has new rows: give its editor a fresh state
        st.session_state.table_version = st.session_state.get("table_version", 0) + 1
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
    preset_name = DEFAULTS.name                                         # built-in settings: nothing to choose, everything is automatic
    preset = load_preset(preset_name)
    fixed = [r["name"] if isinstance(r, dict) else r for r in (preset["themes"].get("required") or [])]
    if "edits" not in st.session_state:
        st.session_state.edits = {**empty_edits(), "n_themes": max(preset["themes"].get("n_themes") or 10, len(fixed))}
    edits = ss("edits", empty_edits())
    n_themes = s1.slider("How many themes in total", 2, max(30, len(fixed) + 10), edits["n_themes"],
                         help="The built-in general-comment theme and the themes you add count too. If there are more, the smallest ones are "
                              "dropped (themes found in the answers first). A 'No theme' column is always added on top.")
    req_text = s1.text_area("Extra themes that must be included (one per line, optional words after a colon)",
                            value="\n".join(r["name"] + (": " + ", ".join(r["keywords"]) if r.get("keywords") else "")
                                            for r in edits["required"]),
                            placeholder="Price: cost, expensive, cheap\nPackaging", height=110)
    short = export.short_names(questions)
    s2.write("**What does each question ask?** A bare mention is read as praise or as a complaint.")
    polarity = {}
    choices = ["Auto", "Positive (a 'like' question)", "Negative (a 'dislike' question)", "None"]
    preset_pol = preset["output"].get("question_polarity") or {}
    for q in questions:
        polarity[short[q]] = s2.selectbox(f"{short[q]}", choices, index=choices.index(polarity_choice(preset_pol.get(short[q]))),
                                          key=f"pol_{q}_{preset_name}")

    new_required = parse_required(req_text)
    if n_themes != edits["n_themes"] or new_required != edits["required"]:
        # A different number of themes gives a different set of themes, so renames / merges / removals made on the old set
        # no longer apply (they name themes that may not exist any more). Start the edits again; added themes are kept.
        ops = [] if n_themes != edits["n_themes"] else edits["ops"]
        st.session_state.edits = {**edits, "n_themes": n_themes, "required": new_required, "ops": ops}
        st.session_state.pop("art", None)
        st.session_state.table_version = st.session_state.get("table_version", 0) + 1
    stem = Path(path).stem
    settings = dict(path=path, sheet=sheet, header=header, layout=layout, question_col=question_col, text_col=text_col,
                    text_cols=text_cols, polarity=polarity, preset=preset_name, min_sim=MIN_SIM, fallback=FALLBACK_SIM, evidence=EVIDENCE,
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
        all_themes = [t for ts, _ in art["themes"].values() for t in ts]
        found_n = max((t.get("candidates_found") or 0 for t in all_themes), default=0)
        fixed_n = sum(1 for t in all_themes if t.get("required") or t.get("seeded"))     # from the preset or typed on the Data tab
        found_kept = len(all_themes) - fixed_n
        parts = []
        if found_n:
            parts.append(f"**{found_n} topics were found in the answers; {found_kept} of them are kept** (the others were merged into the nearest kept one)")
        if fixed_n:
            parts.append(f"**{fixed_n} built-in or added themes** (general comments, plus any you typed)")
        if parts:
            st.write(" and ".join(parts) + f". The table shows all {len(table)}.")
        edited = st.data_editor(
            table.drop(columns="_orig"), key=f"theme_table_{st.session_state.get('table_version', 0)}", hide_index=True, width="stretch",
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
    themed = sum(int((sheets[q][[c for c in sheets[q].columns if c not in NOT_THEMES]].notna().any(axis=1)).sum())
                 for q in question_sheets)
    m[0].metric("Answers", f"{n_answers:,}")
    m[1].metric("With at least one theme", f"{themed / max(n_answers, 1):.0%}")
    m[2].metric("Questions", len(question_sheets))
    m[3].metric("Themes", len(sheets["themes"]))

    qtabs = st.tabs(question_sheets)
    for qt, qname in zip(qtabs, question_sheets):
        with qt:
            d = sheets[qname]
            theme_cols = [c for c in d.columns if c not in NOT_THEMES]
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
