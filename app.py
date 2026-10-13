"""Streamlit front end for the survey NLP pipeline.   Run:  streamlit run app.py

1  Data      pick a file, check the detected layout and columns
2  Themes    find themes from the answers, then rename / merge / remove / add / change how many
3  Analyse   run sentiment for every theme
4  Results   coloured tables per question, charts, download
"""
import contextlib
import copy
import glob
import hashlib
import io
import json
import logging
import os
import shutil
import time
import sys
import traceback
import uuid
import warnings
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))                                          # the package is imported from the source folder: nothing needs to be pip-installed (Streamlit Cloud)
os.environ.setdefault("SURVEY_NLP_QUIET", "1")                                  # theme diagnostics (splits, renames, merges, quality line) stay out of the run log here
# Terminal noise that does not matter: Streamlit's file watcher probes every transformers module and logs a traceback for the ones that need
# the optional torchvision; the Hugging Face hub warns about unauthenticated requests although the models are local; a library deprecation
# notice and pandas' note about regex groups. Real errors are not touched.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*has match groups.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)
DEFAULTS = ROOT / "config_default.yaml"
PRESETS = sorted(p.name for p in ROOT.glob("config*.yaml"))                 # settings presets: every config*.yaml next to the app
DEFAULT_PRESET = DEFAULTS.name                                                # generic: themes come from the answers; other presets are optional
APP_RUNS = ROOT / "runs" / "app"
MIN_SIM, FALLBACK_SIM, EVIDENCE = 0.40, 0.12, 0.60      # matching strictness: fixed defaults, not shown on the page
GREEN, YELLOW, RED = "#C6EFCE", "#FFEB9C", "#FFC7CE"
NOT_THEMES = ("respondent_id", "Answer", "Overall", "Question")      # columns on a question sheet that are not themes
INTERNAL_SHEETS = {"results", "results_codes", "detail", "pairs", "themes", "quality", "timings", "spelling", "Combined"}

st.set_page_config(page_title="Survey themes and sentiment", layout="wide")

CSS = """
<style>
.block-container {padding-top: 4.5rem; max-width: 1280px;}
footer {visibility: hidden;}
.hero {background: #1f3a5f; color: #fff; padding: 1.4rem 1.8rem;
       border-radius: 8px; margin-bottom: 1.2rem;}
.hero h1 {color: #fff; margin: 0 !important; padding: 0 !important; font-size: 1.9rem; font-weight: 700; letter-spacing: -.01em; line-height: 1.2;}
.hero p {margin: .35rem 0 0; opacity: .92; font-size: 1.02rem;}
.stTabs [data-baseweb="tab-list"] {gap: .4rem; border-bottom: 1px solid #dfe2ee;}
.stTabs [data-baseweb="tab"] {height: 2.8rem; padding: 0 1.1rem; border-radius: 10px 10px 0 0; font-weight: 600;}
.stTabs [aria-selected="true"] {background: transparent;}
[data-testid="stMetric"] {background: transparent; border: 1px solid rgba(128,134,160,.35); border-radius: 8px; padding: .9rem 1.1rem;
                          box-shadow: 0 1px 3px rgba(20,24,60,.05);}
[data-testid="stMetricLabel"] {opacity: .7;}
[data-testid="stMetricValue"] {font-weight: 700;}
.stButton > button, .stDownloadButton > button {border-radius: 10px; font-weight: 600; padding: .45rem 1.1rem;}
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {border: 1px solid #e3e6f2; border-radius: 12px; overflow: hidden;}
[data-testid="stElementToolbarButton"]:has(button[aria-label^="Show data"]) {display: none;}   /* the chart's "Show data" button: there is no way back from it */
[data-testid="stExpander"] {background: transparent; border-radius: 12px; border: 1px solid rgba(128,134,160,.35);}
h2, h3 {letter-spacing: -.01em;}
.pill {display: inline-block; padding: .1rem .6rem; border-radius: 999px; font-size: .8rem; font-weight: 600; margin-right: .3rem;}
.pill.pos {background: #dcfce7; color: #166534;} .pill.neu {background: #fef9c3; color: #854d0e;} .pill.neg {background: #fee2e2; color: #991b1b;}
.sentbar {display: flex; height: 14px; border-radius: 7px; overflow: hidden; margin: .2rem 0 .8rem;}
</style>
"""


# ----------------------------------------------------------------------------- helpers
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


def session_dir() -> Path:
    """A folder of this visitor's own: on a shared server two people uploading "survey.xlsx" must not overwrite each other's file or results."""
    if "sid" not in st.session_state:
        st.session_state.sid = uuid.uuid4().hex[:12]
    return APP_RUNS / st.session_state.sid


def remove_old_sessions(max_age_hours=24):
    """Visitors' folders are not kept for ever (the server's disk is small). Only folders named like a session id are touched."""
    if not APP_RUNS.exists():
        return
    for d in APP_RUNS.iterdir():
        try:
            if d.is_dir() and len(d.name) == 12 and all(ch in "0123456789abcdef" for ch in d.name) and time.time() - d.stat().st_mtime > max_age_hours * 3600:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def save_upload(upload) -> str:
    folder = session_dir() / "uploads"
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


def n_general(cfg: dict) -> int:
    """Built-in 'not specific' themes (general: true, e.g. General comment). They come on top of the number the user chooses."""
    return sum(1 for r in (cfg["themes"].get("required") or []) if isinstance(r, dict) and r.get("general"))


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
    cfg["themes"]["n_themes"] = cfg["themes"]["n_total"] = s["edits"]["n_themes"] + n_general(cfg)   # the number chosen, plus the general-comment theme on top
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
    found = set()           # names that discovered themes carry now (set by an earlier rename / merge): an edit on such a name is not about a required theme
    for op in edits["ops"]:
        hit = [r for r in req if op[0] in ("drop", "rename") and r["name"].lower() == str(op[1]).lower()
               and str(op[1]).lower() not in found]
        if not hit:
            ops.append(op)
            if op[0] == "rename":
                found.discard(str(op[1]).lower())
                found.add(str(op[2]).lower())
            elif op[0] == "drop":
                found.discard(str(op[1]).lower())
            elif op[0] == "merge":
                found.difference_update(str(n).lower() for n in op[2])
                if op[1]:
                    found.add(str(op[1]).lower())
        elif op[0] == "drop":
            if hit[0].get("general"):
                hit[0]["hidden"] = True                    # the search still uses it: removing it must not change the other themes
            else:
                req.remove(hit[0])
            ops.append(["drop_if_present", op[1]])         # the answers may also have produced a theme of this name: it goes too
        else:
            hit[0].setdefault("defined_by", hit[0]["name"])    # a new name is a label: the theme keeps what it was defined by
            ops.append(["rename_if_present", op[1], op[2]])
            hit[0]["name"] = op[2]
    n = edits["n_themes"] + n_general(cfg)
    shown_general = sum(1 for r in req if isinstance(r, dict) and r.get("general") and not r.get("hidden"))
    return {"themes": {"n_themes": n, "n_total": edits["n_themes"] + shown_general, "required": req, "curation": {"ops": ops}}}


def name_taken(name: str, names, except_name=None) -> bool:
    """Is there already a theme with this name (case does not matter)? `except_name` is the theme being renamed itself."""
    n = str(name).strip().lower()
    return any(x.lower() == n for x in names if x != except_name)


def settings_sig(cfg: dict, edits: dict) -> str:
    """Fingerprint of everything that decides WHICH themes are found (not of the merge / rename / remove edits made on them)."""
    ov = overrides_for(cfg, {**edits, "ops": []})          # without the edits made on the table (they change the list, not the search)
    ov["themes"] = {k: v for k, v in ov["themes"].items() if k != "curation"}
    return hashlib.sha1(json.dumps([cfg, ov], sort_keys=True, default=str).encode()).hexdigest()


def run_stage(cfg: dict, edits: dict, until: str, force=False, slot=None):
    """Write the config, run the pipeline up to `until`, show the progress live in `slot` (a full-width container; the place of the
    call otherwise, which inside a narrow column squeezes it). Returns the stage outputs."""
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    with (slot if slot is not None else st).spinner("Running…"):
        try:
            with contextlib.redirect_stdout(io.StringIO()):                # progress lines are not shown, only that it is running
                # for the themes step also run assign (it takes a moment): the table then shows how often each theme really occurs
                art = pipeline.run(str(cfg_path), "assign" if until == "themes" else until, force, overrides=overrides_for(cfg, edits))
        except Exception as e:                  # shown as a message on the page instead of crashing the app
            if isinstance(e, ValueError):
                msg = str(e)
            else:                               # name where it was raised: an empty message (e.g. EOFError) says nothing alone
                fr = traceback.extract_tb(e.__traceback__)[-1]
                msg = f"{type(e).__name__}: {e} (at {Path(fr.filename).name}:{fr.lineno}, in {fr.name})"
            if "usable clauses" in msg or "themes found" in msg:
                msg = ("There are too few answers, or they are too alike, to find themes. Use a file with more answers, "
                       "or ask for fewer themes. (" + msg.split(" Lower ")[0].split(" Use themes")[0].strip() + ")")
            if "unknown theme" in msg:
                msg += "  (The table on screen is out of date. Press 'Find themes' again and redo the edit.)"
            raise ValueError(msg) from e
    if until == "themes":                       # the table of themes has new rows: give its editor a fresh state
        st.session_state.art_sig = settings_sig(cfg, edits)
        st.session_state.table_version = st.session_state.get("table_version", 0) + 1
    return art


def themes_table(art) -> pd.DataFrame:
    rows = []
    ans = {}                                            # (scope, theme index) -> different answers the theme was assigned to
    if "assign" in art:
        a = art["assign"].merge(art["segment"][["clause_id", "text_key"]], on="clause_id")
        ans = a.groupby(["scope", "theme_idx"]).text_key.nunique().to_dict()
    for scope, (themes, _) in art["themes"].items():
        for j, t in enumerate(themes):
            rows.append({"Keep": True, "Theme": t["theme_name"],
                         "Answers": ans.get((scope, j), 0) if ans else (t["frequency"] or None),
                         "Main words": ", ".join((t.get("curated_keywords") or t["keywords"])[:6]), "_orig": t["theme_name"]})
    return pd.DataFrame(rows)


def latest_results(run_dir: str):
    files = glob.glob(os.path.join(run_dir, "aspect_sentiment_results*.xlsx"))
    return max(files, key=os.path.getmtime) if files else None


def colour_cell(v):
    return {"Positive": f"background-color:{GREEN};color:#111", "Negative": f"background-color:{RED};color:#111",
            "Neutral": f"background-color:{YELLOW};color:#111"}.get(v, "")


# ----------------------------------------------------------------------------- page
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <h1>Survey themes and sentiment</h1>
</div>
""", unsafe_allow_html=True)


def sentiment_bar(pos, neu, neg):
    """A thin stacked bar with the share of Positive / Neutral / Negative, and a legend of pills with the counts."""
    tot = max(pos + neu + neg, 1)
    st.markdown(
        f'<div class="sentbar"><div style="width:{pos / tot:.1%};background:#43a047"></div>'
        f'<div style="width:{neu / tot:.1%};background:#f9a825"></div><div style="width:{neg / tot:.1%};background:#e53935"></div></div>'
        f'<span class="pill pos">Positive {pos:,} ({pos / tot:.0%})</span><span class="pill neu">Neutral {neu:,} ({neu / tot:.0%})</span>'
        f'<span class="pill neg">Negative {neg:,} ({neg / tot:.0%})</span>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the language tools. This takes about 30 seconds the first time the app starts.")
def backend():
    from survey_nlp import export, ingest, pipeline      # heavy imports (torch, transformers): load once, after the page shows
    remove_old_sessions()
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()                      # no model load reports in the terminal
    return export, ingest, pipeline


export, ingest, pipeline = backend()

TAB_RESULTS = "3 · Results"
tab_data, tab_themes, tab_results = st.tabs(["1 · Data", "2 · Themes and analysis", TAB_RESULTS], key="main_tabs", on_change="rerun")

# ----------------------------------------------------------------------------- 1. data
with tab_data:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Your file")
        path = None
        up = st.file_uploader("Excel or CSV", type=["xlsx", "xls", "csv"])
        if up is not None:
            path = save_upload(up)
        sheet, header = 0, "auto"
        if path and path.lower().endswith((".xlsx", ".xls")):
            try:
                names = sheet_names(path, os.path.getmtime(path))
            except Exception:
                st.error("This file could not be opened as an Excel workbook. Check that it is a real .xlsx / .xls file and not damaged.")
                st.stop()
            sheet = st.selectbox("Sheet", names) if len(names) > 1 else names[0]
        if path:
            auto = st.checkbox("Find the header row automatically", value=True)
            header = "auto" if auto else st.number_input("Header row (0 = first row)", 0, 50, 0)

    if not path:
        st.stop()

    try:
        df = load_preview(path, os.path.getmtime(path), sheet, header)
    except Exception as e:
        st.error(f"This file could not be read: {e}")
        st.stop()
    if df.empty or df.shape[1] == 0:
        st.warning("The file has no rows to analyse.")
        st.stop()
    c_in = {"path": path, "header_row": header, "sheet": sheet}
    found_layout = ingest.detect_layout(df, c_in)
    with right:
        st.subheader("Preview")
        st.caption(f"{len(df):,} rows · {df.shape[1]} columns · first 8 rows shown")
        st.dataframe(df.head(8).astype(str), width="stretch", height=250)

    layout = found_layout["layout"]                                     # detected automatically; nothing to choose on the page
    cols = list(df.columns)
    question_col = text_col = None
    text_cols = []
    if layout == "long":
        question_col, text_col = found_layout["question_col"], found_layout["text_col"]
        questions = [str(x) for x in df[question_col].dropna().astype(str).unique()]
    else:
        text_cols = [x for x in found_layout["text_cols"] if x in cols]
        questions = [str(x) for x in text_cols]
    if not questions:
        st.warning("No open-ended questions were found in this file.")
        st.stop()

    st.subheader("Settings")
    s1 = st.container()
    preset_name = DEFAULTS.name                                         # built-in settings: nothing to choose, everything is automatic
    preset = load_preset(preset_name)
    fixed = [r["name"] if isinstance(r, dict) else r for r in (preset["themes"].get("required") or [])]
    if "edits" not in st.session_state:
        st.session_state.edits = {**empty_edits(), "n_themes": preset["themes"].get("n_themes") or 10}
    edits = ss("edits", empty_edits())
    if "n_box" not in st.session_state:                                 # Streamlit can drop the key; restore it
        st.session_state["n_box"] = edits["n_themes"]
    n_themes = int(s1.number_input(
        "How many themes", 2, 100, key="n_box",
        help="Type the number of themes. The built-in general-comment theme ('not specified') is added on top of this number, "
             "so 10 gives 10 themes + 1. Themes you add yourself count inside the number."))
    req_text = s1.text_area("Themes that must be included",
                            value="\n".join(r["name"] + (": " + ", ".join(r["keywords"]) if r.get("keywords") else "")
                                            for r in edits["required"]),
                            placeholder="Price: cost, expensive, cheap\nPackaging", height=110)
    polarity = {}           # no per-question choice on the page: what each question asks is always read from the answers (output.auto_polarity)

    new_required = parse_required(req_text)
    if n_themes != edits["n_themes"] or new_required != edits["required"]:
        # A different number of themes gives a different set of themes, so renames / merges / removals made on the old set
        # no longer apply (they name themes that may not exist any more). Start the edits again; added themes are kept.
        # A changed must-include list does the same: earlier edits may name a theme that was typed there (e.g. "rename Price").
        st.session_state.edits = {**edits, "n_themes": n_themes, "required": new_required, "ops": []}
        st.session_state.pop("art", None)
        st.session_state.table_version = st.session_state.get("table_version", 0) + 1
    stem = Path(path).stem
    settings = dict(path=path, sheet=sheet, header=header, layout=layout, question_col=question_col, text_col=text_col,
                    text_cols=text_cols, polarity=polarity, preset=preset_name, min_sim=MIN_SIM, fallback=FALLBACK_SIM, evidence=EVIDENCE,
                    run_dir=str(session_dir() / stem), edits=st.session_state.edits)
    st.session_state.settings = settings
    st.success("Data ready. Go to the **Themes and analysis** tab.")
    st.button("Go to Themes and analysis", on_click=lambda: st.session_state.update(main_tabs="2 · Themes and analysis"))

# ----------------------------------------------------------------------------- 2. themes + analysis
with tab_themes:
    s = ss("settings", None)
    if not s:
        st.info("Choose a file in the first tab.")
        st.stop()
    edits = st.session_state.edits
    cfg = build_config(s)
    sig = settings_sig(cfg, edits)
    if "art" in st.session_state and st.session_state.get("art_sig") != sig:
        # The table on screen was made with other settings than the ones in force now (a changed number of themes, an added theme,
        # a new version of the app): edits made on it would name themes the next search does not produce. Search again from scratch.
        st.session_state.pop("art", None)
        edits["ops"] = []
        st.session_state.auto_find = True

    if st.session_state.pop("run_analysis", False):
        # Runs before the theme editors are drawn: otherwise the old Merge / Add sections stay on screen while it works and show twice.
        try:
            st.session_state.final = run_stage(cfg, edits, "export")
            st.session_state.result_dir = cfg["run_dir"]
            st.session_state.analysis_done = True
        except ValueError as e:
            st.error(str(e))
        else:
            st.rerun()

    go1, go2 = st.columns([1, 3])
    if go1.button("Find themes", type="primary") or "art" not in st.session_state and st.session_state.get("auto_find"):
        st.session_state.auto_find = True
        try:
            st.session_state.art = run_stage(cfg, edits, "themes")
        except ValueError as e:
            if edits["ops"]:            # edits typed earlier name themes that this search did not produce: start them again, never get stuck
                edits["ops"] = []
                try:
                    st.session_state.art = run_stage(cfg, edits, "themes")
                    st.warning(f"Earlier merge / rename / remove edits no longer matched the themes found now, so they were cleared. ({e})")
                except ValueError as e2:
                    st.error(str(e2))
            else:
                st.error(str(e))
    go2.caption("Themes are found from the answers themselves. Edit them below, then run the analysis.")

    art = st.session_state.get("art")
    if art and "themes" in art:
        table = themes_table(art)
        all_themes = [t for ts, _ in art["themes"].values() for t in ts]
        tv = st.session_state.get("table_version", 0)              # new key whenever the themes change: a kept widget would still show the old names
        st.dataframe(table.drop(columns=["_orig", "Keep"]), hide_index=True, width="stretch")

        names = list(table["Theme"])
        e1, e2, e3, e4 = st.columns(4)
        progress = st.container()                                  # full width, right under the buttons: where the progress of an edit shows
        with e1:
            st.markdown("**Rename a theme**")
            old_name = st.selectbox("Theme", names, key=f"rename_old_{tv}")
            new_name = st.text_input("New name", key=f"rename_new_{tv}")
            if st.button("Rename") and new_name.strip() and new_name.strip() != old_name:
                if name_taken(new_name, names, except_name=old_name):
                    st.warning(f"A theme called '{new_name.strip()}' already exists. Choose another name, or use Merge.")
                else:
                    edits["ops"].append(["rename", old_name, new_name.strip()])
                    try:
                        st.session_state.art = run_stage(cfg, edits, "themes", slot=progress)
                        st.rerun()
                    except ValueError as e:
                        edits["ops"].pop()
                        st.error(str(e))
        with e2:
            st.markdown("**Remove themes**")
            drop_pick = st.multiselect("Themes to remove", names, key=f"remove_pick_{tv}")
            if st.button("Remove") and drop_pick:
                new_ops = [["drop", n] for n in drop_pick]
                edits["ops"] += new_ops
                try:
                    st.session_state.art = run_stage(cfg, edits, "themes", slot=progress)
                    st.rerun()
                except ValueError as e:
                    edits["ops"] = edits["ops"][:-len(new_ops)]
                    st.error(str(e))
        with e3:
            st.markdown("**Merge themes**")
            pick = st.multiselect("Themes to merge", names, key=f"merge_pick_{tv}")
            merged_name = st.text_input("Name of the merged theme", key=f"merge_name_{tv}")
            if st.button("Merge"):
                built_in = {t["theme_name"] for t in all_themes if t.get("required") or t.get("seeded")}
                if len(pick) < 2:
                    st.warning("Pick at least two themes to merge.")
                elif built_in & set(pick):
                    st.warning("Built-in and added themes ({}) cannot be merged. Merge themes found in the answers, or use Remove."
                               .format(", ".join(sorted(built_in & set(pick)))))
                else:
                    edits["ops"].append(["merge", merged_name.strip() or None, pick])
                    try:
                        st.session_state.art = run_stage(cfg, edits, "themes", slot=progress)
                        st.rerun()
                    except ValueError as e:
                        edits["ops"].pop()
                        st.error(str(e))
        with e4:
            st.markdown("**Add a theme**")
            add_name = st.text_input("Theme name", key="add_name")
            add_kw = st.text_input("Words that signal it (optional, comma-separated)", key="add_kw")
            if st.button("Add") and add_name.strip():
                item = {"name": add_name.strip()}
                if add_kw.strip():
                    item["keywords"] = [k.strip() for k in add_kw.split(",") if k.strip()]
                if name_taken(add_name, names):
                    st.warning(f"A theme called '{add_name.strip()}' already exists. Rename or remove it first, or choose another name.")
                else:
                    before = edits["required"]
                    edits["required"] = before + [item]
                    try:
                        st.session_state.art = run_stage(cfg, edits, "themes", slot=progress)
                        st.rerun()
                    except ValueError as e:
                        edits["required"] = before         # a theme that could not be added must not stay in the list: it would clear every other edit
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
        if st.session_state.get("analysis_done"):
            st.success("Finished. Open the **Results** tab.")
            st.button("Go to results", on_click=lambda: st.session_state.update(main_tabs=TAB_RESULTS))
        if st.button("Run the analysis", type="primary"):
            st.session_state.analysis_done = False
            st.session_state.run_analysis = True          # the run happens at the top of the next script pass, before the editors are drawn
            st.rerun()

# ----------------------------------------------------------------------------- 3. results
with tab_results:
    s = st.session_state.get("settings")
    run_dir = st.session_state.get("result_dir") or (s["run_dir"] if s else None)
    f = latest_results(run_dir) if run_dir else None
    if not f:
        st.info("Run the analysis in the previous tab to see results here.")
        st.stop()
    for attempt in range(5):                       # the file may still be being written (a run just finished): wait and read again
        try:
            sheets = pd.read_excel(f, sheet_name=None)
            break
        except (zipfile.BadZipFile, EOFError):         # a half-written .xlsx raises either one
            if attempt == 4:
                st.error("The results file is not readable yet (it may still be saving). Reload the page in a moment.")
                st.stop()
            time.sleep(1)
    question_sheets = [n for n in sheets if n not in INTERNAL_SHEETS]          # one sheet per open-ended question
    st.caption(f"Results file: {f}")
    if Path(f).name != "aspect_sentiment_results.xlsx":
        st.warning("The standard results file is open in Excel, so this run was saved under a different name.")

    # headline numbers
    m = st.columns(4, gap="medium")
    n_answers = sum(len(sheets[q]) for q in question_sheets)
    themed = sum(int((sheets[q][[c for c in sheets[q].columns if c not in NOT_THEMES]].notna().any(axis=1)).sum())
                 for q in question_sheets)
    m[0].metric("Answers", f"{n_answers:,}")
    m[1].metric("With at least one theme", f"{themed / max(n_answers, 1):.0%}")
    m[2].metric("Questions", len(question_sheets))
    m[3].metric("Themes", len(sheets["themes"]))

    graphs = st.container()                       # filled below, once the question filter in the Answers section has been read
    st.subheader("Answers")
    f0, f1, f2 = st.columns([1, 2, 2])
    pick = f0.selectbox("Show questions", ["All"] + question_sheets, key="flt_questions")      # All = every question
    shown = question_sheets if pick == "All" else [pick]
    only = f1.multiselect("Show answers about", list(dict.fromkeys(c for q in shown for c in sheets[q].columns if c not in NOT_THEMES)), key="flt_themes")
    word = f2.text_input("Search the answer text", key="srch_text")
    with graphs:
        # one set of graphs per question, side by side (a new row after every three questions)
        for start in range(0, len(shown), 3):
            row = shown[start:start + 3]
            for col, qname in zip(st.columns(len(row), gap="medium"), row):
                with col:
                    d = sheets[qname]
                    theme_cols = [c for c in d.columns if c not in NOT_THEMES]
                    counts = pd.DataFrame({c: d[c].value_counts() for c in theme_cols}).T.reindex(
                        columns=["Positive", "Neutral", "Negative"]).fillna(0).astype(int)
                    counts = counts.assign(total=counts.sum(axis=1)).sort_values("total", ascending=False).drop(columns="total")
                    ov = d["Overall"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).astype(int)
                    st.subheader(qname)
                    sentiment_bar(int(ov["Positive"]), int(ov["Neutral"]), int(ov["Negative"]))
                    with st.container(border=True):
                        st.markdown("**How each theme is rated**")
                        st.bar_chart(counts, color=["#43a047", "#f9a825", "#e53935"],
                                     horizontal=True, height=max(220, 34 * len(counts)))

    # all shown questions in one table, with the question named in the first column
    combined = pd.concat([sheets[q].assign(Question=q) for q in shown], ignore_index=True)
    combined = combined[["Question"] + [c for c in combined.columns if c != "Question"]]
    view = combined
    if only:
        view = view[view[only].notna().any(axis=1)]
    if word:
        view = view[view["Answer"].astype(str).str.contains(word.strip(), case=False, na=False, regex=False)]
    st.write(f"{len(view):,} answers")
    st.dataframe(view.style.map(colour_cell), width="stretch", height=520, hide_index=True)

    st.divider()
    d1, d2 = st.columns(2)
    d1.download_button("Download the Excel workbook", data=Path(f).read_bytes(), file_name="survey_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
    with d2.expander("Run times"):
        st.dataframe(sheets["timings"], hide_index=True)
