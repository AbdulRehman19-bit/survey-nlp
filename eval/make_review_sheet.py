"""Creates runs/latest/review_sheet.xlsx: one sheet per open-ended question for manual checking.
Columns: respondent, answer, Overall, one column per theme (colour-coded), Evidence (clause behind each theme), Correct?, Comment."""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter
sys.path.insert(0, "src")
from survey_nlp.config import Cfg

cfg = Cfg.load("config.yaml"); rd = Path(cfg["run_dir"])
ql = cfg["output"].get("question_labels") or {}
long, _ = pickle.loads((rd / "ingest.pkl").read_bytes())
x = pd.read_excel(rd / "aspect_sentiment_results.xlsx", sheet_name=None)
res, pairs = x["results"], x["pairs"]
labels = np.array(["Negative", "Neutral", "Positive"])
pairs["lab"] = labels[pairs[["p_neg", "p_neu", "p_pos"]].to_numpy().argmax(1)]
pairs["ev"] = pairs.theme + " [" + pairs.lab + " " + pairs[["p_neg", "p_neu", "p_pos"]].max(axis=1).round(2).astype(str) + "]: " + pairs.clause
evidence = pairs.groupby("text_key").ev.apply(lambda s: "\n".join(dict.fromkeys(s)))

FILL = {"Positive": "C6EFCE", "Negative": "FFC7CE", "Neutral": "FFEB9C"}
path = rd / (sys.argv[1] if len(sys.argv) > 1 else "review_sheet.xlsx")
with pd.ExcelWriter(path, engine="openpyxl") as xw:
    for q in long.question.unique():
        name = ql.get(q, q)
        Lq = long[long.question == q].set_index("row_id")
        cols = [c for c in res.columns if c.startswith(name + " | ")]
        df = pd.DataFrame({"Respondent": res.respondent_id, name + " (answer)": res[name]})
        for c in cols:
            df[c.split(" | ", 1)[1]] = res[c]
        df["Evidence (clauses used)"] = res.row_id.map(Lq.text_key).map(evidence).fillna("")
        df["Counted as"] = np.where(res.row_id.map(Lq.valid), "answer", "non-answer (skipped)")
        df["Correct? (Y/N)"], df["Comment"] = "", ""
        df.to_excel(xw, sheet_name=name[:31], index=False)

wb = load_workbook(path)
for ws in wb:
    ws.freeze_panes = "C2"
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="305496")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    heads = [c.value for c in ws[1]]
    for j, h in enumerate(heads, 1):
        w = 55 if "answer" in h else 70 if h.startswith("Evidence") else 14 if h in ("Respondent", "Correct? (Y/N)") else 30 if h == "Comment" else 12
        ws.column_dimensions[get_column_letter(j)].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
            if c.value in FILL:
                c.fill = PatternFill("solid", fgColor=FILL[c.value])
wb.save(path)
print("wrote", path)
