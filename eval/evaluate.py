import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
sys.path.insert(0, "src")
from survey_nlp.config import Cfg
from survey_nlp.sentiment import decide

cfg = Cfg.load("config.yaml"); rd = Path(cfg["run_dir"])
df = pd.read_excel(rd / (sys.argv[1] if len(sys.argv) > 1 else "label_sheet_labeled.xlsx"))
df = df[df.gold_relevant.notna() & (df.gold_relevant.astype(str) != "")].copy()
df["gold_relevant"] = df.gold_relevant.astype(int)

print("== Relevance: sweep of assign.min_sim (semantic score only) ==")
best = (0, None)
for t in np.arange(0.0, 0.96, 0.025):
    pred = (df.sim >= t).astype(int)
    p, r, f, _ = precision_recall_fscore_support(df.gold_relevant, pred, average="binary", zero_division=0)
    print(f"min_sim={t:.3f}  P={p:.2f}  R={r:.2f}  F1={f:.2f}")
    if f > best[0]:
        best = (f, round(float(t), 3))
print("best min_sim:", best[1], "F1:", round(best[0], 3))

print("\n== ABSA on relevant, assigned pairs ==")
d = df[(df.kind == "assigned") & (df.gold_relevant == 1) & df.gold_sentiment.notna()].copy()
d["gold_sentiment"] = d.gold_sentiment.astype(str).str.strip().str.lower()
d["pred_argmax"] = np.array(["neg", "neu", "pos"])[d[["p_neg", "p_neu", "p_pos"]].to_numpy().argmax(1)]
print("macro-F1 (argmax):", round(f1_score(d.gold_sentiment, d.pred_argmax, average="macro"), 3))
print(confusion_matrix(d.gold_sentiment, d.pred_argmax, labels=["neg", "neu", "pos"]))

best = (0, None)
for thr in np.arange(0.40, 0.90, 0.05):
    pos, neg, neu = decide(d.p_pos, d.p_neg, thr, cfg["absa"]["allowance"])
    pos, neg = pos.to_numpy().astype(bool), neg.to_numpy().astype(bool)
    pred = np.where(pos & ~neg, "pos", np.where(neg & ~pos, "neg", "neu"))
    f = f1_score(d.gold_sentiment, pred, average="macro")
    print(f"evidence_threshold={thr:.2f}  macro-F1={f:.3f}")
    if f > best[0]:
        best = (f, round(float(thr), 2))
print("best evidence_threshold:", best[1])
