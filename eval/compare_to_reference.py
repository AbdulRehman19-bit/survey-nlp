"""Score a survey_nlp results workbook against a human-coded reference (Eleni's sheet).

    python compare_to_reference.py --pipeline survey_results.xlsx --reference Comparison.xlsx [--sheet Eleni] [--out report.xlsx]

Why not the "% same" cell in Comparison.xlsx: it counts blank-vs-blank cells as "same", and about 97% of all cells are blank
for both coders, so it stays high whatever the model does. This script scores what matters instead:
  1. overall sentiment: accuracy, macro-F1, confusion matrix
  2. per theme: was the theme detected at all (precision / recall / F1 on presence)
  3. per theme: when both mention it, do they agree on Positive / Negative

Pipeline sheets are read per question (sheet names "Like" / "Dislike", in the same order as the reference rows).
Themes are compared in Eleni's codebook groups; GROUPS maps each group to the pipeline theme names that mean the same thing.
If you switch to the codebook themes (config_beverage.yaml) the names match and the default mapping applies.
"""
import argparse
import numpy as np
import pandas as pd

# Eleni group -> (her columns that mean Positive, her columns that mean Negative, her columns that mean Neutral)
REF = {
    "Flavour":      (["Flavour (+)"], ["Flavour (-)"], ["Flavour (Neutral)"]),
    "Sweetness":    (["Sweetness (+)"], ["Sweetness (-)"], ["Sweetness (Neutral)"]),
    "Mouthfeel":    (["Mouth (+)"], ["Mouth (-)"], ["Mouth (Neutral)"]),
    "Aftertaste":   (["Aftertaste (+)"], ["Aftertaste (-)"], ["Aftertaste (Neutral)"]),
    "Aroma":        (["Aroma (+)"], ["Aroma (-)"], []),
    "Artificial":   ([], ["Artificial"], []),
    "Medicine":     (["Drug (+)", "Cough syrup (+)"], ["Drug (-)", "Cough syrup (-)"], ["Drug (Neutral)", "Cough syrup (Neutral)"]),
    "Packaging":    (["Packaging (+)"], ["Packaging (-)"], ["Packaging (Neutral)"]),
    "Design":       (["Design (+)"], ["Design (-)"], ["Design (Neutral)"]),
    "Colour":       (["Colour (+)"], ["Colour (-)"], ["Colour (Neutral)"]),
    "Energy":       (["Energy (+)"], ["Energy (-)"], ["Energy (Neutral)"]),
    "Refreshing":   (["Refreshing"], [], []),
    "Price":        (["Price  (+)"], ["Price (-)"], []),
    "Brand":        (["Brand (+)"], ["Brand (-)"], []),
    "Non-specific": (["Non-specific(+)"], ["Non-specific(1)"], []),
}
# pipeline theme names (any run) that mean the same as an Eleni group; names equal to the group are matched automatically
GROUPS = {
    "Flavour": ["Flavour", "Taste", "Orange Flavour", "Cherry Flavour"],
    "Artificial": ["Artificial", "Artificial Taste"],
    "Medicine": ["Medicine", "Cough Syrup"],
    "Packaging": ["Packaging", "Attractive Packaging", "Quality Bottle"],
    "Design": ["Design", "Attractive Packaging", "Quality Bottle"],
    "Colour": ["Colour"],
    "Energy": ["Energy", "Exercise"],
    "Refreshing": ["Refreshing", "Exercise"],
}
WORD = {"positive": 1, "negative": 2, "neutral": 0, 1: 1, 2: 2, 0: 0}


def ref_tables(df):
    pres, pol = {}, {}
    for g, (p, n, u) in REF.items():
        P = df[[c for c in p if c in df]].notna().any(axis=1).to_numpy() if p else np.zeros(len(df), bool)
        N = df[[c for c in n if c in df]].notna().any(axis=1).to_numpy() if n else np.zeros(len(df), bool)
        U = df[[c for c in u if c in df]].notna().any(axis=1).to_numpy() if u else np.zeros(len(df), bool)
        pres[g] = P | N | U
        pol[g] = np.where(P & N, "mixed", np.where(P, "Positive", np.where(N, "Negative", np.where(U, "Neutral", ""))))
    return pres, pol


def score(ref, mine, rows):
    """All metrics on the row subset `rows` (boolean mask over the 500 rows). Returns (overall dict, confusion, theme table)."""
    ref, mine = ref[rows].reset_index(drop=True), mine[rows].reset_index(drop=True)
    y, p = ref["Sentiment"].astype(str), mine["Overall"].astype(str)
    conf = pd.crosstab(y, p, rownames=["Eleni"], colnames=["Pipeline"], margins=True)
    f1s = []
    for l in sorted(set(y) | set(p)):
        tp = ((y == l) & (p == l)).sum()
        pr, rc = tp / max((p == l).sum(), 1), tp / max((y == l).sum(), 1)
        f1s.append(2 * pr * rc / max(pr + rc, 1e-9))
    pres, pol = ref_tables(ref)
    cols = {c.lower(): c for c in mine.columns}
    rows_ = []
    for g in REF:
        names = [n for n in GROUPS.get(g, [g]) if n.lower() in cols]
        if g.lower() in cols and g not in names:
            names.append(g)
        truth = pres[g]
        if names:
            sub = mine[[cols[n.lower()] for n in names]]
            pred = sub.notna().any(axis=1).to_numpy()
            code = sub.apply(lambda r: next((WORD.get(str(v).lower()) for v in r if pd.notna(v) and str(v).lower() in WORD), None), axis=1)
            code = code.map({1: "Positive", 2: "Negative", 0: "Neutral"}).to_numpy()
        else:
            pred, code = np.zeros(len(ref), bool), np.array([None] * len(ref))
        tp = (pred & truth).sum()
        pr, rc = tp / max(pred.sum(), 1), tp / max(truth.sum(), 1)
        both = pred & truth & np.isin(pol[g], ["Positive", "Negative", "Neutral"])
        agree = (code[both] == pol[g][both]).mean() if both.any() else np.nan
        rows_.append(dict(theme=g, n_reference=int(truth.sum()), n_pipeline=int(pred.sum()), precision=pr, recall=rc,
                          F1=2 * pr * rc / max(pr + rc, 1e-9), polarity_agree=agree, n_both=int(both.sum()),
                          pipeline_columns=", ".join(names) or "(none)"))
    t = pd.DataFrame(rows_)
    w = t.n_reference
    ok = t.polarity_agree.notna()
    summary = dict(accuracy=(y == p).mean(), macro_F1=np.mean(f1s), weighted_theme_F1=(t.F1 * w).sum() / w.sum(),
                   polarity_agreement=np.nansum(t.polarity_agree * t.n_both) / t.n_both[ok].sum(),
                   mean_themes_per_answer=float(sum(pres[g] for g in REF).mean()))
    return summary, conf, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--sheet", default="Eleni")
    ap.add_argument("--questions", nargs="+", default=["Like", "Dislike"])
    ap.add_argument("--split", choices=["all", "even", "odd"], default="all",
                    help="score only the even- or odd-numbered rows (calibration on even, held-out report on odd)")
    ap.add_argument("--out", default="comparison_report.xlsx")
    a = ap.parse_args()

    ref = pd.read_excel(a.reference, sheet_name=a.sheet)
    # with layout auto the sheets keep blank rows for the other question's answers: only rows with an answer count
    mine = pd.concat([pd.read_excel(a.pipeline, sheet_name=s).dropna(subset=["Answer"]) for s in a.questions], ignore_index=True)
    assert len(ref) == len(mine), f"row count differs: reference {len(ref)}, pipeline {len(mine)}"
    same = (ref["Text"].astype(str).str.strip().to_numpy() == mine["Answer"].astype(str).str.strip().to_numpy()).mean()
    assert same > 0.99, f"texts do not line up row by row ({same:.0%} equal); sort both the same way first"

    idx = np.arange(len(ref))
    masks = {"all": idx >= 0, "even": idx % 2 == 0, "odd": idx % 2 == 1}
    summary, conf, t = score(ref, mine, masks[a.split])
    print(f"[{a.split} rows: {masks[a.split].sum()}]")
    print(f"OVERALL SENTIMENT  accuracy {summary['accuracy']:.3f}   macro-F1 {summary['macro_F1']:.3f}")
    print(conf, "\n")
    print(t.round(2).to_string(index=False))
    print(f"\nTHEME DETECTION weighted F1 {summary['weighted_theme_F1']:.3f}   "
          f"POLARITY agreement (cells both coded) {summary['polarity_agreement']:.3f}   "
          f"themes per answer (reference) {summary['mean_themes_per_answer']:.2f}")
    if a.split != "all":                                   # tuning on the even rows must not show the held-out half
        return
    per = {k: score(ref, mine, m) for k, m in masks.items()}
    print("\nSPLITS        " + "".join(f"{k:>9}" for k in per))
    for key in ["accuracy", "macro_F1", "weighted_theme_F1", "polarity_agreement"]:
        print(f"{key:<19}" + "".join(f"{per[k][0][key]:9.3f}" for k in per))
    with pd.ExcelWriter(a.out) as xw:
        t.to_excel(xw, sheet_name="themes", index=False)
        conf.to_excel(xw, sheet_name="overall_confusion")
        pd.DataFrame({k: v[0] for k, v in per.items()}).to_excel(xw, sheet_name="splits")
        pd.DataFrame({k: v[2].set_index("theme").F1 for k, v in per.items()}).to_excel(xw, sheet_name="theme_F1_by_split")
    print("saved", a.out)


if __name__ == "__main__":
    main()
