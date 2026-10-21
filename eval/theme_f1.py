"""Scores theme assignment against the human coder (sheet "Eleni" of data/Comparison.xlsx, same 500 rows as data/open_ends.xlsx).

    survey-nlp run --config config_default.yaml --input data/open_ends.xlsx --run-dir runs/x --until assign --no-prompt
    python eval/theme_f1.py runs/x [runs/y ...]        # one line per run
    python eval/theme_f1.py --diff runs/x runs/y        # also list the answer-theme pairs that differ between two runs (works for any file)

Each coder theme (the columns "Flavour (+)", "Flavour (-)", ... of one theme are merged) is paired with the discovered theme that overlaps it best
(presence F1), then the F1s are averaged weighted by the coder's count per theme. The pairing is chosen after seeing the result, which favours us a
little; use the numbers to compare runs, not as an absolute accuracy. One coder only."""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load(run):
    """(row per answer -> set of theme names, clause table, theme names) of a run directory that reached the assign stage."""
    run = Path(run)
    ing = pd.read_pickle(run / "ingest.pkl")[0]
    seg = pd.read_pickle(run / "segment.pkl")
    a = pd.read_pickle(run / "assign.pkl")
    names = [t["theme_name"] for t in pd.read_pickle(run / "themes.pkl")["ALL"][0]]
    return ing, seg, a, names


def presence(run, n):
    ing, seg, a, names = load(run)
    row = dict(zip(ing.text_key, ing.row_id))
    m = a.merge(seg[["clause_id", "text_key"]], on="clause_id")
    P = np.zeros((n, len(names)), bool)
    for tk, j in zip(m.text_key, m.theme_idx):
        if tk in row:
            P[row[tk], j] = True
    return P, m, names


def coder():
    g = pd.read_excel(ROOT / "data" / "Comparison.xlsx", sheet_name="Eleni")
    grp = {}
    for c in g.columns[2:]:
        grp.setdefault(re.sub(r"\s*\((\+|-|Neutral|1)\)\s*$", "", c).strip(), []).append(c)
    return {k: g[v].notna().any(axis=1).to_numpy() for k, v in grp.items()}, len(g)


def score(run):
    Y, n = coder()
    P, m, names = presence(run, n)
    tot = w = 0.0
    for y in Y.values():
        best = 0.0
        for j in range(len(names)):
            t = (P[:, j] & y).sum()
            best = max(best, 0.0 if t == 0 else 2 * t / (P[:, j].sum() + y.sum()))
        tot += best * y.sum()
        w += y.sum()
    ns = names.index("Not Specific") if "Not Specific" in names else None
    return tot / w, P.sum(1).mean(), (m.theme_idx == ns).mean() * 100 if ns is not None else float("nan")


def diff(a_run, b_run):
    _, seg, a, names = load(a_run)
    _, _, b, names_b = load(b_run)
    A = set(map(tuple, a[["clause_id", "theme_idx"]].values))
    B = set(map(tuple, b[["clause_id", "theme_idx"]].values))
    tx = dict(zip(seg.clause_id, seg.clause))
    print(f"added {len(B - A)}, removed {len(A - B)}")
    for cid, j in sorted(B - A):
        print("+", names_b[j], "|", tx[cid][:110])
    for cid, j in sorted(A - B):
        print("-", names[j], "|", tx[cid][:110])


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--diff":
        diff(args[1], args[2])
    else:
        print(f"{'run':<28} {'theme F1':>8} {'themes/answer':>13} {'Not Specific %':>14}")
        for r in args:
            f1, per, ns = score(r)
            print(f"{Path(r).name:<28} {f1:8.3f} {per:13.2f} {ns:14.1f}")
