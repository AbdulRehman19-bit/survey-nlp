"""Creates label_sheet.xlsx: ~N assigned pairs (stratified by theme) + near-miss pairs just below the threshold."""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from survey_nlp.config import Cfg
from survey_nlp.pipeline import scopes

cfg = Cfg.load("config.yaml"); rd = Path(cfg["run_dir"])
L = lambda n: pickle.loads((rd / f"{n}.pkl").read_bytes())
(long, _), clauses, E, T, pairs, scored = L("ingest"), L("segment"), L("embed"), L("themes"), L("assign"), L("absa")
N_ASSIGNED, N_NEAR = 150, 50
n_themes_total = sum(len(t) for t, _ in T.values())
per = max(1, N_ASSIGNED // n_themes_total)

rows = []
for scope, m in scopes(cfg, long, clauses).items():
    themes, C = T[scope]
    sub = clauses[m].reset_index(drop=True)
    Es = E[m]
    sc = scored[scored.scope == scope]
    # one sample per theme (a loop, not groupby.apply: pandas 3 drops the grouping column inside apply)
    a = pd.concat([d.sample(min(per, len(d)), random_state=0) for _, d in sc.groupby("theme_idx")])
    rows.append(a.assign(kind="assigned")[["kind", "scope", "clause_id", "clause", "theme_idx", "theme", "sim", "p_pos", "p_neg", "p_neu"]])

    S = Es @ C.T
    lo, hi = cfg["assign"]["min_sim"] - 0.15, cfg["assign"]["min_sim"]
    have = set(zip(pairs[pairs.scope == scope].clause_id, pairs[pairs.scope == scope].theme_idx))
    ii, jj = np.nonzero((S >= lo) & (S < hi))
    cand = [(i, j) for i, j in zip(ii, jj) if (sub.clause_id.iloc[i], j) not in have]
    rng = np.random.default_rng(0)
    k = min(max(1, N_NEAR * len(themes) // n_themes_total), len(cand))
    pick = [cand[x] for x in rng.choice(len(cand), k, replace=False)] if k else []
    rows.append(pd.DataFrame({"kind": "near_miss", "scope": scope,
                              "clause_id": [sub.clause_id.iloc[i] for i, _ in pick],
                              "clause": [sub.clause.iloc[i] for i, _ in pick],
                              "theme_idx": [j for _, j in pick],
                              "theme": [themes[j]["theme_name"] for _, j in pick],
                              "sim": [S[i, j] for i, j in pick]}))
sheet = pd.concat(rows).sample(frac=1, random_state=1).reset_index(drop=True)
sheet["gold_relevant"] = ""      # fill: 1 if the clause is about the theme, else 0
sheet["gold_sentiment"] = ""     # fill (only when relevant): pos / neg / neu
out = rd / (sys.argv[1] if len(sys.argv) > 1 else "label_sheet.xlsx")
sheet.to_excel(out, index=False)
print("wrote", out, len(sheet), "rows")
