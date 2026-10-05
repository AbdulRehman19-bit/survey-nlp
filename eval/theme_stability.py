"""Re-discovers themes on 80% subsamples; reports how stable the centroids are (matched cosine)."""
import pickle, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "src")
from survey_nlp.config import Cfg
from survey_nlp.pipeline import scopes
from survey_nlp.themes import discover

cfg = Cfg.load("config.yaml"); rd = Path(cfg["run_dir"])
L = lambda n: pickle.loads((rd / f"{n}.pkl").read_bytes())
(long, _), clauses, E, T = L("ingest"), L("segment"), L("embed"), L("themes")
rng = np.random.default_rng(0)
for scope, m in scopes(cfg, long, clauses).items():
    sub, Es = clauses[m].reset_index(drop=True), E[m]
    _, C = T[scope]
    for i in range(5):
        idx = rng.choice(len(sub), int(0.8 * len(sub)), replace=False)
        cfg.raw["seed"] = i + 1
        t_i, C_i = discover(sub.iloc[idx].reset_index(drop=True), Es[idx], cfg)
        sim = C @ C_i.T
        r, c = linear_sum_assignment(-sim)
        print(f"[{scope}] run {i}: themes={len(t_i)}  mean matched cosine={sim[r, c].mean():.3f}  min={sim[r, c].min():.3f}")
