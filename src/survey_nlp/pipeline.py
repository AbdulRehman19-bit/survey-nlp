import os, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Cfg
from .util import sha, set_seed
from . import ingest, segment, embed, themes as th, assign as asg, sentiment, export

# (stage, parent stages, config sections that invalidate it)
PLAN = [
    ("ingest",  [],                  ["input", "embed.model"]),
    ("segment", ["ingest"],          ["segment"]),
    ("embed",   ["segment"],         ["embed"]),
    ("themes",  ["embed"],           ["themes", "seed"]),
    ("assign",  ["themes"],          ["assign"]),
    ("overall", ["ingest"],          ["overall"]),
    # evidence_threshold / allowance are applied at export (sentiment.aggregate), so they don't invalidate ABSA
    ("absa",    ["assign"],          ["absa.model", "absa.batch_size", "absa.max_length", "absa.quantize_cpu"]),
    ("export",  ["absa", "overall"], ["output"]),
]
ORDER = [p[0] for p in PLAN]


def _file_sig(p):
    s = os.stat(p)
    return f"{p}:{s.st_size}:{int(s.st_mtime)}"


def unique_texts(long):
    return long[long.valid].drop_duplicates("text_key").reset_index(drop=True)


def scopes(cfg, long, clauses):
    """scope name -> boolean mask over clauses. pooled: everything; per_question: that question's texts."""
    if cfg["themes"]["scope"] == "pooled":
        return {"ALL": np.ones(len(clauses), bool)}
    v = long[long.valid]
    return {q: clauses.text_key.isin(set(v[v.question == q].text_key)).to_numpy() for q in v.question.unique()}


def _compute(name, cfg, art, timings):
    long = art["ingest"][0] if "ingest" in art else None
    if name == "ingest":
        return ingest.run(cfg)
    if name == "segment":
        return segment.run(unique_texts(long), cfg)
    if name == "embed":
        return embed.run(art["segment"], cfg)
    if name == "themes":
        out = {}
        for s, m in scopes(cfg, long, art["segment"]).items():
            try:
                out[s] = th.discover(art["segment"][m].reset_index(drop=True), art["embed"][m], cfg)
            except ValueError as e:
                raise ValueError(f"[scope: {s}] {e}")
        return out
    if name == "assign":
        parts = []
        for s, m in scopes(cfg, long, art["segment"]).items():
            themes, C = art["themes"][s]
            a = asg.run(art["segment"][m].reset_index(drop=True), art["embed"][m], themes, C, cfg)
            a["scope"] = s
            parts.append(a)
        return pd.concat(parts, ignore_index=True)
    if name == "overall":
        return sentiment.run_overall(unique_texts(long), cfg)
    if name == "absa":
        return sentiment.run_absa(art["segment"], art["assign"], {s: t for s, (t, _) in art["themes"].items()}, cfg)
    if name == "export":
        return export.run(art, cfg, timings)


def _deep_update(d, u):
    for k, v in (u or {}).items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v


def run(config="config.yaml", until="export", force=False, overrides=None):
    cfg = Cfg.load(config)
    _deep_update(cfg.raw, overrides)                    # e.g. themes.required chosen at the prompt
    set_seed(cfg["seed"])
    rd = Path(cfg["run_dir"])
    rd.mkdir(parents=True, exist_ok=True)
    art, keys, timings = {}, {}, {}
    for name, parents, sections in PLAN:
        seed = [_file_sig(cfg["input"]["path"])] if name == "ingest" else [keys[p] for p in parents]
        keys[name] = sha("|".join(seed + [cfg.section_hash(*sections)]), 12)
        f, kf = rd / f"{name}.pkl", rd / f"{name}.key"
        if name != "export" and not force and f.exists() and kf.exists() and kf.read_text() == keys[name]:
            art[name] = pickle.loads(f.read_bytes())
            timings[name] = 0.0
            print(f"[{name}] cached")
        else:
            t = time.perf_counter()
            art[name] = _compute(name, cfg, art, timings)
            timings[name] = round(time.perf_counter() - t, 2)
            if name != "export":
                f.write_bytes(pickle.dumps(art[name]))
                kf.write_text(keys[name])
            print(f"[{name}] {timings[name]}s")
        if name == until:
            break
    print(f"total {sum(timings.values()):.1f}s")
    return art
