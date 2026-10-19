"""Suggest how many themes to ask for.

The themes are found again for each count in a range (the cleaning, cutting and embedding are done once), every set is scored with the
quality report (`quality.compute`), and the best count is picked by a fixed rule. Nothing here is saved: the sweep never writes
clusters or results, so it cannot change a later run."""
import pandas as pd
from . import pipeline, quality
from .config import Cfg
from .util import set_seed

COLUMNS = ["themes", "mixed", "weak_names", "below_bar", "near_duplicates", "name_coverage_min", "name_coverage_mean", "coherence", "smallest_theme",
           "not_specific", "unmatched", "largest_theme", "gini", "smallest_share"]


def default_range(n_usable: int, lo: int = 4, hi_cap: int = 16):
    """From 4 up to one theme per 20 usable clauses, at most 16 (more than that is rarely readable)."""
    return lo, max(lo + 1, min(hi_cap, n_usable // 20))


def _row(n, report):
    summ, small, share = [], [], []
    for rep in report["scopes"].values():
        summ.append(rep["summary"])
        small.append(min([t["size"] for t in rep["themes"] if not t["general"]] or [0]))
        share.append(small[-1] / max(rep["summary"]["n_clauses"], 1))
    mean = lambda k: float(sum(s[k] for s in summ) / len(summ))
    return {"asked": n, "themes": int(sum(s["n_themes"] for s in summ) / len(summ)),
            "mixed": int(sum(s["n_mixed"] for s in summ)), "weak_names": int(sum(s["n_weak_names"] for s in summ)),
            "below_bar": int(sum(s["n_below_bar"] for s in summ)), "near_duplicates": int(sum(s["n_near_duplicate_pairs"] for s in summ)),
            "name_coverage_min": min(s["name_coverage_min"] for s in summ), "name_coverage_mean": mean("name_coverage_mean"),
            "coherence": mean("coherence_npmi_mean"), "smallest_theme": int(min(small)),
            "not_specific": mean("general_share"), "unmatched": mean("unassigned_share"), "largest_theme": mean("largest_share"), "gini": mean("gini"), "smallest_share": float(max(share))}


def sweep(config_path, overrides_for, lo=None, hi=None, progress=None) -> pd.DataFrame:
    """One row per asked-for count. `overrides_for(n)` gives the config overrides for n themes (the caller knows about required themes
    and the built-in general-comment theme). `progress(done, total, n)` is called after each count."""
    art = pipeline.run(config_path, "embed", overrides=overrides_for(lo or 4))        # cached stages are reused; themes are not computed here
    long = art["ingest"][0]
    if lo is None or hi is None:
        n_ok = int((art["segment"]["clause"].str.split().str.len() >= Cfg.load(config_path)["themes"]["min_words"]).sum())
        lo, hi = default_range(n_ok)
    counts = list(range(lo, hi + 1))
    rows = []
    for i, n in enumerate(counts):
        cfg = Cfg.load(config_path)
        pipeline._deep_update(cfg.raw, overrides_for(n))
        cfg.raw["themes"]["freeze"] = False                                           # never read or write the saved clusters
        set_seed(cfg["seed"])
        try:
            a = dict(art)
            a["themes"] = pipeline._compute("themes", cfg, a, {})
            a["assign"] = pipeline._compute("assign", cfg, a, {})
            rows.append(_row(n, quality.compute(a, cfg, pipeline.scopes(cfg, long, a["segment"]))))
        except ValueError:                                                            # too few clauses for this many themes
            pass
        if progress:
            progress(i + 1, len(counts), n)
    return pd.DataFrame(rows, columns=["asked"] + COLUMNS)


def recommend(df: pd.DataFrame, tolerance: float = 0.10, min_size: int = 12, max_unspecific: float = 0.25, max_smallest_share: float = 0.15):
    """(count, reason). Fewer themes always look tighter, so the tightest set alone would always be the smallest count. But with few themes
    every theme is broad (even the smallest holds a large part of the file) and many answers fit none and fall into Not Specific. The rule:
      1. clean sets only: no theme mixed, weakly named or under the name bar (a weakly named theme is usually also under the bar: it counts once), no two themes near-duplicates, no theme smaller than
         `min_size` clauses, no more than `max_unspecific` of the clauses in Not Specific or without a theme, and the smallest theme
         no more than `max_smallest_share` of all clauses (otherwise every theme is too broad to be useful);
      2. score = coherence + mean name coverage - the share of clauses that are Not Specific or have no theme - the smallest theme's share
         of the clauses; the best clean set sets the bar;
      3. every clean set within `tolerance` of the best score is about as good, and of those the one with the MOST themes is chosen:
         the same quality, with more detail for the reader.
    When no set is clean, the one with the fewest problems (then the best score)."""
    if df.empty:
        return None, "No count could be tested: the file has too few usable answers."
    zero = pd.Series(0.0, index=df.index)
    left = df.get("not_specific", zero) + df.get("unmatched", zero)               # share of clauses no specific theme explains
    broad = df.get("smallest_share", zero)                                         # share of all clauses in the smallest theme
    d = df.assign(left=left, broad=broad,
                  problems=df.mixed + df[["weak_names", "below_bar"]].max(axis=1) + df.near_duplicates + (df.smallest_theme < min_size).astype(int)
                  + (left > max_unspecific).astype(int) + (broad > max_smallest_share).astype(int),
                  score=df.coherence + df.name_coverage_mean - left - broad)
    clean = d[d.problems == 0]
    if len(clean):
        top = clean.score.max()
        good = clean[clean.score >= top - tolerance]
        best = good.sort_values("asked").iloc[-1]
        return int(best.asked), (f"{int(best.asked)} themes: none is mixed or badly named, none repeats another, the smallest has "
                                 f"{int(best.smallest_theme)} answers and {best.left:.0%} of the clauses are Not Specific or without a theme. "
                                 f"{len(clean)} counts were clean; {len(good)} of them score about as well as the best, and this is the one with the most detail.")
    best = d.sort_values(["problems", "score"], ascending=[True, False]).iloc[0]
    return int(best.asked), (f"{int(best.asked)} themes: no count was free of problems; this one has the fewest "
                             f"({int(best.problems)}: mixed, badly named, repeated, too broad, very small or too many unexplained clauses).")
