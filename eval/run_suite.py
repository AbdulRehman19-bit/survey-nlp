"""Runs every dataset of eval/suite/datasets.json and prints per-dataset and aggregate theme-quality numbers.

    python eval/run_suite.py --legacy --save eval/baseline_suite.json      # the behaviour before the theme-quality work (new settings removed)
    python eval/run_suite.py --compare eval/baseline_suite.json            # the current code against it
    python eval/run_suite.py --set themes.merge_k=2.0 --tag _k2 --compare eval/baseline_suite.json    # an experiment, in its own folders

Datasets with role "heldout" are reported on their own and are never part of the aggregate: they are not to be used to choose a default
or a threshold. Datasets with "aggregate": false (a fixed codebook, where nearly every theme is yours) are shown but not averaged.
Everything is local and deterministic; stages are cached in runs/_suite/<name><tag>/."""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from survey_nlp import pipeline, quality                       # noqa: E402
from survey_nlp.config import Cfg                              # noqa: E402

MANIFEST = ROOT / "eval" / "suite" / "datasets.json"
# Settings added by the theme-quality work. --legacy removes them, which switches every one of those features off: the result is the
# behaviour before that work (checked against eval/baseline_quality.json).
LEGACY_KEYS = ["name_min_coverage", "n_phrases", "phrase_mmr_lambda", "n_examples", "aspect_labels", "aspect_label_min_sim",
               "merge_sim", "merge_jaccard", "merge_name_sim", "filler_mode", "filler_k", "name_min_lift", "merge_k", "split_ratio", "split_k", "split_min_gain", "quality_bar"]
METRICS = ["n_themes", "n_found_themes", "n_below_bar", "n_weak_names", "n_mixed", "name_coverage_min", "name_coverage_mean", "coherence_npmi_mean",
           "largest_share", "gini", "general_share", "unassigned_share", "n_adjective_names", "n_near_duplicate_pairs"]
AGGREGATE = ["n_below_bar", "n_weak_names", "n_mixed", "name_coverage_min", "name_coverage_mean", "coherence_npmi_mean", "largest_share", "gini",
             "general_share", "n_adjective_names", "n_near_duplicate_pairs"]


def legacy_config(config, run_dir):
    raw = yaml.safe_load((ROOT / config).read_text())
    for k in LEGACY_KEYS:
        raw.get("themes", {}).pop(k, None)
    path = ROOT / run_dir / "legacy_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return str(path)


def run_one(spec, until, force, legacy, sets, tag):
    run_dir = f"runs/_suite/{spec['name']}{'_legacy' if legacy else ''}{tag}"
    overrides = {"run_dir": run_dir}
    for kv in sets:
        key, val = kv.split("=", 1)
        sec, sub = key.split(".", 1)
        overrides.setdefault(sec, {})[sub] = json.loads(val)
    if spec.get("input"):
        overrides["input"] = {"path": spec["input"]}
    config = legacy_config(spec["config"], run_dir) if legacy else str(ROOT / spec["config"])
    art = pipeline.run(config, until=until, force=force, overrides=overrides)
    cfg = Cfg.load(config)
    pipeline._deep_update(cfg.raw, overrides)
    report = quality.compute(art, cfg, pipeline.scopes(cfg, art["ingest"][0], art["segment"]))
    print(quality.summary_line(report))
    return report


def summary(report):
    return next(iter(report["scopes"].values()))["summary"]


def mean(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


def fmt(v):
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def tables(specs, now, before):
    role = {s["name"]: s for s in specs}
    for name, rep in now.items():
        new, old = summary(rep), (summary(before[name]) if name in before else {})
        print(f"\n== {name}  [{role[name]['role']}{'' if role[name].get('aggregate', True) else ', not aggregated'}]")
        print(f"{'metric':26}{'before':>12}{'now':>12}")
        for k in METRICS:
            print(f"{k:26}{fmt(old.get(k)):>12}{fmt(new.get(k)):>12}")
        names = lambda r: "; ".join(f"{t['theme']} ({t['size']}, cov {t['name_coverage']:.2f}{', mixed' if t['mixed'] else ''})"
                                    for t in next(iter(r['scopes'].values()))['themes'] if not t['general'])
        print("themes now   :", names(rep))
        if name in before:
            print("themes before:", names(before[name]))
    for label, pick in (("DEV AGGREGATE (mean over dev datasets)", lambda s: s["role"] == "dev" and s.get("aggregate", True)),
                        ("HELD-OUT (not used to choose defaults)", lambda s: s["role"] == "heldout")):
        chosen = [s["name"] for s in specs if pick(s) and s["name"] in now]
        if not chosen:
            continue
        have = [n for n in chosen if n in before]
        print(f"\n== {label}: {', '.join(chosen)}")
        print(f"{'metric':26}{'before':>12}{'now':>12}{'change':>12}")
        for k in AGGREGATE:
            a = mean([summary(before[n]).get(k) for n in have])
            b = mean([summary(now[n]).get(k) for n in chosen])
            print(f"{k:26}{fmt(a):>12}{fmt(b):>12}{('' if a is None or b is None else format(b - a, '+.3f')):>12}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--until", default="assign")
    p.add_argument("--force", action="store_true")
    p.add_argument("--legacy", action="store_true", help="switch the theme-quality settings off (the behaviour before that work)")
    p.add_argument("--save", help="write the numbers to this file")
    p.add_argument("--compare", help="compare with the numbers saved in this file")
    p.add_argument("--set", action="append", default=[], help="override a setting, e.g. --set themes.merge_k=2 (use with --tag)")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    specs = [s for s in json.loads(MANIFEST.read_text()) if not a.datasets or s["name"] in a.datasets]
    now = {s["name"]: run_one(s, a.until, a.force, a.legacy, a.set, a.tag) for s in specs}
    if a.save:
        Path(a.save).write_text(json.dumps(now, indent=2))
        print(f"\nsaved {a.save}")
    before = json.loads(Path(a.compare).read_text()) if a.compare else {}
    tables(specs, now, before)


if __name__ == "__main__":
    main()
