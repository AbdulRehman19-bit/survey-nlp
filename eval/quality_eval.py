"""Theme quality on the three reference datasets: save the numbers, or compare a run against saved ones.

    python eval/quality_eval.py --save eval/baseline_quality.json
    python eval/quality_eval.py --compare eval/baseline_quality.json

Each dataset runs in its own folder under runs/_quality/ (stages are cached there, saved clusters included), so your other runs are
untouched. Stages up to `assign` are enough for the report; --until export also writes the workbook with its `quality` sheet."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from survey_nlp import pipeline, quality                       # noqa: E402
from survey_nlp.config import Cfg                              # noqa: E402

DATASETS = {                                                    # name -> (config, input file or None for the config's own)
    # config.yaml (hand-curated for the older theme discovery) stops on a fresh run: its curation.seeds names a theme "Flavour"
    # that the current discovery does not produce. The Coco data is therefore run with the generic config.
    "coco": ("config_default.yaml", "data/coco_pops_VR.xlsx"),
    "beverage": ("config_beverage.yaml", None),
    "synthetic": ("config_default.yaml", "eval/suite/snack_drink_synthetic.xlsx"),
}
METRICS = ["n_themes", "n_found_themes", "n_below_bar", "n_mixed", "name_coverage_min", "name_coverage_mean", "coherence_npmi_mean",
           "largest_share", "gini", "general_share", "unassigned_share", "n_adjective_names", "n_near_duplicate_pairs"]


def run_one(name, until, force, sets=(), tag=""):
    config, inp = DATASETS[name]
    overrides = {"run_dir": f"runs/_quality/{name}{tag}"}
    for kv in sets:                                             # --set themes.merge_sim=0.7
        key, val = kv.split("=", 1)
        sec, sub = key.split(".", 1)
        overrides.setdefault(sec, {})[sub] = json.loads(val)
    if inp:
        overrides["input"] = {"path": inp}
    art = pipeline.run(str(ROOT / config), until=until, force=force, overrides=overrides)
    cfg = Cfg.load(str(ROOT / config))
    pipeline._deep_update(cfg.raw, overrides)
    long = art["ingest"][0]
    report = quality.compute(art, cfg, pipeline.scopes(cfg, long, art["segment"]))
    print(quality.summary_line(report))
    return report


def table(before, after):
    for ds in after:
        print(f"\n== {ds}")
        b = before.get(ds, {"scopes": {}})
        for scope, r in after[ds]["scopes"].items():
            old = (b["scopes"].get(scope) or {}).get("summary", {})
            print(f"{'metric':26}{'baseline':>12}{'now':>12}")
            for k in METRICS:
                print(f"{k:26}{str(old.get(k, '-')):>12}{str(r['summary'].get(k)):>12}")
            print("themes now:", "; ".join(f"{t['theme']} ({t['size']}, cov {t['name_coverage']:.2f}{', mixed' if t['mixed'] else ''})" for t in r["themes"] if not t["general"]))
            if old:
                print("themes baseline:", "; ".join(f"{t['theme']} ({t['size']}, cov {t['name_coverage']:.2f})" for t in b["scopes"][scope]["themes"] if not t["general"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", help="write the numbers to this file")
    p.add_argument("--compare", help="compare with the numbers saved in this file")
    p.add_argument("--datasets", nargs="*", default=list(DATASETS))
    p.add_argument("--until", default="assign")
    p.add_argument("--force", action="store_true", help="ignore stage checkpoints")
    p.add_argument("--set", action="append", default=[], help="override a setting, e.g. --set themes.merge_sim=0.7 (use with --tag)")
    p.add_argument("--tag", default="", help="suffix for the run folders, so an experiment does not touch the normal ones")
    a = p.parse_args()
    now = {ds: run_one(ds, a.until, a.force, a.set, a.tag) for ds in a.datasets}
    if a.save:
        Path(a.save).write_text(json.dumps(now, indent=2))
        print(f"\nsaved {a.save}")
    before = json.loads(Path(a.compare).read_text()) if a.compare else {}
    table(before, now)


if __name__ == "__main__":
    main()
