import argparse
import json
import sys
from pathlib import Path
from .config import Cfg
from .pipeline import run, ORDER

HELP = """  Edit the themes (Enter or 'done' to accept):
    n 8                       change how many themes to keep
    add Price                 add a theme (optional signal words:  add Price: cost, expensive, cheap)
    remove Cough, Cherry      remove themes
    merge Easy + Bottle       merge themes (optional new name:  merge Easy + Bottle = Opening)
    rename Exercise = Refreshing
    reset                     undo all edits"""


def _names(items):
    return ", ".join(d["name"] if isinstance(d, dict) else d for d in items)


_BASE = {}                                      # overrides from --input / --run-dir, added to every run


def _empty():
    return {"n_themes": None, "required": [], "ops": []}


def _load_edits(path):
    try:
        e = json.loads(path.read_text())
        return {**_empty(), **e}
    except (OSError, ValueError):
        return _empty()


def _overrides(cfg, edits):
    """Turn the user's choices into config overrides for the themes section."""
    cur = dict(cfg["themes"].get("curation") or {})
    cur["ops"] = list(cur.get("ops") or []) + edits["ops"]
    th = {"required": edits["required"], "curation": cur}
    if edits["n_themes"]:
        th["n_themes"] = edits["n_themes"]
    return {**_BASE, "themes": th}


def _default_n(cfg, edits):
    return edits["n_themes"] or cfg["themes"].get("n_themes") or cfg["themes"].get("max_themes") or 10


def _ask_required(default):
    print("\nTHEMES")
    print("Themes are found automatically from the answers. You can also name themes that must be included,")
    print("for example  Packaging, Price  (they are added to the discovered ones).")
    raw = input(f"Required themes [{_names(default) or 'none'}]  (Enter = keep, - = none): ").strip()
    if raw == "":
        return default
    if raw == "-":
        return []
    out = []
    for name in [x.strip() for x in raw.split(",") if x.strip()]:
        kw = input(f"  Extra words that signal '{name}' (comma-separated, Enter = just the name): ").strip()
        item = {"name": name}
        if kw:
            item["keywords"] = [k.strip() for k in kw.split(",") if k.strip()]
        out.append(item)
    return out


def _show(art):
    for scope, (themes, _) in art["themes"].items():
        found = themes[0].get("candidates_found", len(themes)) if themes else 0
        print(f"\nTHEMES FOUND{'' if scope == 'ALL' else ' for ' + scope}: {found} candidate topics, showing {len(themes)}")
        for i, t in enumerate(themes, 1):
            size = t["frequency"] if t["frequency"] else "added"
            print(f"  {i:>2}. {t['theme_name']:<18} {str(size):>6}   {', '.join((t.get('curated_keywords') or t['keywords'])[:6])}")
    return [t["theme_name"] for ts, _ in art["themes"].values() for t in ts]


def _apply(line, edits, names):
    """Apply one typed command to `edits`. Returns an error message, or None when it worked."""
    low = {n.lower(): n for n in names}

    def find(x):
        x = x.strip()
        return low.get(x.lower())

    cmd, _, rest = line.strip().partition(" ")
    cmd, rest = cmd.lower(), rest.strip()
    if cmd in ("n", "number"):
        if not rest.isdigit() or not 2 <= int(rest) <= 50:
            return "Give a number between 2 and 50, for example:  n 8"
        edits["n_themes"] = int(rest)
    elif cmd == "add":
        name, _, kw = rest.partition(":")
        name = name.strip()
        if not name:
            return "Say which theme to add, for example:  add Price"
        item = {"name": name}
        if kw.strip():
            item["keywords"] = [k.strip() for k in kw.split(",") if k.strip()]
        edits["required"] = [r for r in edits["required"] if (r["name"] if isinstance(r, dict) else r).lower() != name.lower()] + [item]
    elif cmd in ("remove", "drop", "delete"):
        for x in [p for p in rest.split(",") if p.strip()]:
            req = [r for r in edits["required"] if (r["name"] if isinstance(r, dict) else r).lower() == x.strip().lower()]
            if req:
                edits["required"].remove(req[0])
            elif find(x):
                edits["ops"].append(["drop", find(x)])
            else:
                return f"No theme called '{x.strip()}'. Current themes: {', '.join(names)}"
    elif cmd == "merge":
        group, _, new = rest.partition("=")
        parts = [find(p) for p in group.split("+")]
        if len(parts) < 2 or None in parts:
            return "Name at least two existing themes, for example:  merge Easy + Bottle = Opening"
        edits["ops"].append(["merge", new.strip() or None, parts])
    elif cmd == "rename":
        old, _, new = rest.partition("=")
        if not find(old) or not new.strip():
            return "Use:  rename OldName = NewName"
        edits["ops"].append(["rename", find(old), new.strip()])
    elif cmd == "reset":
        edits.update(_empty())
    else:
        return "Unknown command.\n" + HELP
    return None


def _review(a, cfg, edits):
    """Show the discovered themes and let the user change them until they accept."""
    print(HELP)
    good = json.dumps(edits)                                  # last set of edits that worked
    while True:
        try:
            art = run(a.config, "themes", False, overrides=_overrides(cfg, edits))
        except ValueError as e:                               # e.g. a merged theme no longer exists after lowering n
            print(f"\nThat edit cannot be applied, so it was undone: {e}")
            edits.update(json.loads(good))
            art = run(a.config, "themes", False, overrides=_overrides(cfg, edits))
        good = json.dumps(edits)
        names = _show(art)
        line = input("\nEdit themes (add / remove / merge / rename / n <count> / reset / done): ").strip()
        if line.lower() in ("", "done", "ok", "yes", "y"):
            return
        before = json.dumps(edits)
        err = _apply(line, edits, names)
        if err:
            print(err)
            edits.update(json.loads(before))


def main():
    p = argparse.ArgumentParser("survey-nlp")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--until", default="export", choices=ORDER)
    r.add_argument("--force", action="store_true", help="ignore stage checkpoints (embedding cache is kept)")
    r.add_argument("--input", help="survey file to analyse (overrides input.path); results go to runs/<file name>/")
    r.add_argument("--run-dir", help="where to write results (default runs/<input file name> with --input)")
    r.add_argument("--n-themes", type=int, help="how many themes to keep (default from config, 10)")
    r.add_argument("--themes", help='required themes, comma-separated, e.g. "Packaging,Price" (skips the prompts)')
    r.add_argument("--prompt", action="store_true", help="ask about themes even if stdin is not a terminal")
    r.add_argument("--no-prompt", action="store_true", help="never ask questions; use the config and saved choices")
    a = p.parse_args()

    cfg = Cfg.load(a.config)
    if a.input:
        _BASE["input"] = {"path": a.input}
        _BASE["run_dir"] = a.run_dir or f"runs/{Path(a.input).stem}"
    elif a.run_dir:
        _BASE["run_dir"] = a.run_dir
    if "run_dir" in _BASE:
        cfg.raw["run_dir"] = _BASE["run_dir"]
    last = Path(cfg["run_dir"]) / "theme_edits.json"
    edits = _load_edits(last)
    interactive = not a.no_prompt and a.themes is None and (a.prompt or sys.stdin.isatty())

    if a.themes is not None:
        edits = _empty()
        edits["required"] = [{"name": x.strip()} for x in a.themes.split(",") if x.strip()]
    if a.n_themes:
        edits["n_themes"] = a.n_themes
    if interactive:
        if edits["ops"] or edits["required"] or edits["n_themes"]:
            summary = f"{len(edits['ops'])} edits, required: {_names(edits['required']) or 'none'}, count: {edits['n_themes'] or 'default'}"
            if input(f"Reuse your previous theme choices ({summary})? [Y/n]: ").strip().lower().startswith("n"):
                edits = _empty()
        if not edits["ops"]:
            edits["required"] = _ask_required(edits["required"])
            n = input(f"How many themes to keep? [{_default_n(cfg, edits)}]  (Enter = keep): ").strip()
            if n.isdigit() and 2 <= int(n) <= 50:
                edits["n_themes"] = int(n)
        _review(a, cfg, edits)
    last.parent.mkdir(parents=True, exist_ok=True)
    last.write_text(json.dumps(edits, indent=2))              # remembered as the default for the next run
    print(f"\nThemes: keep {_default_n(cfg, edits)}; required: {_names(edits['required']) or 'none'}; {len(edits['ops'])} edits")
    run(a.config, a.until, a.force, overrides=_overrides(cfg, edits))


if __name__ == "__main__":
    main()
