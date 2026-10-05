import argparse
import json
import sys
from pathlib import Path
from .config import Cfg
from .pipeline import run, ORDER


def _names(items):
    return ", ".join(d["name"] if isinstance(d, dict) else d for d in items)


def _ask_required(default):
    """Interactive: which themes MUST appear in the results. Everything else found in the data is added to them."""
    print("\nTHEMES")
    print("Type the themes that must appear in the results (comma-separated), e.g.  Packaging, Price, Flavour")
    print("Themes found automatically in the answers are added on top of these.")
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


def main():
    p = argparse.ArgumentParser("survey-nlp")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--until", default="export", choices=ORDER)
    r.add_argument("--force", action="store_true", help="ignore stage checkpoints (embedding cache is kept)")
    r.add_argument("--themes", help='required themes, comma-separated, e.g. "Packaging,Price" (skips the prompt)')
    r.add_argument("--prompt", action="store_true", help="ask for the required themes even if stdin is not a terminal")
    r.add_argument("--no-prompt", action="store_true", help="never ask questions; use themes.required from the config")
    a = p.parse_args()

    cfg = Cfg.load(a.config)
    required = cfg["themes"].get("required") or []
    last = Path(cfg["run_dir"]) / "required_themes.json"
    if a.themes is not None:
        required = [{"name": x.strip()} for x in a.themes.split(",") if x.strip()]
    elif not a.no_prompt and (a.prompt or sys.stdin.isatty()):
        if last.exists() and not required:
            required = json.loads(last.read_text())
        required = _ask_required(required)
    elif not a.no_prompt:
        print("(not an interactive terminal: skipping the theme prompt. Use --prompt, --themes \"A,B\", or themes.required in the config)")
    last.parent.mkdir(parents=True, exist_ok=True)
    last.write_text(json.dumps(required, indent=2))          # remembered as the default for the next run
    print("Required themes:", _names(required) or "none (themes are discovered only)")
    run(a.config, a.until, a.force, overrides={"themes": {"required": required}})


if __name__ == "__main__":
    main()
