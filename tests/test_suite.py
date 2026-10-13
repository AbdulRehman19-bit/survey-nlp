import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _suite():
    spec = importlib.util.spec_from_file_location("run_suite", ROOT / "eval" / "run_suite.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_suite_manifest_has_one_heldout_non_food_set_and_committed_inputs_exist():
    specs = json.loads((ROOT / "eval" / "suite" / "datasets.json").read_text())
    assert len({s["name"] for s in specs}) == len(specs)
    assert [s["name"] for s in specs if s["role"] == "heldout"] == ["app_reviews"]
    for s in specs:
        if s.get("input") and s["input"].startswith("eval/"):                 # generated files live in the repo, real surveys in data/
            assert (ROOT / s["input"]).exists(), s["input"]


def test_legacy_config_removes_exactly_the_new_theme_settings(tmp_path, monkeypatch):
    suite = _suite()
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({"seed": 1, "themes": {"n_themes": 5, "name_min_coverage": 0.25, "n_phrases": 6, "merge_sim": 0.85}}))
    out = yaml.safe_load(Path(suite.legacy_config("c.yaml", "runs/x")).read_text())
    assert out == {"seed": 1, "themes": {"n_themes": 5}}


def test_aggregate_helpers_ignore_missing_values():
    suite = _suite()
    assert suite.mean([1.0, None, 3.0]) == 2.0 and suite.mean([None]) is None
    assert suite.fmt(None) == "-" and suite.fmt(0.12345) == "0.123" and suite.fmt(3) == "3"


def test_every_internal_sheet_the_export_writes_is_hidden_from_the_apps_question_tabs():
    """The app treats every sheet not in INTERNAL_SHEETS as a question. A sheet added to the export (such as `quality`) and missed there
    made the results tab crash on numeric columns."""
    import re
    export = (ROOT / "src" / "survey_nlp" / "export.py").read_text(encoding="utf-8")
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    reserved = set(re.findall(r'"(\w+)"', re.search(r"reserved = \{(.*?)\}", export).group(1)))
    internal = set(re.findall(r'"(\w+)"', re.search(r"INTERNAL_SHEETS = \{(.*?)\}", app).group(1)))
    assert reserved - {"combined"} <= {n for n in internal} and "Combined" in internal and "quality" in internal


def test_theme_diagnostics_print_on_the_command_line_but_not_when_the_app_asks_for_quiet(capsys, monkeypatch):
    from survey_nlp.util import detail
    monkeypatch.delenv("SURVEY_NLP_QUIET", raising=False)
    detail("[themes] shown")
    assert "[themes] shown" in capsys.readouterr().out
    monkeypatch.setenv("SURVEY_NLP_QUIET", "1")
    detail("[themes] hidden")
    assert capsys.readouterr().out == ""
    assert 'os.environ.setdefault("SURVEY_NLP_QUIET"' in (ROOT / "app.py").read_text(encoding="utf-8")      # the app switches it on


def test_no_local_variable_in_the_export_shadows_a_function_it_imports():
    """`detail` (the detail sheet's DataFrame) once shadowed an imported logging function and broke every full export."""
    import ast
    tree = ast.parse((ROOT / "src" / "survey_nlp" / "export.py").read_text(encoding="utf-8"))
    imported = {(a.asname or a.name).split(".")[0] for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    local = {t.id for n in ast.walk(run) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assert not (imported & local), imported & local


def test_the_app_script_is_valid_python_and_shows_edit_progress_in_a_full_width_slot():
    import ast
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    ast.parse(src)                                                       # Streamlit would only show the SyntaxError in the browser
    assert src.count("slot=progress") == 4 and "progress = st.container()" in src      # apply / merge / add report under the buttons, not inside a column
    assert "wrap_lines=True" in src                                      # the progress lines wrap instead of running off the box


def test_the_apps_log_setting_hides_streamlits_file_watcher_tracebacks_for_modules_that_need_torchvision():
    import io
    import logging
    import types
    from streamlit.watcher import local_sources_watcher as watcher

    class Lazy(types.ModuleType):                         # like a transformers lazy module whose import needs the missing torchvision
        def __getattr__(self, name):
            raise ModuleNotFoundError("No module named 'torchvision'")
    log = logging.getLogger("streamlit.watcher.local_sources_watcher")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    log.addHandler(handler)
    old = log.level
    try:
        watcher.get_module_paths(Lazy("transformers.models.vilt.image_processing_vilt_fast"))
        assert "Examining the path" in buf.getvalue()                       # the noise exists without the setting
        buf.truncate(0)
        buf.seek(0)
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        line = 'logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.CRITICAL)'
        assert line in src
        exec(line, {"logging": logging})                                    # exactly what the app does
        watcher.get_module_paths(Lazy("transformers.models.vilt.image_processing_vilt_fast"))
        assert buf.getvalue() == ""
    finally:
        log.setLevel(old)
        log.removeHandler(handler)
