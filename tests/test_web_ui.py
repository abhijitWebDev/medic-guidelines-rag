"""Contract tests across the Python/JavaScript boundary.

The pipeline panel is rendered in the browser now, so the rendering logic that
`tests/test_ui.py` used to cover directly lives in web/static/index.html and is
not reachable from pytest. What *is* worth protecting is the seam: the JS reads
keys out of `Response.trace`, and nothing in Python knows that. Rename a trace
key and the panel does not error -- it silently renders "not reached" or "—",
which is the worst possible failure for a panel whose entire job is to show you
which gate made a decision.

These tests read both sides and compare them.
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_project import api as api_mod
from rag_project.config import ROOT

ASSISTANT_SRC = (ROOT / "src" / "rag_project" / "assistant.py").read_text()
UI_SRC: str = (api_mod.STATIC / "index.html").read_text()


def _js_stage_keys() -> list[str]:
    """The stage keys listed in the UI's STAGES table, in order."""
    block = re.search(r"const STAGES = \[(.*?)\];", UI_SRC, re.S)
    assert block, "STAGES table not found in index.html"
    return re.findall(r'\["([a-z_]+)",', block.group(1))


def test_stage_keys_match_the_pipeline():
    """Every stage the pipeline records must have a row, in pipeline order."""
    python_stages = re.findall(r'trace\["stages"\]\.append\("([a-z_]+)"\)', ASSISTANT_SRC)
    assert python_stages, "no stages found in assistant.py"
    assert _js_stage_keys() == python_stages, (
        "UI STAGES has drifted from assistant.py; a mismatched key renders as "
        "'not reached' rather than failing"
    )


def test_ui_reads_only_trace_keys_python_writes():
    written = set(re.findall(r'trace\["([a-z_]+)"\]', ASSISTANT_SRC))
    written.add("cache")  # set by Assistant.ask on a cache hit, not in _run
    read = set(re.findall(r"trace\.([a-z_]+)", UI_SRC))
    assert read <= written, f"UI reads trace keys nothing writes: {sorted(read - written)}"


def test_degraded_is_surfaced_in_the_ui():
    """A fail-closed refusal and a considered one are indistinguishable to a
    reader unless the UI says so. cache.py depends on this distinction too."""
    assert "trace.degraded" in UI_SRC
    assert "degraded" in ASSISTANT_SRC


def test_citation_markers_are_highlighted_by_the_same_pattern():
    """The Python renderer is gone, but the marker grammar it defined is now
    duplicated in JS. Pin it: [C1] and [C2, C3] are markers, prose is not."""
    pattern = re.search(r"replace\(/\\\[\(C(.*?)\)\\\]/g", UI_SRC)
    assert pattern, "citation-marker regex not found in index.html"
    js = pattern.group(0)
    assert "C\\d+" in js and "," in js, "marker pattern must accept [C1] and [C2, C3]"


def test_ui_escapes_before_it_highlights():
    """Answers are model-authored, so escaping must happen before any span is
    introduced. If highlightCitations ever stops calling esc() first, an answer
    containing markup becomes markup."""
    fn = re.search(r"function highlightCitations\(.*?\n\}", UI_SRC, re.S)
    assert fn, "highlightCitations not found"
    body = fn.group(0)
    assert "esc(text)" in body, "highlightCitations must escape its input first"
    # esc() must be applied to the text before .replace introduces the span
    assert body.index("esc(text)") < body.index('class="cite"')


def test_no_stale_streamlit_references():
    """The Streamlit UI is gone; nothing should still point at it."""
    src = ROOT / "src" / "rag_project"
    assert not (src / "ui").exists(), "src/rag_project/ui was not removed"
    offenders = [
        p.relative_to(ROOT)
        for p in list(src.rglob("*.py")) + [ROOT / "pyproject.toml", ROOT / "README.md"]
        if p.exists() and re.search(r"streamlit|rag-ui|rag_project\.ui", p.read_text(), re.I)
    ]
    assert not offenders, f"stale Streamlit references in: {offenders}"


def test_runtime_dependencies_carry_no_ingest_weight():
    """[project.dependencies] is the deploy manifest -- Vercel installs exactly
    this set. The PDF parser is 64 MB and belongs in the `ingest` extra, which
    a deploy never installs; if it creeps back into the runtime list, every
    cold start pays for it. Vercel's unzipped bundle limit is 250 MB."""
    toml = (ROOT / "pyproject.toml").read_text()
    runtime = toml.split("[project.optional-dependencies]")[0].lower()
    for heavy in ("streamlit", "pymupdf", "pyarrow", "pandas"):
        assert heavy not in runtime, f"{heavy} must not be a runtime dependency"


def test_vercel_entrypoint_ships_the_data_the_query_path_needs():
    """HybridRetriever reads data/chunks at boot; without it every question is
    refused. A blacklist-only bundle config would not guarantee it ships."""
    import json
    cfg = json.loads((ROOT / "vercel.json").read_text())
    fn = cfg["functions"]["app.py"]
    for needed in ("src/**", "data/chunks/**", "data/index_manifest.json"):
        assert needed in fn["includeFiles"], f"{needed} missing from includeFiles"
    assert "data/embed_cache/**" in fn["excludeFiles"], "89 MB cache would ship"
    assert (ROOT / "app.py").is_file(), "Vercel entrypoint app.py is missing"


def test_static_page_is_reachable_from_the_installed_package():
    """STATIC is resolved relative to the package, so a wheel that omits the
    HTML would break the UI while every test still passed."""
    assert (api_mod.STATIC / "index.html").is_file()
    assert Path(api_mod.STATIC).name == "static"


def test_hidden_attribute_is_enforced_against_author_display_rules():
    """The browser hides `[hidden]` from its own stylesheet, at a specificity
    any author rule setting `display` outranks. The verification banner is
    styled `display:flex`, so without this guard `el.hidden = true` did
    nothing and the banner showed permanently -- including to accounts that
    had already confirmed, and on instances with no accounts at all.
    """
    css = UI_SRC.split("<style>", 1)[1].split("</style>", 1)[0]
    assert re.search(r"\[hidden\]\s*\{[^}]*display:\s*none\s*!important", css), (
        "no [hidden] guard: any element whose class sets `display` will ignore "
        "the attribute the JS toggles"
    )


def test_every_js_toggled_element_relies_on_that_guard():
    """Names the elements the rule above is protecting, so that removing it
    fails here with the list rather than silently somewhere in a browser."""
    toggled = set(re.findall(r'\$\("([a-z-]+)"\)\.hidden\s*=', UI_SRC))
    assert {"verify", "history", "signout"} <= toggled, (
        f"the set of hidden-toggled elements changed: {sorted(toggled)}"
    )


def test_the_composer_is_restored_when_verification_passes():
    """applyVerification runs again after confirming an address, so it has to
    re-enable the box as well as disable it -- otherwise verifying leaves the
    composer greyed out until a reload."""
    block = re.search(r"function applyVerification\(info\)\s*\{(.*?)\n\}", UI_SRC, re.S)
    assert block, "applyVerification not found"
    body = block.group(1)
    assert 'disabled = !verified' in body, "the disabled state is only ever set one way"
