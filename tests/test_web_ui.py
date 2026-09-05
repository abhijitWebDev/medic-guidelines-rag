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


def test_requirements_carries_no_ui_or_ingest_weight():
    """requirements.txt is the deploy manifest. streamlit and pymupdf together
    are ~290 MB against Vercel's 250 MB unzipped limit."""
    reqs = (ROOT / "requirements.txt").read_text().lower()
    for heavy in ("streamlit", "pymupdf", "pyarrow", "pandas"):
        assert heavy not in reqs, f"{heavy} must not ship in the deploy bundle"


def test_static_page_is_reachable_from_the_installed_package():
    """STATIC is resolved relative to the package, so a wheel that omits the
    HTML would break the UI while every test still passed."""
    assert (api_mod.STATIC / "index.html").is_file()
    assert Path(api_mod.STATIC).name == "static"
