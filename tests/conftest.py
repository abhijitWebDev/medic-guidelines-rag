from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from rag_project import security
from rag_project.cache import reset_cache
from rag_project.config import get_settings
from rag_project.models import SourceDoc


@pytest.fixture(autouse=True)
def isolate_from_live_redis(monkeypatch):
    """No test may touch the real Upstash instance.

    .env carries a live REDIS_URL, so without this the suite reads and writes
    production keys: rate-limit counters survive between runs (making the
    limiter tests pass alone and fail together), and cached answers for test
    queries pile up in a store real users share. Tests that want Redis build
    their own client -- see tests/test_cache.py.
    """
    monkeypatch.setenv("REDIS_URL", "")
    get_settings.cache_clear()
    reset_cache()
    security.reset_rate_limits()
    yield
    get_settings.cache_clear()
    reset_cache()
    security.reset_rate_limits()


def _build_pdf(path: Path, blocks: list[tuple[str, float, bool]]) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60.0
    for text, size, bold in blocks:
        if y > 760:
            page = doc.new_page()
            y = 60.0
        page.insert_text((60, y), text, fontsize=size, fontname="hebo" if bold else "helv")
        y += size + 6
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def stg_pdf(tmp_path: Path) -> Path:
    """A miniature document shaped like an MOHFW Standard Treatment Guideline."""
    return _build_pdf(
        tmp_path / "stg_tb.pdf",
        [
            ("Pulmonary Tuberculosis", 18, True),
            ("Case Definition", 13, True),
            ("Tuberculosis is a communicable disease caused by Mycobacterium tuberculosis.", 10, False),
            ("Pulmonary TB refers to disease involving the lung parenchyma.", 10, False),
            ("Diagnosis", 13, True),
            ("Sputum smear microscopy remains the primary diagnostic modality at", 10, False),
            ("peripheral health facilities. NAAT is recommended as the initial test.", 10, False),
            ("Investigations", 11, True),
            ("Chest radiography is advised where smear results are negative but", 10, False),
            ("clinical suspicion remains high.", 10, False),
            ("Treatment", 13, True),
            ("Adults", 11, True),
            ("The intensive phase consists of two months of isoniazid, rifampicin,", 10, False),
            ("pyrazinamide and ethambutol administered daily.", 10, False),
            ("Referral Criteria", 13, True),
            ("Refer to a higher centre where drug resistance is suspected.", 10, False),
            ("12", 9, False),
        ],
    )


@pytest.fixture
def stg_doc(stg_pdf: Path) -> SourceDoc:
    return SourceDoc(
        doc_id="stg-tb",
        title="Pulmonary Tuberculosis",
        filename=stg_pdf.name,
        url="https://example.gov.in/tb.pdf",
        specialty="Respiratory",
        sha256="0" * 64,
    )


@pytest.fixture
def two_chapter_pdf(tmp_path: Path) -> Path:
    """Two chapters whose titles are sized INCONSISTENTLY, the later one smaller,
    each starting its own page -- the shape found in paediatrics.pdf."""
    doc = pymupdf.open()

    page = doc.new_page()
    y = 60.0
    for text, size, bold in [
        ("Dengue Fever", 18, True),
        ("Dengue is transmitted by Aedes mosquitoes and has a wide spectrum.", 10, False),
        ("Treatment", 14, True),
        ("Isotonic fluid therapy is started according to the haematocrit value.", 10, False),
    ]:
        page.insert_text((55, y), text, fontsize=size, fontname="hebo" if bold else "helv")
        y += size + 6

    page = doc.new_page()  # new chapter starts a new page, at a SMALLER size
    y = 60.0
    for text, size, bold in [
        ("EMPYEMA THORACIS", 16, True),
        ("Empyema thoracis is a collection of pus within the pleural cavity.", 10, False),
        ("Treatment", 14, True),
        ("Intercostal drainage is the mainstay of management in most children.", 10, False),
    ]:
        page.insert_text((55, y), text, fontsize=size, fontname="hebo" if bold else "helv")
        y += size + 6

    for _ in range(4):  # pad past the cover-page threshold
        doc.new_page().insert_text((55, 60), "Additional body text for padding.", fontsize=10)

    out = tmp_path / "two_chapters.pdf"
    doc.save(out)
    doc.close()
    return out
