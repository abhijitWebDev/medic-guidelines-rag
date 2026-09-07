from __future__ import annotations

import os
from pathlib import Path

import pymupdf
import pytest

from rag_project import security
from rag_project.cache import reset_cache
from rag_project.config import get_settings
from rag_project.db import get_db, reset_db
from rag_project.models import SourceDoc


@pytest.fixture(autouse=True)
def isolate_from_live_services(monkeypatch, tmp_path):
    """No test may touch the real Upstash instance or a real database.

    .env carries a live REDIS_URL, so without this the suite reads and writes
    production keys: rate-limit counters survive between runs (making the
    limiter tests pass alone and fail together), and cached answers for test
    queries pile up in a store real users share. Tests that want Redis build
    their own client -- see tests/test_cache.py.

    The same applies to accounts. DATABASE_URL is cleared and the SQLite
    fallback is pointed at a fresh file per test, so no test can read or write
    a real person's account or history, and no test inherits another's users.
    APP_PASSWORD is cleared for a subtler reason: it is deprecated as a
    credential but still forces the auth gate on, so leaving it set would make
    every test in the suite require a login.
    """
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setenv("APP_PASSWORD", "")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "accounts.db"))
    # No test may send email, and none may depend on whether the developer
    # happens to have SES configured. Both matter: .env carries live SES
    # credentials once someone sets them up, so without this a signup test
    # opens a real SMTP connection to Amazon -- and a suite that passes on a
    # laptop with no MAIL_FROM starts failing on one that has it, because
    # configuring mail is what turns verification on. Tests that want a mailer
    # set these themselves; see tests/test_email.py.
    monkeypatch.setenv("SES_SMTP_HOST", "")
    monkeypatch.setenv("SES_SMTP_USER", "")
    monkeypatch.setenv("SES_SMTP_PASSWORD", "")
    monkeypatch.setenv("MAIL_FROM", "")
    monkeypatch.setenv("PUBLIC_BASE_URL", "")
    # SQLite unless a throwaway Postgres is offered. Both backends share every
    # statement, so the suite passing on SQLite is most of the evidence -- but
    # only Postgres can prove the dialect translation, the type mapping and the
    # pool actually work, so it is worth being able to point the same tests at
    # one:  TEST_DATABASE_URL=postgresql://... uv run pytest
    postgres = os.environ.get("TEST_DATABASE_URL", "")
    monkeypatch.setenv("DATABASE_URL", postgres)
    # Pinned off, so that offering the suite a database changes *where rows go*
    # and nothing else. Without this, running with TEST_DATABASE_URL would also
    # switch the auth gate on (see Settings.auth_required) and every test of the
    # open instance would start failing on a 401. Tests about accounts turn it
    # on for themselves.
    monkeypatch.setenv("AUTH_ENABLED", "false")
    # Fixed so tokens issued in one request verify in the next; without it each
    # process invents a key, which is right for production and useless here.
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret")
    get_settings.cache_clear()
    reset_cache()
    reset_db()
    if postgres:
        # A shared server has no per-test tmp_path. Dropping the tables leaves
        # the next get_db() to recreate them, which is the same isolation the
        # SQLite path gets for free from a fresh file.
        db = get_db()
        db.execute("DROP TABLE IF EXISTS history")
        db.execute("DROP TABLE IF EXISTS users")
        reset_db()
    security.reset_rate_limits()
    security.reset_sessions()
    yield
    get_settings.cache_clear()
    reset_cache()
    reset_db()
    security.reset_rate_limits()
    security.reset_sessions()


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
