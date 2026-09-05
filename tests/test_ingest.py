from __future__ import annotations

from pathlib import Path

from rag_project.ingest.chunk import chunk_document, embed_text
from rag_project.ingest.parse import parse_pdf
from rag_project.models import SourceDoc


def test_sections_keep_their_parent_heading(stg_pdf: Path):
    """Regression: the H1 was once dropped as 'furniture', orphaning children."""
    parsed = parse_pdf(stg_pdf, "stg-tb", "Pulmonary Tuberculosis")
    paths = [s.path for s in parsed.sections]
    assert all(p.startswith("Pulmonary Tuberculosis") for p in paths), paths


def test_heading_hierarchy_nests_and_pops(stg_pdf: Path):
    parsed = parse_pdf(stg_pdf, "stg-tb", "Pulmonary Tuberculosis")
    paths = [s.path for s in parsed.sections]
    assert "Pulmonary Tuberculosis > Diagnosis > Investigations" in paths
    assert "Pulmonary Tuberculosis > Treatment > Adults" in paths
    # Referral Criteria is a sibling of Treatment, not a child of it.
    assert "Pulmonary Tuberculosis > Referral Criteria" in paths


def test_short_but_complete_sections_survive(stg_pdf: Path, stg_doc: SourceDoc):
    """Regression: a token floor once deleted whole short sections."""
    chunks = chunk_document(parse_pdf(stg_pdf, "stg-tb", stg_doc.title), stg_doc)
    assert any("Referral Criteria" in c.section_path for c in chunks)
    assert any("Investigations" in c.section_path for c in chunks)


def test_page_numbers_are_stripped(stg_pdf: Path, stg_doc: SourceDoc):
    chunks = chunk_document(parse_pdf(stg_pdf, "stg-tb", stg_doc.title), stg_doc)
    assert not any(c.text.strip() == "12" for c in chunks)


def test_chunks_never_straddle_sections(stg_pdf: Path, stg_doc: SourceDoc):
    chunks = chunk_document(parse_pdf(stg_pdf, "stg-tb", stg_doc.title), stg_doc)
    for c in chunks:
        assert c.section_path  # every chunk is attributable to exactly one section
    assert len({c.chunk_id for c in chunks}) == len(chunks), "chunk_ids must be unique"


def test_embedded_text_carries_context_the_body_lacks(stg_pdf: Path, stg_doc: SourceDoc):
    """The Treatment chunk never says 'tuberculosis' -- the prefix must."""
    chunks = chunk_document(parse_pdf(stg_pdf, "stg-tb", stg_doc.title), stg_doc)
    treat = next(c for c in chunks if "Treatment" in c.section_path)
    assert "tuberculosis" not in treat.text.lower()
    assert "tuberculosis" in embed_text(treat.title, treat.section_path, treat.text).lower()


def test_citation_label_is_human_checkable(stg_doc: SourceDoc, stg_pdf: Path):
    chunks = chunk_document(parse_pdf(stg_pdf, "stg-tb", stg_doc.title), stg_doc)
    label = chunks[0].cite_label("C1")
    assert "stg-tb" in label and "p." in label and "§" in label


def test_runt_tail_merges_when_there_are_exactly_two_pieces(stg_doc):
    """Regression: the merge wrote to pieces[-2] after pop() had shortened the
    list, which is out of range at exactly two pieces."""
    from rag_project.ingest.chunk import chunk_section
    from rag_project.ingest.parse import Section

    long_sentence = "The guidelines describe the recommended approach in detail. " * 30
    section = Section(
        path="A > B", page_start=1, page_end=1, text=long_sentence + "Short tail."
    )
    chunks = chunk_section(section, stg_doc, "now", 0)
    assert chunks, "must not raise, and must produce chunks"


def test_later_chapter_does_not_nest_under_the_first(two_chapter_pdf: Path):
    """Regression: chapter titles are not sized consistently across a document.
    Sizing alone nested EMPYEMA THORACIS under Dengue Fever, so every empyema
    chunk cited 'Dengue Fever' as its section -- a wrong citation, which in a
    medical corpus is the failure this project exists to prevent."""
    parsed = parse_pdf(two_chapter_pdf, "paed", "Paediatrics")
    paths = [s.path for s in parsed.sections]
    empyema = [p for p in paths if "EMPYEMA" in p]
    assert empyema, paths
    assert not any("Dengue" in p for p in empyema), f"empyema nested under dengue: {empyema}"
    assert any(p.startswith("EMPYEMA THORACIS") for p in empyema), empyema


# --- plain-text ingestion (for sources whose PDF has no text layer) ---------

_DENGUE_TEXT = """\
NATIONAL GUIDELINES FOR CLINICAL MANAGEMENT OF DENGUE

1. Introduction

Dengue is a mosquito borne viral infection transmitted by Aedes aegypti.
The incidence has risen substantially over recent decades across India.

1.1 Case Classification

Dengue is classified as dengue without warning signs, dengue with warning
signs, and severe dengue according to the revised WHO classification.
\f2. Clinical Management

2.1 Fluid Therapy

Isotonic crystalloid solutions are recommended for initial fluid replacement.
The rate is adjusted according to the haematocrit and clinical response.

WARNING SIGNS

Abdominal pain, persistent vomiting, mucosal bleeding and lethargy are
recognised warning signs requiring closer observation.
"""


def test_text_numbered_headings_nest(tmp_path: Path):
    from rag_project.ingest.text import parse_text

    f = tmp_path / "dengue.txt"
    f.write_text(_DENGUE_TEXT)
    paths = [s.path for s in parse_text(f, "dengue").sections]
    assert "1. Introduction > 1.1 Case Classification" in paths
    assert "2. Clinical Management > 2.1 Fluid Therapy" in paths
    # A top-level numbered heading must not nest under the previous chapter.
    assert not any(p.startswith("1. Introduction > 2.") for p in paths)


def test_text_pages_come_from_form_feeds(tmp_path: Path):
    from rag_project.ingest.text import parse_text

    f = tmp_path / "dengue.txt"
    f.write_text(_DENGUE_TEXT)
    parsed = parse_text(f, "dengue")
    assert parsed.n_pages == 2
    by_path = {s.path: s for s in parsed.sections}
    assert by_path["1. Introduction"].page_start == 1
    assert by_path["2. Clinical Management > 2.1 Fluid Therapy"].page_start == 2


def test_text_without_page_breaks_refuses_to_invent_page_numbers(tmp_path: Path):
    """A citation reading 'p.1' for a 55-page document is confidently wrong,
    which is worse than admitting the page is unknown."""
    from rag_project.ingest.text import parse_text

    f = tmp_path / "dengue.txt"
    f.write_text(_DENGUE_TEXT.replace("\f", "\n"))
    parsed = parse_text(f, "dengue")
    assert all(s.page_start == 0 for s in parsed.sections)


def test_unknown_page_renders_honestly_in_a_citation(stg_doc: SourceDoc):
    from rag_project.models import Chunk

    c = Chunk(
        chunk_id="d::1", doc_id="dengue", title="Dengue Guidelines", source_url="",
        publisher="MOHFW", section_path="Management", page_start=0, page_end=0,
        text="...", n_tokens=5, index_version="v1", ingested_at="now",
    )
    assert "page n/a" in c.cite_label("C1")
    assert "p.1" not in c.cite_label("C1")
