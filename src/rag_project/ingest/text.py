"""Plain-text ingestion, for sources whose PDF has no text layer.

The PDF parser infers structure from typography. Text files have none, so
headings are recognised from the conventions that survive extraction: markdown
hashes, decimal numbering, ALL CAPS lines, and short title-case lines standing
alone between blank lines.

Page numbers matter here more than they look. A citation reading "p.14" is
checkable against the source PDF; one reading "p.1" for a 55-page document is
worse than useless because it is confidently wrong. So pages come from real
evidence -- form feeds (what `pdftotext` emits at every page break) or explicit
markers -- and when there is no such evidence the document is recorded as a
single unpaginated unit rather than having page numbers invented for it.
"""

from __future__ import annotations

import re
from pathlib import Path

from .parse import (
    FRONT_MATTER_RE,
    MIN_CHARS_PER_PAGE,
    ParsedDoc,
    Section,
    _render_path,
)

#: pdftotext writes \f at every page break; some exporters use explicit markers.
_FORM_FEED = "\f"
_PAGE_MARKER = re.compile(r"^\s*(?:\[\[|<)?\s*page\s+(\d+)\s*(?:\]\]|>)?\s*$", re.IGNORECASE)

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_NUMBERED = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
_ALL_CAPS = re.compile(r"^[^a-z]*[A-Z][^a-z]{2,}$")

#: A heading names a thing; a sentence makes a statement about one. Prose that
#: happens to open with a figure ("11 countries of SEAR are endemic...", "1991
#: are shown in Figure 1.") satisfies the numbering pattern otherwise, and once
#: promoted it becomes the parent of every real section beneath it.
_MAX_TITLE_WORDS = 9
_SENTENCE = re.compile(r"[.!?]\s+\S")

#: Enough letters to be a word. Text recovered by OCR carries figure furniture
#: -- arrowheads, axis ticks, "V +" -- that is punctuation and single capitals,
#: and every one of those otherwise reads as an ALL-CAPS chapter title.
_MIN_TITLE_LETTERS = 4


def _letters(text: str) -> int:
    return sum(c.isalpha() for c in text)


def _looks_like_heading(line: str, prev_blank: bool, next_blank: bool) -> int:
    """Return a heading level, or 0 for body text."""
    text = line.strip()
    if not text or len(text) > 110:
        return 0

    md = _MD_HEADING.match(text)
    if md:
        return len(md.group(1))

    # Scattered single letters are chart furniture -- an axis label, an arrow
    # beside a value ("12 A n 18 80Ng t") -- not a title. Explicit markdown
    # above is exempt; everything below here is inference from shape.
    if sum(1 for t in text.split() if len(t) == 1 and t.isalpha()) >= 2:
        return 0

    num = _NUMBERED.match(text)
    if (
        num
        and len(text) <= 90
        and not text.rstrip().endswith((".", ";", ","))
        and len(num.group(2).split()) <= _MAX_TITLE_WORDS
        and _letters(num.group(2)) >= 3
        and not _SENTENCE.search(num.group(2))
    ):
        return min(6, num.group(1).count(".") + 1)

    if _ALL_CAPS.match(text) and 3 <= len(text) <= 80 and _letters(text) >= _MIN_TITLE_LETTERS:
        return 1

    # A short line alone between blank lines, not ending like a sentence.
    if (
        prev_blank
        and next_blank
        and len(text) <= 70
        and not text.endswith((".", ";", ",", ":"))
        and text[:1].isupper()
        and _letters(text) >= _MIN_TITLE_LETTERS
    ):
        return 2

    return 0


def _clean(text: str) -> str:
    md = _MD_HEADING.match(text.strip())
    if md:
        return md.group(2).strip()
    return text.strip().rstrip(":").strip()


def parse_text(path: Path, doc_id: str, doc_title: str = "") -> ParsedDoc:
    raw = path.read_text(encoding="utf-8", errors="replace")

    paginated = _FORM_FEED in raw
    pages = raw.split(_FORM_FEED) if paginated else [raw]

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    current: Section | None = None
    buf: list[str] = []
    n_chars = 0

    def flush() -> None:
        nonlocal current, buf
        if current is not None:
            current.text = "\n".join(buf).strip()
            if current.text:
                sections.append(current)
        buf = []

    page_no = 0
    for page_index, page_text in enumerate(pages, start=1):
        page_no = page_index
        lines = page_text.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()

            marker = _PAGE_MARKER.match(stripped)
            if marker:
                page_no = int(marker.group(1))
                continue
            if not stripped:
                continue

            n_chars += len(stripped)
            prev_blank = i == 0 or not lines[i - 1].strip()
            next_blank = i + 1 >= len(lines) or not lines[i + 1].strip()
            level = _looks_like_heading(line, prev_blank, next_blank)

            if level:
                flush()
                title = _clean(line)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                current = Section(
                    path=_render_path(stack),
                    page_start=page_no,
                    page_end=page_no,
                    heading_level=len(stack),
                    is_front_matter=bool(FRONT_MATTER_RE.match(title)),
                )
            else:
                if current is None:
                    current = Section(path="Body", page_start=page_no, page_end=page_no)
                current.page_end = page_no
                buf.append(stripped)

    flush()

    # Preface, contributor lists and abbreviation tables, dropped for the same
    # reason parse_pdf drops them: they are dense with section names and proper
    # nouns, so they match many queries lexically while holding no clinical
    # content to answer them with. A text source has no typography to fall back
    # on, so the heading is the only signal -- but it is the same signal.
    sections = [s for s in sections if not s.is_front_matter]

    # Without page evidence, report one page so `has_text_layer` still works and
    # citations do not claim a page number the source cannot support.
    n_pages = len(pages) if paginated else 1
    if not paginated:
        for section in sections:
            section.page_start = section.page_end = 0  # 0 == "page unknown"

    return ParsedDoc(
        doc_id=doc_id, n_pages=n_pages, sections=sections,
        n_chars=max(n_chars, MIN_CHARS_PER_PAGE * n_pages if sections else 0),
    )
