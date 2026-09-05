"""PDF -> ordered sections with a heading path.

Chunk quality is mostly a function of section detection: a chunk that says
"Treatment > Adults > Intensive phase" retrieves far better than a naked
paragraph, and it lets the answer cite a location a human can actually check
in the source PDF.

Headings are found structurally (font size relative to body text, plus weight)
rather than by matching a fixed list of titles, so this survives documents whose
section names differ. The STG-specific regex is a supplement, not the basis.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Section names that recur across MOHFW Standard Treatment Guidelines. Used to
# rescue headings that are typeset at body size (which happens in these PDFs).
STG_SECTION_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s*[.)]?\s*)?("
    r"case\s+definition|introduction|incidence|epidemiology|aetiology|etiology"
    r"|risk\s+factors?|clinical\s+features?|signs?\s+and\s+symptoms?"
    r"|differential\s+diagnos[ei]s|diagnos(?:is|tic\s+criteria)"
    r"|investigations?|laboratory\s+\w+|management|treatment(?:\s+\w+)?"
    r"|pharmacological\s+\w+|non[- ]pharmacological\s+\w+"
    r"|prevention|prophylaxis|follow[- ]?up|referral\s+criteria"
    r"|complications?|prognosis|counsell?ing|references?"
    r")\s*:?\s*$",
    re.IGNORECASE,
)

_BOLD_FLAG = 1 << 4

#: Below this, a PDF is images rather than text and needs OCR.
MIN_CHARS_PER_PAGE = 100

#: A heading at least this much larger than body text, opening a page, starts a
#: new chapter rather than nesting under whatever came before.
CHAPTER_SIZE_RATIO = 1.15

#: Front matter. Indexed, a table of contents is actively harmful: it is dense
#: with section names and page numbers, so it matches many queries lexically
#: while containing no clinical content to answer them with.
FRONT_MATTER_RE = re.compile(
    r"^\s*(table\s+of\s+)?contents?\s*$|^\s*(acknowledge?ments?|foreword|preface"
    r"|abbreviations?|list\s+of\s+(tables?|figures?|contributors?|abbreviations?)"
    r"|contributors?|expert\s+group|copyright|disclaimer|message\s+from"
    r"|about\s+(this|the)\s+\w+|references?|further\s+reading"
    r"|bibliography|suggested\s+reading)\s*$",
    re.IGNORECASE,
)

#: Deepest heading path shown. These documents nest 6-8 levels once cover-page
#: lines are counted, and the shallow levels duplicate the document title we
#: already store separately, so the *most specific* levels are the ones kept.
MAX_PATH_DEPTH = 4

#: A section shorter than this is merged into its previous sibling rather than
#: indexed alone -- a 12-token fragment retrieves noise, not content.
RUNT_SECTION_CHARS = 200


@dataclass
class Line:
    text: str
    page: int
    size: float
    bold: bool
    #: Lines in this line's parent text block. A heading is typeset as its own
    #: paragraph, so block_lines == 1; bold *inside* running text shares a block
    #: with its neighbours. This is the signal that separates the two.
    block_lines: int = 1


@dataclass
class Section:
    path: str
    page_start: int
    page_end: int
    text: str = ""
    heading_level: int = 0
    #: Set from the raw heading at creation time. It cannot be recovered from
    #: `path` afterwards, because _render_path strips front-matter titles out.
    is_front_matter: bool = False


@dataclass
class ParsedDoc:
    doc_id: str
    n_pages: int
    sections: list[Section] = field(default_factory=list)
    n_chars: int = 0

    @property
    def chars_per_page(self) -> float:
        return self.n_chars / self.n_pages if self.n_pages else 0.0

    @property
    def has_text_layer(self) -> bool:
        """False for scanned-image PDFs, which extract to nothing.

        Worth an explicit flag rather than "produced no chunks": a scanned
        document silently contributing zero chunks looks identical, in every
        count that matters, to one that ingested fine.
        """
        return self.chars_per_page >= MIN_CHARS_PER_PAGE


def _extract_lines(path: Path) -> tuple[list[Line], int]:
    # Imported here, not at module scope. `load_chunks` lives in this
    # package and is on the *query* path, so a module-level import would
    # make every deployment that only answers questions carry a 64 MB PDF
    # library it never calls -- and fail at import if it were omitted.
    import pymupdf

    lines: list[Line] = []
    with pymupdf.open(path) as doc:
        n_pages = doc.page_count
        for pno, page in enumerate(doc, start=1):
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                if block.get("type") != 0:  # 0 = text
                    continue
                block_lines = len(block.get("lines", []))
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if not text:
                        continue
                    # Weight size by the longest span so a stray superscript
                    # doesn't misreport the line's size.
                    lead = max(spans, key=lambda s: len(s.get("text", "")))
                    lines.append(
                        Line(
                            text=text,
                            page=pno,
                            size=round(float(lead.get("size", 0.0)), 1),
                            bold=bool(int(lead.get("flags", 0)) & _BOLD_FLAG)
                            or "bold" in str(lead.get("font", "")).lower(),
                            block_lines=block_lines,
                        )
                    )
    return lines, n_pages


def _body_size(lines: list[Line]) -> float:
    """Most common font size weighted by character count = body text."""
    counter: Counter[float] = Counter()
    for ln in lines:
        counter[ln.size] += len(ln.text)
    return counter.most_common(1)[0][0] if counter else 10.0


def _is_heading(ln: Line, body: float) -> bool:
    if len(ln.text) > 120 or ln.text.endswith((".", ";", ",")):
        return False
    if ln.size >= body * 1.12:
        return True
    if STG_SECTION_RE.match(ln.text):
        return True
    # Bold alone is not enough: these documents bold drug names and list labels
    # mid-paragraph. Requiring the line to be its own block is what distinguishes
    # a heading from emphasis, and it is what stopped ~48% of chunks coming out
    # under 100 tokens.
    if ln.bold and ln.block_lines == 1 and len(ln.text) <= 80 and ln.size >= body * 0.98:
        return True
    return False


def _furniture_lines(lines: list[Line], n_pages: int) -> set[str]:
    """Find running headers/footers by repetition, not by matching the title.

    A running header is defined by appearing on many pages; the document's H1
    title appears once. Testing title-equality instead would delete the H1 and
    orphan every section beneath it from its parent path.
    """
    pages_by_text: dict[str, set[int]] = {}
    for ln in lines:
        pages_by_text.setdefault(ln.text.strip().lower(), set()).add(ln.page)
    if n_pages < 3:
        return set()
    threshold = max(3, int(n_pages * 0.4))
    return {t for t, pages in pages_by_text.items() if len(pages) >= threshold}


def _is_page_number(text: str) -> bool:
    return bool(
        re.fullmatch(r"(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?", text.strip(), re.IGNORECASE)
    )


def parse_pdf(path: Path, doc_id: str, doc_title: str = "") -> ParsedDoc:
    lines, n_pages = _extract_lines(path)
    n_chars = sum(len(ln.text) for ln in lines)
    if not lines:
        return ParsedDoc(doc_id=doc_id, n_pages=n_pages, sections=[], n_chars=0)

    body = _body_size(lines)
    furniture = _furniture_lines(lines, n_pages)
    sections: list[Section] = []
    # (size, title) stack -- a bigger font pops smaller ones off, which is how
    # the hierarchical path gets built.
    stack: list[tuple[float, str]] = []
    current: Section | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal current, buf
        if current is not None:
            current.text = "\n".join(buf).strip()
            if current.text:
                sections.append(current)
        buf = []

    # A cover page is title art, not structure. Its letter-spaced, overlapping
    # text ("Hy ypertension", "ELINES") parses into fragments that would
    # otherwise become permanent ancestors of every heading in the document and
    # show up in every citation it supports.
    cover_page = 1 if n_pages > 4 else 0
    #: Pages whose first content line we have already handled -- used to spot a
    #: heading that opens a page.
    page_opened: set[int] = set()

    for ln in lines:
        if _is_page_number(ln.text) or ln.text.strip().lower() in furniture:
            continue

        if ln.page <= cover_page:
            if current is None:
                current = Section(path="Front cover", page_start=ln.page, page_end=ln.page)
            current.page_end = ln.page
            buf.append(ln.text)
            continue

        page_leading = ln.page not in page_opened
        page_opened.add(ln.page)

        if _is_heading(ln, body):
            flush()
            if page_leading and ln.size >= body * CHAPTER_SIZE_RATIO:
                # A prominent heading opening a page is a new chapter. These
                # guidelines put one condition per chapter and start each on a
                # fresh page, but they do NOT size chapter titles consistently:
                # in paediatrics.pdf the first is 18pt and later ones 16.1pt, so
                # size alone nested "EMPYEMA THORACIS" *under* "Dengue Fever"
                # and every hernia chunk cited Dengue Fever as its section.
                stack.clear()
            while stack and stack[-1][0] <= ln.size:
                stack.pop()
            stack.append((ln.size, ln.text.rstrip(":").strip()))
            current = Section(
                path=_render_path(stack),
                page_start=ln.page,
                page_end=ln.page,
                heading_level=len(stack),
                is_front_matter=bool(FRONT_MATTER_RE.match(ln.text.rstrip(":").strip())),
            )
        else:
            if current is None:
                current = Section(path="Preamble", page_start=ln.page, page_end=ln.page)
            current.page_end = ln.page
            buf.append(ln.text)

    flush()
    return ParsedDoc(
        doc_id=doc_id, n_pages=n_pages,
        sections=_merge_runts(sections), n_chars=n_chars,
    )


def _render_path(stack: list[tuple[float, str]]) -> str:
    """Heading path for display and for the contextual embedding prefix.

    Front-matter titles are dropped from *ancestors*, not just leaves. A
    mis-detected "Contents" heading that never pops leaves real clinical content
    filed under "... > Contents > DIFFERENTIAL DIAGNOSIS"; discarding those
    sections would delete guidance, so the content is kept and the misleading
    ancestor is stripped from the path instead.
    """
    titles = [t for _, t in stack if not FRONT_MATTER_RE.match(t)]
    return " > ".join(titles[-MAX_PATH_DEPTH:]) or "Body"


def _parent(path: str) -> str:
    return path.rsplit(" > ", 1)[0] if " > " in path else path


def _merge_runts(sections: list[Section]) -> list[Section]:
    """Fold tiny sections into the previous sibling under a shared parent.

    Chunks still never straddle unrelated sections: a merge only happens between
    consecutive sections sharing a parent heading, and the merged result is
    attributed to that parent. So the citation stays true -- it just points one
    level up, which is where the content actually lives.
    """
    merged: list[Section] = []
    for section in sections:
        if section.is_front_matter:
            continue
        if (
            merged
            and len(section.text) < RUNT_SECTION_CHARS
            and _parent(section.path) == _parent(merged[-1].path)
            and " > " in section.path
        ):
            prev = merged[-1]
            leaf = section.path.rsplit(" > ", 1)[-1]
            prev.text = f"{prev.text}\n{leaf}: {section.text}".strip()
            prev.page_end = max(prev.page_end, section.page_end)
            prev.path = _parent(section.path)
            continue
        merged.append(section)
    return merged
