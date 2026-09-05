"""Corpus manifest: the allow-list of documents this assistant may ingest.

A PDF sitting in data/raw/ is not part of the corpus. A PDF *described in the
manifest, with a matching hash* is. That distinction is what lets the project
honestly claim a curated rather than scraped knowledge base -- and it means
swapping in a random PDF fails loudly instead of silently widening scope.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import get_settings
from ..models import SourceDoc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return re.sub(r"-{2,}", "-", s)


#: Metadata titles that are authoring-tool noise rather than a real title.
_JUNK_TITLE = re.compile(
    r"^(microsoft word|untitled|document\d*|book\d*|\d+)|\.(docx?|cdr|indd|pdf)$",
    re.IGNORECASE,
)
#: Cover-page lines that are navigation furniture, not the title.
_NOT_A_TITLE = re.compile(
    r"^(page\s*(no\.?|\d+)|contents?|index|topics?|s\.?l?\.?\s*no\.?"
    r"|table of contents|chapter\s*\d*|\d+)\.?$",
    re.IGNORECASE,
)


def _cover_title(doc, page_no: int) -> str:
    """Join the largest-font lines on a page, in reading order.

    Cover titles are typeset as several big lines ("STANDARD" / "TREATMENT" /
    "GUIDELINES FOR" / "ORTHOPAEDICS"), so they must be joined in document
    order -- sorting by font size alone scrambles them.
    """
    lines: list[tuple[float, str]] = []
    for block in doc[page_no].get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(sp.get("text", "") for sp in spans).strip()
            if text:
                lines.append((round(max(sp.get("size", 0) for sp in spans), 1), text))
    if not lines:
        return ""

    biggest = max(size for size, _ in lines)
    parts = [t for size, t in lines if size >= biggest * 0.95 and not _NOT_A_TITLE.match(t)]
    title = re.sub(r"\s+", " ", " ".join(parts)).strip()

    words = [w for w in title.split() if any(c.isalpha() for c in w)]
    if not (8 <= len(title) <= 130 and len(words) >= 2):
        return ""
    return title.title() if title.isupper() else title


def _text_title(path: Path) -> str:
    """First non-blank, heading-shaped line of a text file, else the filename."""
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
            t = line.strip().lstrip("#").strip()
            if 8 <= len(t) <= 130 and len([w for w in t.split() if w.isalpha()]) >= 2:
                return t.title() if t.isupper() else t
    except OSError:
        pass
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


def _pdf_title(path: Path) -> str:
    """Human title: the document's own cover page, then metadata, then filename.

    Metadata is checked *after* the cover page because these PDFs were exported
    from Word and carry titles like "Microsoft Word - 822.docx", which would
    otherwise appear in every citation this document supports.
    """
    try:
        import pymupdf

        with pymupdf.open(path) as doc:
            for page_no in range(min(3, doc.page_count)):
                title = _cover_title(doc, page_no)
                if title:
                    return title

            meta = ((doc.metadata or {}).get("title") or "").strip()
            if len(meta) > 3 and not _JUNK_TITLE.search(meta):
                return meta
    except Exception:
        pass
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


@dataclass
class ManifestDiff:
    listed_ok: list[str]
    missing: list[str]      # in manifest, not on disk
    unlisted: list[str]     # on disk, not in manifest -> refused
    changed: list[str]      # hash mismatch -> refused

    @property
    def clean(self) -> bool:
        return not (self.missing or self.unlisted or self.changed)


def scan_raw() -> list[SourceDoc]:
    """Generate manifest entries from whatever PDFs are in data/raw/.

    Metadata that cannot be inferred (url, specialty, version) is left blank
    on purpose -- a human fills it in. Blank provenance is visible; guessed
    provenance is not.
    """
    settings = get_settings()
    docs: list[SourceDoc] = []
    sources = sorted(
        [*settings.raw_dir.glob("*.pdf"), *settings.raw_dir.glob("*.txt")],
        key=lambda p: p.name,
    )
    for pdf in sources:
        docs.append(
            SourceDoc(
                doc_id=slugify(pdf.stem)[:64],
                title=_pdf_title(pdf) if pdf.suffix == ".pdf" else _text_title(pdf),
                filename=pdf.name,
                url="",
                specialty=None,
                version=None,
                published=None,
                sha256=sha256_file(pdf),
            )
        )
    return docs


def write_manifest(docs: list[SourceDoc], path: Path | None = None) -> Path:
    settings = get_settings()
    path = path or settings.manifest_path
    payload = {
        "source": "MOHFW Standard Treatment Guidelines",
        "source_page": (
            "https://clinicalestablishments.mohfw.gov.in/en/"
            "standard-treatment-guidelines"
        ),
        "documents": [d.model_dump() for d in docs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return path


def load_manifest(path: Path | None = None) -> list[SourceDoc]:
    settings = get_settings()
    path = path or settings.manifest_path
    if not path.exists():
        raise FileNotFoundError(
            f"No corpus manifest at {path}. Put the MOHFW PDFs in "
            f"{settings.raw_dir} and run `uv run rag-corpus scan`."
        )
    data = yaml.safe_load(path.read_text()) or {}
    return [SourceDoc(**d) for d in data.get("documents", [])]


def verify() -> ManifestDiff:
    """Compare manifest against data/raw/. Ingestion runs only if clean."""
    settings = get_settings()
    docs = load_manifest()
    by_name = {d.filename: d for d in docs}
    on_disk = {
        p.name: p
        for p in (*settings.raw_dir.glob("*.pdf"), *settings.raw_dir.glob("*.txt"))
    }

    listed_ok, missing, changed = [], [], []
    for name, doc in by_name.items():
        path = on_disk.get(name)
        if path is None:
            missing.append(name)
        elif sha256_file(path) != doc.sha256:
            changed.append(name)
        else:
            listed_ok.append(name)

    unlisted = sorted(set(on_disk) - set(by_name))
    return ManifestDiff(listed_ok, sorted(missing), unlisted, sorted(changed))
