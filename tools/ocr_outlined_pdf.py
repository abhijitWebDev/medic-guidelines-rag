"""OCR a PDF whose text was flattened to vector outlines -> plain text.

    python tools/ocr_outlined_pdf.py <in.pdf> <out.txt>

Needs `rapidocr-onnxruntime`, `opencv-python` and `pymupdf`, none of which the
project depends on at runtime -- install them into a throwaway venv. This is
run by hand when a source document is added, not as part of `rag ingest`.

Written for dengue.pdf, the one document in this corpus that needs it.

That document was exported by PrimoPDF with every glyph flattened to vector
outlines: 55 pages, zero fonts, ~12k bezier paths per page, and nothing at all
for a text extractor to read.  It is not a scan, though -- the render is exact,
so lines and word gaps can be found by pixel projection and only the *recogniser*
half of the OCR stack is needed.  That matters: the stock text detector, tuned
for photographed pages, silently dropped ~20% of the lines on every page.

Output is one \f-separated block per PDF page, which is the page evidence
ingest/text.py needs to cite "p.N" against the source PDF.
"""
from __future__ import annotations

import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from rapidocr_onnxruntime import RapidOCR

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])

INK = 190
DPI = 300
PAD = 6

#: A bullet glyph comes back from the recogniser as a stray dot or nothing.
_BULLET = re.compile(r"^[.·•∙・,]{1,2}$")
#: Folios come back with the digits spaced apart ("1 8"), so allow the gaps.
_PAGE_NUM = re.compile(r"^(page\s*)?[ivxlcdm\d]( ?[ivxlcdm\d]){0,5}$", re.IGNORECASE)

#: The recogniser is a Chinese-English model, so on English-only pages it
#: reaches for CJK forms of shapes it half-recognises: Roman numerals become
#: their ideographs and ASCII punctuation becomes fullwidth. Left alone,
#: "DHF grade III" indexes as "DHF grade 三" and never matches a query.
_CONFUSIONS = {
    "\u4e00": "-", "\u4e8c": "II", "\u4e09": "III", "\u56db": "IV",
    "\uff0c": ",", "\uff1a": ":", "\uff1b": ";", "\uff0e": ".",
    "\uff08": "(", "\uff09": ")", "\uff1c": "<", "\uff1e": ">",
    "\uff05": "%", "\uff0f": "/", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u3000": " ",
}
_CJK = re.compile(r"[\u2e80-\u9fff\uac00-\ud7ff\uff00-\uffef]")
_SPACE_BEFORE = re.compile(r"\s+([,;:)%])")
_SPACE_AFTER = re.compile(r"([(])\s+")

#: The cover is title art -- emblems, seals and a letter-spaced masthead. The
#: recogniser returns a line of debris per logo, and unlike a table cell ("1.5",
#: "20 mmHg") none of it is content, so on this page alone a line has to carry a
#: real word to be kept. parse.py draws the same boundary for the PDF corpus.
COVER_PAGES = 1
_HAS_WORD = re.compile(r"[A-Za-z]{3}")


def _normalise(text: str) -> str:
    for bad, good in _CONFUSIONS.items():
        text = text.replace(bad, good)
    # Whatever CJK survives is a glyph the model could not place at all; this
    # document has no CJK content, so it is noise rather than lost text.
    text = _CJK.sub("", text)
    text = _SPACE_BEFORE.sub(r"\1", text)
    text = _SPACE_AFTER.sub(r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _runs(mask, min_len=1):
    d = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return [(int(a), int(b)) for a, b in zip(d[::2], d[1::2]) if b - a >= min_len]


def _strip_rules(ink):
    """Erase table and box borders.

    A ruled box joins every row it encloses into one unbroken band of ink, so a
    horizontal projection sees a 600px "line" and the height filter throws the
    whole table away.  Removing the rules first is what lets the rows inside a
    box separate normally -- and the boxes here are the fluid-regimen and WHO
    grading tables, the highest-value content in the document.
    """
    u8 = ink.astype(np.uint8)
    h, w = ink.shape
    horiz = cv2.morphologyEx(
        u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 12, 1))
    )
    vert = cv2.morphologyEx(
        u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 20))
    )
    rules = cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8))
    return ink & ~rules.astype(bool)


def _rows_of(ink, y0, y1, x0, x1):
    """Text lines inside one region, split on the row-ink profile.

    Splitting on a low threshold rather than on fully blank rows matters: the
    leading here is tight enough that descenders reach the next line's
    ascenders, so on many pages no row between two lines is truly empty and a
    blank-row projection merges four lines into one band. Overlap contributes a
    few pixels to a row where a real line contributes hundreds.
    """
    out = []
    block = ink[y0:y1, x0:x1]
    for a0, b0 in _runs(block.any(axis=1)):
        profile = block[a0:b0].sum(axis=1)
        cut = max(1, int(profile.max() * 0.06))
        for a, b in _runs(profile > cut, min_len=8):
            a, b = max(0, a0 + a - 2), min(block.shape[0], a0 + b + 2)
            cols = np.zeros(ink.shape[1], dtype=bool)
            cols[x0:x1] = block[a:b].any(axis=0)
            xs = np.flatnonzero(cols)
            if xs.size:
                out.append((y0 + a, y0 + b, int(xs[0]), int(xs[-1]) + 1, cols))
    return out


def _regions(ink, y0, y1, x0, x1, line_h, depth=0):
    """Recursive XY-cut, so side-by-side boxes are read column by column.

    The flowcharts and the grading tables put two boxes at the same height. A
    plain row projection reads straight across them and interleaves the two --
    "Pulse / Tourniquet test" -- which for a dosing or grading table would pair
    a value with the wrong label. Cutting on vertical gutters first keeps each
    column intact.

    The height guard is what makes this safe on ordinary prose: in a region only
    one line tall every word gap is a full-height "gutter", so without it every
    word would become its own column.
    """
    block = ink[y0:y1, x0:x1]
    if not block.any():
        return []

    if depth < 4 and (y1 - y0) >= line_h * 2.5:
        cols = block.any(axis=0)
        gutter = max(int(ink.shape[1] * 0.030), 12)
        inner = [(a, b) for a, b in _runs(~cols, min_len=gutter) if a > 0 and b < len(cols)]
        if inner:
            parts, prev = [], 0
            for a, b in inner:
                if a > prev:
                    parts.append((prev, a))
                prev = b
            if prev < len(cols):
                parts.append((prev, len(cols)))
            # A column has to be wide enough to be a column. Without this, the
            # gap between a bullet glyph and its text is a full-height gutter,
            # and every bulleted list gets filed as a column of naked bullets
            # followed by a column of orphaned text.
            inked = [(a, b) for a, b in parts if block[:, a:b].any()]
            narrow = any(b - a < ink.shape[1] * 0.12 for a, b in inked)
            if len(inked) >= 2 and not narrow:
                out = []
                for a, b in inked:
                    out += _regions(ink, y0, y1, x0 + a, x0 + b, line_h, depth + 1)
                return out

    # No usable gutter: cut into horizontal strips and, if a strip is still tall
    # enough to hold stacked lines, try for gutters inside it.
    rows = block.any(axis=1)
    strips = _runs(rows)
    if depth < 4 and len(strips) > 1:
        out = []
        for a, b in strips:
            out += _regions(ink, y0 + a, y0 + b, x0, x1, line_h, depth + 1)
        return out
    return _rows_of(ink, y0, y1, x0, x1)


def _lines(ink):
    ink = _strip_rules(ink)
    rough = [b[1] - b[0] for b in _rows_of(ink, 0, ink.shape[0], 0, ink.shape[1])]
    # Measured, not assumed to be body copy: the title page is three lines of
    # 48pt display type, and pinning line_h to a body-text height there makes
    # every one of those lines "too tall to be text" and drops the document's
    # own title.
    line_h = float(np.clip(np.median(rough or [42.0]), 20.0, 150.0))

    bands = _regions(ink, 0, ink.shape[0], 0, ink.shape[1], line_h)
    tall = max(line_h * 2.4, 60)
    return [
        b for b in bands
        # Anything still tall after the rules came out is a photograph or a
        # logo. Ink density tells it from prose: text is mostly white space.
        if not (b[1] - b[0] > tall and ink[b[0]:b[1], b[2]:b[3]].mean() > 0.22)
    ]


def _word_gap(blanks, height):
    """Width separating an intra-word blank from a space, for one line.

    These pages mix justified lines (wide spaces) with tightly kerned ones, so
    any fixed fraction of the line height mis-splits one or the other, and a
    plain "largest jump" rule latches onto noise down among the 2-3px letter
    gaps and shatters words into letters. The widths of one line are bimodal --
    letter gaps against spaces -- which is exactly what Otsu's threshold is for.
    """
    widths = np.array([b - a for a, b in blanks if b - a >= 2])
    fallback = max(4, int(height * 0.20))
    if widths.size < 2 or widths.max() == widths.min():
        return fallback

    counts = np.bincount(widths)
    values = np.arange(counts.size)
    total = counts.sum()
    w0 = np.cumsum(counts)[:-1]
    w1 = total - w0
    valid = (w0 > 0) & (w1 > 0)
    if not valid.any():
        return fallback
    csum = np.cumsum(counts * values)[:-1]
    mu0 = np.divide(csum, w0, out=np.zeros_like(csum, dtype=float), where=w0 > 0)
    mu1 = np.divide(
        csum[-1] + counts[-1] * values[-1] - csum, w1,
        out=np.zeros_like(csum, dtype=float), where=w1 > 0,
    )
    between = w0 * w1 * (mu0 - mu1) ** 2
    between[~valid] = -1.0
    cut = int(np.argmax(between)) + 1

    # A space is never narrower than a letter gap nor wider than the line is
    # tall; Otsu on a line whose gaps are all one kind lands outside that.
    return int(min(max(cut, max(4, height * 0.11)), max(8, height * 0.75)))


def _words(cols, x0, x1, height):
    seg = cols[x0:x1]
    blanks = _runs(~seg, min_len=2)
    gap = _word_gap(blanks, height)
    spans, prev = [], 0
    for a, b in blanks:
        if b - a < gap:
            continue
        if a > prev:
            spans.append((prev, a))
        prev = b
    if prev < len(seg):
        spans.append((prev, len(seg)))
    return [(x0 + a, x0 + b) for a, b in spans if b - a >= 2]


def _letter_spaced(spans, height):
    """True when a line is one word set with wide letter spacing.

    The chapter titles are tracked out ("A C K N O W L E D G M E N T S"), so the
    gap histogram has no word-gap mode and every glyph splits off as its own
    word: "ACKNOWLEDGMENTS" indexes as "AC KN OWL E DG M E N TS" and becomes the
    root of the path for every section under that chapter. Recognising the line
    whole does not help -- the model sees the same wide gaps and inserts the
    same spaces -- so the gaps have to be closed in the image first.
    """
    if len(spans) < 4:
        return False
    widths = [b - a for a, b in spans]
    # A capital glyph is around 0.6-0.8 as wide as the line is tall, not half.
    singles = sum(1 for w in widths if w < height * 0.90)
    return singles >= 2 and float(np.median(widths)) < height * 1.10


def _squeeze(rgb, spans, y0, y1, height):
    """Re-paste a line's glyphs at normal spacing on a white canvas."""
    gap = max(3, height // 12)
    parts = [rgb[y0:y1, a:b] for a, b in spans]
    canvas_w = sum(p.shape[1] for p in parts) + gap * (len(parts) - 1)
    canvas = np.full((y1 - y0, canvas_w, 3), 255, dtype=rgb.dtype)
    x = 0
    for part in parts:
        canvas[:, x:x + part.shape[1]] = part
        x += part.shape[1] + gap
    return canvas


def read_page(engine, page):
    """One page -> [(y0, y1, text)] in reading order, plus its confidences."""
    pix = page.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    ink = gray < INK
    rgb = np.repeat(gray[:, :, None], 3, axis=2)

    crops, owner = [], []
    bands = _lines(ink)
    for i, (y0, y1, x0, x1, cols) in enumerate(bands):
        height = y1 - y0
        spans = _words(cols, x0, x1, height)
        top, bot = max(0, y0 - PAD), y1 + PAD
        for a, b in spans:
            crops.append(rgb[top:bot, max(0, a - PAD):b + PAD])
            owner.append(i)
        if _letter_spaced(spans, height):
            # Read it both ways. Geometry alone cannot tell tracked-out letters
            # from a table row of single digits, but the two read back
            # differently: only the first yields stray single letters.
            crops.append(_squeeze(rgb, spans, top, bot, height))
            owner.append(~i)

    texts = [[] for _ in bands]
    squeezed: dict[int, str] = {}
    confs: list[float] = []
    if crops:
        res, _ = engine.text_rec(crops)
        for (txt, conf), i in zip(res, owner):
            txt = txt.strip()
            if not txt:
                continue
            confs.append(float(conf))
            if i < 0:
                squeezed[~i] = txt
            else:
                texts[i].append("-" if _BULLET.fullmatch(txt) else txt)

    lines = []
    for i, ((y0, y1, *_), toks) in enumerate(zip(bands, texts)):
        if not toks:
            continue
        loose = sum(1 for t in toks if len(t) == 1 and t.isalpha())
        lines.append((y0, y1, squeezed[i] if (loose >= 2 and i in squeezed) else " ".join(toks)))
    return lines, confs, len(crops)


def main() -> int:
    engine = RapidOCR()
    doc = pymupdf.open(SRC)
    pages: list[list[tuple[int, int, str]]] = []
    confs: list[float] = []
    t_start = time.time()

    for pno, page in enumerate(doc, start=1):
        page_lines, page_confs, n_crops = read_page(engine, page)
        confs.extend(page_confs)
        pages.append(page_lines)
        print(
            f"p{pno:02d}/{doc.page_count}  lines={len(page_lines):3d} "
            f"words={n_crops:4d}  {time.time() - t_start:6.0f}s",
            file=sys.stderr, flush=True,
        )

    doc.close()

    # Running headers/footers repeat across pages; a real heading does not.
    seen: dict[str, set[int]] = {}
    for pno, lines in enumerate(pages, start=1):
        for _, _, t in lines:
            seen.setdefault(t.strip().lower(), set()).add(pno)
    furniture = {t for t, ps in seen.items() if len(ps) >= max(3, len(pages) * 0.4)}

    blocks = []
    for pno, lines in enumerate(pages, start=1):
        pitch = np.median([b - a for a, b, _ in lines]) if lines else 0
        kept, prev_end = [], None
        for y0, y1, text in lines:
            flat = _normalise(text)
            if not flat or flat.lower() in furniture or _PAGE_NUM.fullmatch(flat):
                continue
            if pno <= COVER_PAGES and not _HAS_WORD.search(flat):
                continue
            # Paragraph breaks are the evidence text.py uses to spot a heading
            # standing alone, so blank lines have to survive the transcription.
            if prev_end is not None and y0 - prev_end > pitch * 1.15:
                kept.append("")
            kept.append(flat)
            prev_end = y1
        blocks.append("\n".join(kept))

    OUT.write_text("\f".join(blocks), encoding="utf-8")
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    low = sum(1 for c in confs if c < 0.80)
    print(
        f"\nwrote {OUT}  pages={len(blocks)}  words={len(confs)}  "
        f"mean_conf={mean_conf:.3f}  below_0.80={low} ({low / max(1, len(confs)):.1%})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
