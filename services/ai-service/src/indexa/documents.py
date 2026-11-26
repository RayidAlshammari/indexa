"""Evidence documents split into page-anchored chunks.

Every chunk keeps the document and page it came from, so a decision can always
point a reviewer to the exact page that supports it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_BROKEN_LAM_HAMZA = re.compile(r"ا([إأآ])ل")


_ARABIC_WORD = re.compile(r"[\u0600-\u06ff]+")
_LTR_RUN = re.compile(
    r"[0-9A-Za-z\u0660-\u0669](?:[0-9A-Za-z\u0660-\u0669.,:/%_@-]*[0-9A-Za-z\u0660-\u0669])?"
)
_MIRROR = str.maketrans("()[]{}<>«»", ")(][}{><»«")


def _article_balance(text: str) -> int:
    """Words that start with the article "ال" minus words that end with it reversed ("لا")."""
    words = _ARABIC_WORD.findall(text)
    return sum(w.startswith("ال") for w in words) - sum(
        w.endswith("لا") and len(w) > 3 for w in words
    )


def _logical_line(line: str) -> str:
    """Turn one visually ordered line into reading order; digits and Latin keep their direction."""
    rev = line[::-1].translate(_MIRROR)
    return _LTR_RUN.sub(lambda m: m.group(0)[::-1], rev)


def repair_text(text: str) -> str:
    """Normalise PDF text to logical-order Arabic.

    Some producers store Arabic in visual order with presentation-form glyphs, so a
    line reads backwards ("تﺎﻧﺎﻴﺒﻟا" for "البيانات"). Reversing each line before NFKC keeps
    ligatures such as lam-alef in the right order. The page is flipped only when that
    makes far more words start with the article "ال" than end with its mirror.
    Also undoes the lam-alef reversal some producers emit ("اإل" -> "الإ").
    """
    forward = unicodedata.normalize("NFKC", text)
    flipped = unicodedata.normalize("NFKC", "\n".join(_logical_line(ln) for ln in text.split("\n")))
    if _article_balance(flipped) > _article_balance(forward) + 2:
        forward = flipped
    return _BROKEN_LAM_HAMZA.sub(r"ال\1", forward)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    pages: list[str]
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return sum(len(p) for p in self.pages)


def chunk_document(doc: Document, size: int = 650, overlap: int = 120) -> list[Chunk]:
    """Split each page into overlapping windows that break on line boundaries.

    Chunks never cross a page, which keeps page citations exact.
    """
    chunks: list[Chunk] = []
    for page_no, page in enumerate(doc.pages, start=1):
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        if not lines:
            continue
        buf: list[str] = []
        length = 0
        idx = 0
        for line in lines:
            if buf and length + len(line) > size:
                chunks.append(_make(doc, page_no, idx, buf))
                idx += 1
                tail: list[str] = []
                tail_len = 0
                for prev in reversed(buf):
                    if tail_len + len(prev) > overlap:
                        break
                    tail.insert(0, prev)
                    tail_len += len(prev)
                buf, length = tail, tail_len
            buf.append(line)
            length += len(line) + 1
        if buf and any(len(x) > 3 for x in buf):
            chunks.append(_make(doc, page_no, idx, buf))
    return chunks


def _make(doc: Document, page_no: int, idx: int, lines: list[str]) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}:p{page_no}:{idx}",
        doc_id=doc.doc_id,
        doc_title=doc.title,
        page=page_no,
        text="\n".join(lines),
    )
