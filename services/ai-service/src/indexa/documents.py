"""Evidence documents split into page-anchored chunks.

Every chunk keeps the document and page it came from, so a decision can always
point a reviewer to the exact page that supports it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_BROKEN_LAM_HAMZA = re.compile(r"ا([إأآ])ل")


def repair_text(text: str) -> str:
    """Undo the unambiguous lam-alef reversal some PDF producers emit ("اإل" -> "الإ")."""
    return _BROKEN_LAM_HAMZA.sub(r"ال\1", text)


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
