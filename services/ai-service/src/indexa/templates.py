"""Template-copy detection.

Entities often paste the national policy template into their own policy. A
pasted clause states the requirement; it is not evidence that the entity
implemented it. This module measures, deterministically, how much of a quoted
passage is lifted from known public templates, so a decision resting only on
copied text can be downgraded and the copied part shown to the reviewer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .text import normalize

SHINGLE = 5  # words per shingle; long enough to be specific, short enough for PDF noise
COPY_THRESHOLD = 0.7  # share of a quote's shingles found in a template

_WORD = re.compile(r"[ء-ي٠-٩]+|[a-z0-9]+")
_CONFIG = Path(__file__).resolve().parents[2] / "config" / "templates"
_CACHE = Path.home() / ".cache" / "indexa" / "templates"


def _words(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def _shingles(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


class TemplateIndex:
    def __init__(self, texts: list[str] | None = None):
        self.shingles: set[tuple[str, ...]] = set()
        for t in texts or []:
            self.add(t)

    def add(self, text: str) -> None:
        self.shingles |= _shingles(_words(text))

    def __bool__(self) -> bool:
        return bool(self.shingles)

    def copy_ratio(self, text: str) -> float:
        """Share of the passage's word 5-grams that appear in a template (0 for short text)."""
        sh = _shingles(_words(text))
        return sum(s in self.shingles for s in sh) / len(sh) if sh else 0.0

    def is_copied(self, text: str, threshold: float = COPY_THRESHOLD) -> bool:
        return self.copy_ratio(text) >= threshold


def load_template_index(fetch: bool = True) -> TemplateIndex:
    """Build the index from cached template texts, downloading missing ones when allowed."""
    idx = TemplateIndex()
    cfg = json.loads((_CONFIG / "sources.json").read_text(encoding="utf-8"))
    _CACHE.mkdir(parents=True, exist_ok=True)
    for src in cfg.get("sources", []):
        cached = _CACHE / f"{src['id']}.txt"
        if not cached.exists() and fetch:
            try:
                cached.write_text(_download_pdf_text(src["url"]), encoding="utf-8")
            except Exception:  # a missing template only weakens detection
                continue
        if cached.exists():
            idx.add(cached.read_text(encoding="utf-8"))
    return idx


def _download_pdf_text(url: str) -> str:
    import io

    import httpx
    import pdfplumber

    data = httpx.get(url, timeout=120, follow_redirects=True).content
    from .documents import repair_text

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "\n".join(repair_text(p.extract_text() or "") for p in pdf.pages)
