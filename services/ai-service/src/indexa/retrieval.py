"""Lexical evidence retrieval (BM25) over an entity's chunks.

Why lexical and not only vectors: NDMO specifications are English while most
entity evidence is Arabic, and compliance language is term-heavy ("72 ساعة",
"مسؤول حماية البيانات", "سجل"). Each control carries a bilingual keyword set
generated once at framework setup and reviewed by a specialist; BM25 over those
keywords is exact, explainable, runs on a CPU with no model download, and is
the baseline any embedding retriever must beat to be worth its RAM.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from .documents import Chunk
from .text import tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.72):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query_tokens: Iterable[str]) -> list[float]:
        q = Counter(query_tokens)
        scores = []
        for d, length in zip(self.docs, self.lens, strict=True):
            s = 0.0
            for t, qf in q.items():
                tf = d.get(t)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avg or 1))
                s += self.idf.get(t, 0.0) * tf * (self.k1 + 1) / denom * (1 + 0.2 * (qf - 1))
            scores.append(s)
        return scores

    def search(self, query: str, k: int = 6, per_doc: int = 4) -> list[tuple[Chunk, float]]:
        """Top-k chunks, at most `per_doc` from one document, merged when adjacent duplicates."""
        scores = self.score(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        picked: list[tuple[Chunk, float]] = []
        per: Counter = Counter()
        seen_text: set[str] = set()
        for i in ranked:
            if scores[i] <= 0 or len(picked) >= k:
                break
            c = self.chunks[i]
            key = c.text[:120]
            if per[c.doc_id] >= per_doc or key in seen_text:
                continue
            picked.append((c, scores[i]))
            per[c.doc_id] += 1
            seen_text.add(key)
        return picked
