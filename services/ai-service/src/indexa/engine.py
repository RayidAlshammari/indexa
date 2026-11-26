"""Entity assessment: every specification judged against ALL of an entity's evidence.

Replaces the per-file evaluation loop. NDI assesses the entity per
specification; evidence for one specification is often spread over several
documents, and a document should never be marked non-compliant against
specifications it was never meant to cover.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .controls import Control
from .documents import Chunk, Document, chunk_document
from .judge import Judgement, apply_template_rule, build_messages, parse_judgement
from .llm import RequestTooLarge
from .retrieval import BM25Index
from .scoring import domain_scores, prioritised_gaps

LLMCall = Callable[[list[dict]], Awaitable[str]]


@dataclass
class Assessment:
    entity: str
    judgements: dict[str, Judgement]
    excerpts: dict[str, list[Chunk]]
    scores: list = field(default_factory=list)
    gaps: list = field(default_factory=list)


def build_index(documents: list[Document]) -> BM25Index:
    chunks: list[Chunk] = []
    for d in documents:
        chunks.extend(chunk_document(d))
    return BM25Index(chunks)


async def assess(
    entity: str,
    controls: list[Control],
    documents: list[Document],
    llm: LLMCall,
    *,
    top_k: int = 6,
    concurrency: int = 4,
    templates=None,
) -> Assessment:
    index = build_index(documents)
    sem = asyncio.Semaphore(concurrency)
    excerpts = {c.control_id: [ch for ch, _ in index.search(c.query, k=top_k)] for c in controls}

    async def one(c: Control) -> Judgement:
        async with sem:
            ex = excerpts[c.control_id]
            # Shrink the evidence when a provider caps request size, best excerpts first.
            for keep, chars in ((len(ex), 650), (4, 550), (3, 450)):
                shown = ex[:keep]
                try:
                    raw = await llm(build_messages(c, shown, max_chars=chars))
                    break
                except RequestTooLarge as e:
                    err: Exception = e
                    continue
                except Exception as e:  # provider error: never silently "not met"
                    err = e
                    shown = None
                    break
            else:
                shown = None
            if shown is None:
                j = Judgement(c.control_id, "not_met", 0.0, "", needs_review=True)
                j.review_reasons = [f"model call failed: {err}"]
                return j
            excerpts[c.control_id] = shown
            return apply_template_rule(parse_judgement(c, raw, shown), templates)

    results = await asyncio.gather(*(one(c) for c in controls))
    judgements = {j.control_id: j for j in results}
    return Assessment(
        entity=entity,
        judgements=judgements,
        excerpts=excerpts,
        scores=domain_scores(controls, judgements),
        gaps=prioritised_gaps(controls, judgements),
    )
