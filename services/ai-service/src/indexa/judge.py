"""Per-specification judgement: prompt, parsing, quote verification, routing.

The model only *proposes* a decision. Every proposal must cite verbatim text
from the retrieved evidence; the engine checks each quote against the source
and routes anything unverified or low-confidence to a human reviewer.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .controls import Control
from .documents import Chunk
from .text import squash

DECISIONS = ("met", "partial", "not_met")
REVIEW_CONFIDENCE = 0.6

SYSTEM_PROMPT = """You are a senior data-governance assessor preparing an entity for the Saudi National Data Index (NDI), assessed against the NDMO Data Management and Personal Data Protection Standards.
You judge ONE specification at a time, strictly from the evidence excerpts provided. Excerpts are mostly Arabic.

Decision rubric:
- "met": the evidence establishes the requirement AND covers its minimum elements.
- "partial": the requirement is addressed, but minimum elements are missing, or it is only stated as an intention/principle without the required process, plan, register or record.
- "not_met": the evidence does not address the requirement.
Evidence type (given with each specification):
- PROCESS: a documented policy or procedure that defines the requirement with its minimum elements is enough for "met".
- ARTIFACT: the requirement is an activity or output (assessment, plan, register, delivered training, audit report, KPIs). If the evidence only mandates or describes it, the decision is at most "partial"; "met" requires the artifact itself (e.g. the register, the report, the plan with milestones and budget).
Do not infer compliance from general statements or from the entity's name for a topic.

Return ONLY a JSON object:
{"decision": "met|partial|not_met",
 "confidence": <0.0-1.0>,
 "rationale": "<2-3 sentences in Arabic>",
 "evidence": [{"excerpt": "<excerpt id, e.g. E2>", "quote": "<exact words copied from that excerpt, 5-25 words>"}],
 "missing": ["<missing minimum element, in Arabic>"]}
Quotes must be copied character-for-character from the excerpt. Use an empty evidence list when decision is not_met."""


def build_messages(control: Control, excerpts: list[Chunk], max_chars: int = 650) -> list[dict]:
    lines = []
    for i, c in enumerate(excerpts, start=1):
        text = c.text[:max_chars]
        lines.append(f"[E{i}] document: {c.doc_title} | page {c.page}\n{text}")
    evidence = "\n\n".join(lines) if lines else "(no relevant evidence was found)"
    user = (
        f"Specification {control.control_id} ({control.priority}) — {control.title}\n"
        f"Evidence type: {control.evidence_kind.upper()}\n"
        f"Requirement: {control.requirement}\n\n"
        f"Evidence excerpts:\n{evidence}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


@dataclass
class Citation:
    excerpt: str
    quote: str
    doc_title: str = ""
    page: int = 0
    verified: bool = False


@dataclass
class Judgement:
    control_id: str
    decision: str
    confidence: float
    rationale: str
    citations: list[Citation] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    raw: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


def _extract_json(raw: str) -> dict | None:
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def parse_judgement(control: Control, raw: str, excerpts: list[Chunk]) -> Judgement:
    data = _extract_json(raw)
    if data is None:
        return Judgement(
            control.control_id,
            "not_met",
            0.0,
            "",
            needs_review=True,
            review_reasons=["unparseable model output"],
            raw=raw,
        )
    decision = str(data.get("decision", "")).strip().lower().replace(" ", "_")
    reasons = []
    if decision not in DECISIONS:
        reasons.append(f"invalid decision '{decision}'")
        decision = "not_met"
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    citations = []
    for ev in data.get("evidence") or []:
        if not isinstance(ev, dict):
            continue
        label = str(ev.get("excerpt", "")).strip().upper().lstrip("[").rstrip("]")
        quote = str(ev.get("quote", "")).strip()
        cit = Citation(excerpt=label, quote=quote)
        m = re.fullmatch(r"E(\d+)", label)
        if m and 1 <= int(m.group(1)) <= len(excerpts):
            src = excerpts[int(m.group(1)) - 1]
            cit.doc_title, cit.page = src.doc_title, src.page
            q = squash(quote)
            cit.verified = len(q) >= 8 and q in squash(src.text)
        if not cit.verified:
            # The model sometimes cites the wrong excerpt number for a real quote.
            q = squash(quote)
            for src in excerpts:
                if len(q) >= 8 and q in squash(src.text):
                    cit.doc_title, cit.page, cit.verified = src.doc_title, src.page, True
                    break
        citations.append(cit)
    if decision != "not_met" and not any(c.verified for c in citations):
        reasons.append("no verifiable quote supports the decision")
    if any(not c.verified for c in citations):
        reasons.append("unverified quote")
    if confidence < REVIEW_CONFIDENCE:
        reasons.append("low confidence")
    return Judgement(
        control_id=control.control_id,
        decision=decision,
        confidence=confidence,
        rationale=str(data.get("rationale", "")).strip(),
        citations=citations,
        missing=[str(x) for x in (data.get("missing") or []) if str(x).strip()],
        needs_review=bool(reasons),
        review_reasons=reasons,
        raw=raw,
    )
