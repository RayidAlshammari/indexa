"""Per-specification judgement: prompt, parsing, quote verification, routing.

The model only *proposes* a decision. Every proposal must cite verbatim text
from the retrieved evidence; the engine checks each quote against the source
and routes anything unverified or low-confidence to a human reviewer.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

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
Restated requirements: if the evidence only restates the specification or the national regulation (for example "notify the regulator within 72 hours according to the mechanisms set by the regulator") and adds nothing specific to the entity - no named role or unit, steps, channel, system, template, timeline or results - the decision is "not_met" and "restates_requirement" is true. Copying the standard is not implementing it.
Related work: if the entity has done related work but not the specific required element, the decision is "partial", not "not_met".
When acceptance criteria are given for the specification, apply them; they take precedence over the general rubric.
Do not infer compliance from general statements or from the entity's name for a topic.

Return ONLY a JSON object:
{"decision": "met|partial|not_met",
 "confidence": <0.0-1.0>,
 "rationale": "<2-3 sentences in Arabic>",
 "evidence": [{"excerpt": "<excerpt id, e.g. E2>", "quote": "<exact words copied from that excerpt, 5-25 words>"}],
 "missing": ["<missing minimum element, in Arabic>"],
 "restates_requirement": <true|false>,
 "entity_specific_details": ["<entity-specific detail found in the evidence, in Arabic>"]}
Quotes must be copied character-for-character from the excerpt. Use an empty evidence list when decision is not_met."""


def _criteria_block(control: Control) -> str:
    c = control.criteria or {}
    if not c:
        return ""
    lines = ["Acceptance criteria:"]
    for key in ("met", "partial", "not_met"):
        if c.get(key):
            lines.append(f"- {key}: {c[key]}")
    if c.get("entity_specific"):
        lines.append("- entity-specific details look like: " + "، ".join(c["entity_specific"]))
    return "\n".join(lines) + "\n"


# Phrases that hand the mechanism back to the regulator instead of defining it.
# Matched on normalised, space-free text so PDF glyph and spacing noise do not matter.
DEFERRAL_PHRASES = (
    "والإجراءات التي تحددها",
    "حسب ما تحدده الجهات التنظيمية",
    "وفقا لما تحدده الجهة التنظيمية",
    "as specified by the regulatory authority",
)
_DEFERRAL_KEYS = [squash(p) for p in DEFERRAL_PHRASES]


def defers_to_regulator(text: str) -> bool:
    t = squash(text)
    return any(k in t for k in _DEFERRAL_KEYS)


def build_messages(control: Control, excerpts: list[Chunk], max_chars: int = 650) -> list[dict]:
    lines = []
    for i, c in enumerate(excerpts, start=1):
        text = c.text[:max_chars]
        lines.append(f"[E{i}] document: {c.doc_title} | page {c.page}\n{text}")
    evidence = "\n\n".join(lines) if lines else "(no relevant evidence was found)"
    user = (
        f"Specification {control.control_id} ({control.priority}) — {control.title}\n"
        f"Evidence type: {control.evidence_kind.upper()}\n"
        f"Requirement: {control.requirement}\n"
        f"{_criteria_block(control)}\n"
        f"Evidence excerpts:\n{evidence}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


MIN_QUOTE = 8  # squashed characters; shorter fragments prove nothing
NEAR_RATIO = 0.9  # share of the quote found, in order, inside one compact source window


def match_quote(quote: str, source: str) -> str:
    """How a quote relates to its source: "exact", "near" or "none".

    "exact": the normalised quote is a substring of the normalised source.
    "near":  at least 90% of it appears in order inside a window no longer than
             1.3x the quote (a dropped conjunction, a corrected PDF glyph).
    Quotes elided with "..." are checked fragment by fragment; the weakest wins.
    """
    src = squash(source)
    parts = [squash(x) for x in re.split(r"\.{3,}|…", quote or "")]
    parts = [x for x in parts if len(x) >= MIN_QUOTE]
    if not parts:
        return "none"
    grades = []
    for q in parts:
        if q in src:
            grades.append("exact")
            continue
        blocks = [
            b
            for b in SequenceMatcher(None, q, src, autojunk=False).get_matching_blocks()
            if b.size >= 4
        ]
        matched = sum(b.size for b in blocks)
        span = (blocks[-1].b + blocks[-1].size - blocks[0].b) if blocks else 0
        grades.append(
            "near" if blocks and matched / len(q) >= NEAR_RATIO and span <= 1.3 * len(q) else "none"
        )
    return "none" if "none" in grades else ("near" if "near" in grades else "exact")


@dataclass
class Citation:
    excerpt: str
    quote: str
    doc_title: str = ""
    page: int = 0
    verified: bool = False
    match: str = "none"  # exact | near | none


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
    restates_requirement: bool = False
    specific_details: list[str] = field(default_factory=list)
    defers_to_regulator: bool = False
    template_copy: float = 0.0
    decision_before_template_rule: str = ""
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


def apply_template_rule(j: Judgement, index) -> Judgement:
    """Copied national-template text is not implementation: downgrade to not_met.

    Applies when the decision is met or partial and every verified quote is
    template text. A single entity-specific quote keeps the model's decision.
    """
    if not index or j.decision == "not_met":
        return j
    verified = [c for c in j.citations if c.verified]
    if not verified:
        return j
    ratios = [index.copy_ratio(c.quote) for c in verified]
    j.template_copy = max(ratios)
    if all(r >= TEMPLATE_COPY_THRESHOLD for r in ratios):
        j.decision_before_template_rule = j.decision
        j.decision = "not_met"
        j.restates_requirement = True
        j.review_reasons.append("evidence is copied from a national template")
        j.needs_review = True
    return j


TEMPLATE_COPY_THRESHOLD = 0.7


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
        # The cited excerpt first; the model sometimes cites the wrong number for a real quote.
        order = list(excerpts)
        if m and 1 <= int(m.group(1)) <= len(excerpts):
            order.insert(0, excerpts[int(m.group(1)) - 1])
        graded = [(match_quote(quote, src.text), src) for src in order]
        best = next((g for g in graded if g[0] == "exact"), None) or next(
            (g for g in graded if g[0] == "near"), None
        )
        if best:
            cit.match, cit.verified = best[0], True
            cit.doc_title, cit.page = best[1].doc_title, best[1].page
        citations.append(cit)
    if decision != "not_met" and not any(c.verified for c in citations):
        reasons.append("no verifiable quote supports the decision")
    if any(not c.verified for c in citations):
        reasons.append("unverified quote")
    if confidence < REVIEW_CONFIDENCE:
        reasons.append("low confidence")
    restates = bool(data.get("restates_requirement"))
    details = [str(x) for x in (data.get("entity_specific_details") or []) if str(x).strip()]
    defers = any(defers_to_regulator(c.quote) for c in citations if c.verified)
    if restates and decision != "not_met":
        reasons.append("flagged as restating the requirement but not judged not_met")
    if decision == "met" and defers and not details:
        reasons.append("evidence defers the mechanism to the regulator")
    return Judgement(
        control_id=control.control_id,
        decision=decision,
        confidence=confidence,
        rationale=str(data.get("rationale", "")).strip(),
        citations=citations,
        missing=[str(x) for x in (data.get("missing") or []) if str(x).strip()],
        needs_review=bool(reasons),
        review_reasons=reasons,
        restates_requirement=restates,
        specific_details=details,
        defers_to_regulator=defers,
        raw=raw,
    )
