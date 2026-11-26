"""NDMO specification catalogue: loading, cleaning and priorities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_PRIORITY = re.compile(r"\(Priority:\s*(P[1-3])\)")
# Table-layout residue the PDF parser picks up after the specification body.
_NOISE = re.compile(
    r"\s*(Version History|Dependencies\s*-|Domain Name|Domain ID|As specified by NCA).*$", re.S
)

PRIORITY_WEIGHT = {"P1": 3.0, "P2": 2.0, "P3": 1.0}

# What an assessor needs to see to accept a specification as met.
#   "process":  a documented, approved policy/procedure that defines the requirement
#               with its minimum elements is sufficient evidence.
#   "artifact": the requirement is an activity or an output (an assessment, plan,
#               register, delivered training, audit report, KPI set). A policy that only
#               mandates it is at most "partial"; "met" needs the artifact itself.
# Drafted by Indexa for specialist review; extend per domain as catalogues are onboarded.
EVIDENCE_KIND = {
    "PDP.1.1": "artifact",
    "PDP.1.2": "artifact",
    "PDP.2.1": "artifact",
    "PDP.3.1": "process",
    "PDP.3.2": "process",
    "PDP.4.1": "process",
    "PDP.4.2": "process",
    "PDP.4.3": "artifact",
    "PDP.4.4": "artifact",
    "PDP.5.1": "artifact",
    "DC.1.1": "artifact",
    "DC.1.2": "artifact",
    "DC.2.1": "process",
    "DC.3.1": "artifact",
    "DC.3.2": "artifact",
    "DC.3.3": "artifact",
    "DC.3.4": "artifact",
    "DC.3.5": "artifact",
    "DC.4.1": "artifact",
    "DC.5.1": "artifact",
}


@dataclass
class Control:
    control_id: str
    title: str
    requirement: str
    priority: str = "P2"
    domain: str = ""
    keywords: list[str] = field(default_factory=list)
    evidence_kind: str = "process"
    criteria: dict = field(default_factory=dict)

    @property
    def weight(self) -> float:
        return PRIORITY_WEIGHT.get(self.priority, 2.0)

    @property
    def query(self) -> str:
        return " ".join([self.title, *self.keywords])


def clean_description(raw: str) -> tuple[str, str, str]:
    """Split a raw catalogue description into (title, requirement, priority)."""
    m = _PRIORITY.search(raw)
    priority = m.group(1) if m else "P2"
    text = _PRIORITY.sub("", raw)
    text = _NOISE.sub("", text).strip()
    title, _, body = text.partition(". ")
    # Rejoin words the PDF parser split at a line break ("Classificatio n", "Assessment s").
    title = re.sub(r"\b(\w{4,}) ([a-z])\b", r"\1\2", title.strip())
    return title, body.strip() or text, priority


def load_controls(path: Path, domain: str | None = None) -> list[Control]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for c in data.get("controls", []):
        title, body, priority = clean_description(c.get("description", ""))
        out.append(
            Control(
                control_id=c["id"],
                title=title,
                requirement=body,
                priority=c.get("priority") or priority,
                domain=domain or Path(path).stem,
                keywords=list(c.get("keywords", [])),
                evidence_kind=c.get("evidence_kind") or EVIDENCE_KIND.get(c["id"], "process"),
            )
        )
    return out


def attach_keywords(controls: list[Control], keywords: dict[str, list[str]]) -> None:
    for c in controls:
        c.keywords = keywords.get(c.control_id, c.keywords)
