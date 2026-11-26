"""Readiness score, prioritised gaps and cycle-over-cycle comparison.

The score is an *indicative readiness score*, not the official NDI score: the
official calculation methodology is not public. It weights each specification
by its published NDMO priority (P1=3, P2=2, P3=1) and credits met=1,
partial=0.5, not_met=0.
"""

from __future__ import annotations

from dataclasses import dataclass

from .controls import Control
from .judge import Judgement

CREDIT = {"met": 1.0, "partial": 0.5, "not_met": 0.0}


@dataclass
class DomainScore:
    domain: str
    score: float  # 0-100
    met: int
    partial: int
    not_met: int
    needs_review: int


def domain_scores(controls: list[Control], judgements: dict[str, Judgement]) -> list[DomainScore]:
    by_domain: dict[str, list[Control]] = {}
    for c in controls:
        by_domain.setdefault(c.domain, []).append(c)
    out = []
    for domain, cs in by_domain.items():
        total = sum(c.weight for c in cs) or 1.0
        got = sum(
            c.weight * CREDIT[judgements[c.control_id].decision]
            for c in cs
            if c.control_id in judgements
        )
        decisions = [judgements[c.control_id] for c in cs if c.control_id in judgements]
        out.append(
            DomainScore(
                domain=domain,
                score=round(100 * got / total, 1),
                met=sum(j.decision == "met" for j in decisions),
                partial=sum(j.decision == "partial" for j in decisions),
                not_met=sum(j.decision == "not_met" for j in decisions),
                needs_review=sum(j.needs_review for j in decisions),
            )
        )
    return out


def prioritised_gaps(controls: list[Control], judgements: dict[str, Judgement]) -> list[dict]:
    """Unmet and partial specifications, highest score gain first.

    Gain = weight x missing credit, so a P1 that is not met (3.0) ranks above a
    P1 that is partial (1.5), which ranks above a P3 that is not met (1.0).
    "points" is the same gain on the 0-100 domain scale: how far closing this
    gap alone would raise the domain's indicative score.
    """
    totals: dict[str, float] = {}
    for c in controls:
        totals[c.domain] = totals.get(c.domain, 0.0) + c.weight
    gaps = []
    for c in controls:
        j = judgements.get(c.control_id)
        if not j or j.decision == "met":
            continue
        gain = c.weight * (1 - CREDIT[j.decision])
        gaps.append(
            {
                "control_id": c.control_id,
                "title": c.title,
                "domain": c.domain,
                "priority": c.priority,
                "decision": j.decision,
                "gain": gain,
                "points": round(100 * gain / (totals[c.domain] or 1.0), 1),
                "missing": j.missing,
                "actions": getattr(j, "actions", []),
                "evidence_to_prepare": getattr(j, "evidence_to_prepare", []),
                "owner": getattr(j, "owner", ""),
                "needs_review": j.needs_review,
            }
        )
    return sorted(gaps, key=lambda g: (-g["gain"], g["priority"], g["control_id"]))


def compare_cycles(previous: dict[str, str], current: dict[str, str]) -> dict[str, list[str]]:
    """Specifications that improved, regressed or stayed the same between two cycles."""
    rank = {"not_met": 0, "partial": 1, "met": 2}
    out: dict[str, list[str]] = {"improved": [], "regressed": [], "unchanged": [], "new": []}
    for cid, dec in current.items():
        if cid not in previous:
            out["new"].append(cid)
        elif rank[dec] > rank[previous[cid]]:
            out["improved"].append(cid)
        elif rank[dec] < rank[previous[cid]]:
            out["regressed"].append(cid)
        else:
            out["unchanged"].append(cid)
    return out
