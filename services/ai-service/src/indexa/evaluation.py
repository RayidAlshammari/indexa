"""Agreement metrics and the original-design baseline used in the model report.

Kept inside the package so the exact same code scores every run, whether it is
executed by pytest, the CLI, or in a browser sandbox (Pyodide).
"""

from __future__ import annotations

import json
import re
from collections import Counter

from .controls import Control

LABELS = ("not_met", "partial", "met")
_ORD = {lab: i for i, lab in enumerate(LABELS)}


def confusion(gold: list[str], pred: list[str]) -> dict[str, dict[str, int]]:
    m = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred, strict=True):
        m[g][p] += 1
    return m


def cohen_kappa(gold: list[str], pred: list[str], weighted: bool = False) -> float:
    """Cohen's kappa; with weighted=True, linear weights for the ordinal scale."""
    n = len(gold)
    if n == 0:
        return 0.0
    k = len(LABELS)

    def w(i: int, j: int) -> float:
        return abs(i - j) / (k - 1) if weighted else float(i != j)

    obs = sum(w(_ORD[g], _ORD[p]) for g, p in zip(gold, pred, strict=True)) / n
    gc, pc = Counter(gold), Counter(pred)
    exp = sum(
        w(i, j) * gc[a] * pc[b] for i, a in enumerate(LABELS) for j, b in enumerate(LABELS)
    ) / (n * n)
    return 1.0 - obs / exp if exp else 1.0


def macro_f1(gold: list[str], pred: list[str]) -> float:
    f1s = []
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred, strict=True))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred, strict=True))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred, strict=True))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def summarize(gold: list[str], pred: list[str]) -> dict:
    n = len(gold)
    pairs = list(zip(gold, pred, strict=True))
    return {
        "n": n,
        "accuracy": sum(g == p for g, p in pairs) / n if n else 0.0,
        "macro_f1": macro_f1(gold, pred),
        "kappa": cohen_kappa(gold, pred),
        "weighted_kappa": cohen_kappa(gold, pred, weighted=True),
        # Over-claiming is the costly error for a readiness tool: the entity
        # walks into the official assessment believing a gap is closed.
        "over_claim": sum(_ORD[p] > _ORD[g] for g, p in pairs),
        "critical_over_claim": sum(p == "met" and g == "not_met" for g, p in pairs),
        "under_claim": sum(_ORD[p] < _ORD[g] for g, p in pairs),
        "confusion": confusion(gold, pred),
    }


# --------------------------------------------------------------------------
# Original Governance-Agent design, reproduced for a like-for-like baseline.
# Same instructions, same 15,000-character truncation, one file at a time, all
# controls in one reply, binary Compliant / Not Compliant. The original fetched
# control details through tool calls; here the same fields are inlined, which
# gives the model the same information without a tool loop.
# --------------------------------------------------------------------------

ORIGINAL_TRUNCATION = 15000


def original_prompt(
    file_text: str, controls_raw: list[dict], idx: int, total: int, framework: str = "NDI"
) -> str:
    text = (file_text or "")[:ORIGINAL_TRUNCATION]
    details = json.dumps(
        [
            {
                "control_id": c["id"],
                "description": c.get("description", ""),
                "calculation": c.get("calculation", ""),
                "threshold": c.get("threshold", ""),
                "scale": c.get("scale", ""),
            }
            for c in controls_raw
        ],
        ensure_ascii=False,
    )
    return (
        f"You are evaluating file {idx + 1} of {total} for framework '{framework}'.\n"
        f"Field ID for this file: field_{idx + 1}.\n\n"
        "## Controls for this file (as returned by retrieve_control_details)\n"
        f"{details}\n\n"
        "Use the description, calculation, threshold, and scale from each control to understand what it requires.\n\n"
        "## Control types and decision semantics\n"
        "- If a control has a non-empty 'scale' field (e.g. Leader, Excellent, Good, Fair, Low, Unacceptable): use those levels for your decision.\n"
        "- If a control has an empty 'scale' (policy/requirement): use 'Compliant' or 'Not Compliant'.\n\n"
        "## Output format (required)\n"
        "After evaluating all controls, respond with JSON only (no extra text):\n"
        '{"control_decisions": [{"control_id": "DG.1.1", "decision": "Compliant", "rationale": "..."}, ...], '
        '"summary": "1-2 paragraph overall assessment of the file against all controls."}\n'
        "decision must match control type: use scale levels (Leader, Excellent, Good, Fair, Low, Unacceptable) when scale exists; otherwise Compliant or Not Compliant.\n\n"
        f"## File content to evaluate\n\n{text}"
    )


def parse_original(raw: str) -> dict[str, str]:
    """Map the original binary decisions onto the 3-level scale (no 'partial' exists)."""
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    s, e = raw.find("{"), raw.rfind("}")
    try:
        data = json.loads(raw[s : e + 1])
    except (json.JSONDecodeError, ValueError):
        return {}
    out = {}
    for d in data.get("control_decisions", []) or []:
        dec = str(d.get("decision", "")).strip().lower()
        if dec in ("compliant", "leader", "excellent", "good"):
            out[str(d.get("control_id"))] = "met"
        elif dec in ("fair",):
            out[str(d.get("control_id"))] = "partial"
        else:
            out[str(d.get("control_id"))] = "not_met"
    return out


def aggregate_files(per_file: list[dict[str, str]], control_ids: list[str]) -> dict[str, str]:
    """Entity view of per-file results: the best decision any file received.

    This is the most generous reading of the original output (a control is
    treated as covered if any one file was judged compliant).
    """
    out = {}
    for cid in control_ids:
        decs = [f[cid] for f in per_file if cid in f]
        out[cid] = max(decs, key=lambda d: _ORD[d]) if decs else "not_met"
    return out


def controls_by_id(controls: list[Control]) -> dict[str, Control]:
    return {c.control_id: c for c in controls}
