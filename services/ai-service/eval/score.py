"""Score exported model runs against the gold labels.

    python eval/score.py eval/runs/predictions_v1.json > eval/runs/metrics_v1.json

Input: {"runs": {run_key: {"items": {"entity|control": {...}}, "telemetry": {...}}}}
Each item needs at least "decision"; new-design items also carry needs_review,
n_cit (citations) and n_ver (programmatically verified citations).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from gold_pages import gold_pages  # noqa: E402

from indexa.evaluation import summarize  # noqa: E402


def load_gold(path: Path = HERE / "gold" / "labels_v1.json") -> dict[str, dict]:
    labels = json.loads(path.read_text(encoding="utf-8"))["labels"]
    return {f"{x['entity']}|{x['control_id']}": x for x in labels}


def score_run(
    items: dict[str, dict], gold: dict[str, dict], subset: list[str] | None = None
) -> dict:
    keys = sorted(subset or [k for k in gold if not items.get(k, {}).get("not_evaluated")])
    g = [gold[k]["label"] for k in keys]
    p = [items.get(k, {}).get("decision", "not_met") for k in keys]
    out = summarize(g, p)
    clear = [i for i, k in enumerate(keys) if gold[k]["certainty"] == "clear"]
    out["accuracy_clear_items"] = sum(g[i] == p[i] for i in clear) / len(clear)
    if any("needs_review" in items.get(k, {}) for k in keys):
        auto = [i for i, k in enumerate(keys) if not items[k].get("needs_review")]
        out["auto_accepted"] = len(auto)
        out["auto_accepted_accuracy"] = (
            sum(g[i] == p[i] for i in auto) / len(auto) if auto else None
        )
        out["auto_accepted_over_claim"] = sum(
            ["not_met", "partial", "met"].index(p[i]) > ["not_met", "partial", "met"].index(g[i])
            for i in auto
        )
        n_cit = sum(items[k].get("n_cit", 0) for k in keys)
        n_ver = sum(items[k].get("n_ver", 0) for k in keys)
        out["citations"] = n_cit
        out["citation_verified_rate"] = n_ver / n_cit if n_cit else None
        out["failed_calls"] = sum(
            any(str(r).startswith("model call failed") for r in items[k].get("reasons", []))
            for k in keys
        )
        out["unparseable"] = sum(
            "unparseable model output" in items[k].get("reasons", []) for k in keys
        )
    out["errors"] = [
        {"item": k, "gold": gold[k]["label"], "pred": p[i], "certainty": gold[k]["certainty"]}
        for i, k in enumerate(keys)
        if g[i] != p[i]
    ]
    return out


def main() -> None:
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    gold = load_gold()
    result = {"all_items": {}, "paired_subsets": {}}
    for key, run in data["runs"].items():
        result["all_items"][key] = score_run(run["items"], gold) | {
            "telemetry": run.get("telemetry", {})
        }
    # Runs that could not cover every item are compared with the others on the
    # same items only, so the comparison is like for like.
    for key, run in data["runs"].items():
        missing = [k for k, v in run["items"].items() if v.get("not_evaluated")]
        if not missing:
            continue
        subset = [k for k in gold if k not in missing]
        result["paired_subsets"][key] = {
            other: score_run(r["items"], gold, subset) for other, r in data["runs"].items()
        }
    result["retrieval"] = retrieval_recall(gold)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=1)


def retrieval_recall(gold: dict[str, dict]) -> dict:
    """Share of met/partial items where a gold evidence page is among the six retrieved excerpts."""
    ret = json.loads((HERE / "runs" / "retrieval_v1.json").read_text(encoding="utf-8"))
    hits, misses = 0, []
    scored = [k for k, g in gold.items() if g["label"] != "not_met" and gold_pages(g["evidence"])]
    for k in scored:
        if gold_pages(gold[k]["evidence"]) & set(ret[k]):
            hits += 1
        else:
            misses.append(k)
    return {"items": len(scored), "recall_at_6": hits / len(scored), "misses": misses}


if __name__ == "__main__":
    main()
