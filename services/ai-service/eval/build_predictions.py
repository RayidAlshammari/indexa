"""Rebuild runs/predictions_v1.json from the compact transcript of the sandbox export.

Order of items: entity (mewa, iu) x specification in ORDER. Decisions are
M/P/N; review flags, citation and verified-citation counts follow the same order.
Routing reflects the v2 quote verifier (exact + near-exact) re-applied to the raw
model outputs; decisions are unaffected by the verifier.
"""

import json
from pathlib import Path

ORDER = [
    "PDP.1.1",
    "PDP.1.2",
    "PDP.2.1",
    "PDP.3.2",
    "PDP.3.1",
    "PDP.4.1",
    "PDP.4.2",
    "PDP.4.4",
    "PDP.5.1",
    "PDP.4.3",
    "DC.1.1",
    "DC.1.2",
    "DC.3.1",
    "DC.3.2",
    "DC.3.3",
    "DC.3.4",
    "DC.3.5",
    "DC.4.1",
    "DC.5.1",
    "DC.2.1",
]
KEYS = [f"{e}|{c}" for e in ("mewa", "iu") for c in ORDER]
D = {"M": "met", "P": "partial", "N": "not_met"}

RUNS = {
    "new|openai/gpt-oss-120b": dict(
        dec="PNPPMPPNNPNNNPPPPNNMNNPPMPPPNPNNPPPPNNNP",
        review={"mewa|DC.3.3": ["unverified quote"], "iu|DC.3.2": ["unverified quote"]},
        cit=[
            1,
            0,
            2,
            3,
            1,
            5,
            3,
            0,
            0,
            2,
            0,
            0,
            0,
            2,
            3,
            1,
            2,
            0,
            0,
            5,
            0,
            0,
            1,
            3,
            1,
            3,
            2,
            2,
            0,
            2,
            0,
            0,
            1,
            2,
            2,
            2,
            0,
            0,
            0,
            3,
        ],
        unver={"mewa|DC.3.3": 1, "iu|DC.3.2": 1},
        telemetry={
            "calls": 47,
            "errors": 7,
            "errors_note": "7 calls exhausted rate-limit retries on the first pass and were re-run",
            "prompt_tokens": 68577,
            "completion_tokens": 23500,
            "median_ms": 1757,
            "finish": {"stop": 40},
            "max_tokens": 3000,
        },
        v1_routing={"needs_review": 11, "citations": 54, "verified": 37},
    ),
    "new|openai/gpt-oss-20b": dict(
        dec="NNPNMNPNNNNNNPPPNNNMNNNPMPPPNPNNNPPNNNNN",
        review={
            "mewa|PDP.4.1": ["unparseable model output"],
            "mewa|PDP.4.2": ["unverified quote"],
            "iu|DC.2.1": ["unparseable model output"],
        },
        cit=[
            0,
            0,
            2,
            0,
            2,
            0,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            1,
            2,
            0,
            0,
            0,
            5,
            0,
            0,
            0,
            3,
            1,
            5,
            2,
            2,
            0,
            2,
            0,
            0,
            0,
            3,
            3,
            0,
            0,
            0,
            0,
            0,
        ],
        unver={"mewa|PDP.4.2": 1},
        telemetry={
            "calls": 40,
            "errors": 0,
            "prompt_tokens": 68577,
            "completion_tokens": 26296,
            "median_ms": 1207,
            "finish": {"stop": 39, "length": 1},
            "max_tokens": 2000,
        },
    ),
    "new|qwen/qwen3.8-27b": dict(
        dec="PNPPMMPPPPPPNPPPPNPMPNPPMPPPPPPNPPPPPNPM",
        review={"mewa|DC.3.3": ["unverified quote"], "mewa|DC.3.5": ["unverified quote"]},
        cit=[
            2,
            0,
            3,
            3,
            2,
            3,
            2,
            2,
            2,
            2,
            2,
            2,
            0,
            2,
            2,
            2,
            2,
            0,
            2,
            3,
            2,
            0,
            1,
            2,
            1,
            3,
            3,
            2,
            2,
            1,
            2,
            0,
            1,
            2,
            2,
            2,
            2,
            0,
            1,
            3,
        ],
        unver={"mewa|DC.3.3": 1, "mewa|DC.3.5": 1},
        telemetry={
            "calls": 40,
            "errors": 0,
            "prompt_tokens": 66356,
            "completion_tokens": 9083,
            "median_ms": 858,
            "finish": {"stop": 40},
            "max_tokens": 3000,
        },
    ),
    # Every output ran to the token limit (repetition loops) and none parsed; also
    # re-tried at 2,400 tokens and with JSON mode on samples, same result.
    "new|allam-2-7b": dict(
        dec="N" * 40,
        review={k: ["unparseable model output"] for k in KEYS},
        cit=[0] * 40,
        unver={},
        telemetry={
            "calls": 40,
            "errors": 0,
            "prompt_tokens": 62683,
            "completion_tokens": 45158,
            "median_ms": 1119,
            "finish": {"length": 36, "stop": 4},
            "max_tokens": 1200,
            "context_window": 4096,
        },
    ),
    # Original Governance-Agent design on the same model. MEWA's PDP policy and privacy
    # notice could not be sent: one request exceeds the 8,000 tokens-per-minute limit.
    "orig|qwen/qwen3.8-27b": dict(
        dec="NNNNNNNNNNNNNMNMNNNMNNNNNNNNNNNNNNNNNNNM",
        not_evaluated=[f"mewa|{c}" for c in ORDER if c.startswith("PDP")],
        telemetry={
            "calls": 5,
            "errors": 2,
            "errors_note": "mewa_pdp and mewa_notice: request too large for the 8,000 TPM limit",
            "prompt_tokens": 19441,
            "completion_tokens": 3392,
            "median_ms": 2929,
            "finish": {"stop": 3},
            "max_tokens": 4000,
        },
    ),
}


def build() -> dict:
    out = {}
    for key, r in RUNS.items():
        items = {}
        for i, k in enumerate(KEYS):
            it = {"decision": D[r["dec"][i]]}
            if "cit" in r:
                it.update(
                    needs_review=k in r["review"],
                    reasons=r["review"].get(k, []),
                    n_cit=r["cit"][i],
                    n_ver=r["cit"][i] - r["unver"].get(k, 0),
                )
            if k in r.get("not_evaluated", []):
                it["not_evaluated"] = True
            items[k] = it
        out[key] = {"items": items, "telemetry": r["telemetry"]} | (
            {"v1_routing": r["v1_routing"]} if "v1_routing" in r else {}
        )
    return {"runs": out}


if __name__ == "__main__":
    p = Path(__file__).parent / "runs" / "predictions_v1.json"
    p.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", p)
