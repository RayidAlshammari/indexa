"""Rebuild runs/predictions_v2.json (engine v2: acceptance criteria + restatement rule).

Compact transcript of the sandbox export; same item order as build_predictions.py.
rs = model flagged the evidence as restating the requirement; df = verified quote
defers the mechanism to the regulator (programmatic).
"""

import json
from pathlib import Path

from build_predictions import KEYS

D = {"M": "met", "P": "partial", "N": "not_met"}
RUNS = {
    "v2|qwen/qwen3.8-27b": dict(
        dec="NNPPPMPPPPPPNPPPPNPMPNPPPPPPPPPPPPPPNNPM",
        nr="0000000000000010100000000000000000000000",
        rs="0" * 40,
        df="0" * 40,
        citver=[
            sum(
                [
                    0,
                    0,
                    2,
                    3,
                    3,
                    3,
                    3,
                    2,
                    2,
                    2,
                    2,
                    2,
                    0,
                    2,
                    3,
                    2,
                    2,
                    0,
                    2,
                    3,
                    2,
                    0,
                    1,
                    2,
                    2,
                    3,
                    2,
                    2,
                    1,
                    1,
                    2,
                    2,
                    1,
                    2,
                    2,
                    2,
                    0,
                    0,
                    1,
                    2,
                ]
            ),
            sum(
                [
                    0,
                    0,
                    2,
                    3,
                    3,
                    3,
                    3,
                    2,
                    2,
                    2,
                    2,
                    2,
                    0,
                    2,
                    2,
                    2,
                    1,
                    0,
                    2,
                    3,
                    2,
                    0,
                    1,
                    2,
                    2,
                    3,
                    2,
                    2,
                    1,
                    1,
                    2,
                    2,
                    1,
                    2,
                    2,
                    2,
                    0,
                    0,
                    1,
                    2,
                ]
            ),
        ],
        tel={
            "calls": 40,
            "errors": 0,
            "prompt_tokens": 76796,
            "completion_tokens": 11037,
            "median_ms": 1085,
            "finish": {"stop": 40},
        },
    ),
    "v2|openai/gpt-oss-120b": dict(
        dec="NNNPPPNPNPNPNPPPNNNMNNPPNPPPNPNPPPPNNNNP",
        nr="0000000000000000000100000000000100000001",
        rs="0010001000000000000000001000000000000000",
        df="0" * 40,
        citver=[49, 45],
        tel={
            "calls": 40,
            "errors": 0,
            "prompt_tokens": 78629,
            "completion_tokens": 27471,
            "median_ms": 2558,
            "finish": {"stop": 40},
        },
    ),
    "v2|openai/gpt-oss-20b": dict(
        dec="NNNNPPNNNPNNNPPPNNNMNNNPPPPPNNNPNPPNNNNM",
        nr="0000000000000000000000000000000100000001",
        rs="0011000100010000000000000000010010000000",
        df="0000100000000000000000001000000000000000",
        citver=[40, 38],
        tel={
            "calls": 40,
            "errors": 0,
            "prompt_tokens": 78629,
            "completion_tokens": 31231,
            "median_ms": 1502,
            "finish": {"stop": 40},
        },
    ),
}


def build() -> dict:
    out = {}
    for key, r in RUNS.items():
        items = {
            k: {
                "decision": D[r["dec"][i]],
                "needs_review": r["nr"][i] == "1",
                "restates": r["rs"][i] == "1",
                "defers": r["df"][i] == "1",
            }
            for i, k in enumerate(KEYS)
        }
        # Citation totals are run-level in the transcript; spread on the first item for scoring.
        items[KEYS[0]]["n_cit"], items[KEYS[0]]["n_ver"] = r["citver"]
        out[key] = {"items": items, "telemetry": r["tel"]}
    return {"runs": out}


if __name__ == "__main__":
    p = Path(__file__).parent / "runs" / "predictions_v2.json"
    p.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", p)
