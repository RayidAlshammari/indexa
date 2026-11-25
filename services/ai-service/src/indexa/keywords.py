"""Bilingual retrieval vocabularies per specification.

Generated once per catalogue by an LLM from the specification text alone (no
entity documents), then reviewed by a governance specialist. Stored as data so
a reviewer can edit a term without touching code.
"""

from __future__ import annotations

import json
from pathlib import Path

_DIR = Path(__file__).resolve().parents[2] / "config" / "keywords"


def load_keywords(directory: Path = _DIR) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in sorted(Path(directory).glob("*.json")):
        for cid, v in json.loads(f.read_text(encoding="utf-8")).items():
            out[cid] = list(v.get("ar", [])) + list(v.get("en", []))
    return out
