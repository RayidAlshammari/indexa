"""Bilingual retrieval vocabularies per specification.

Generated once per catalogue by an LLM from the specification text alone (no
entity documents), then reviewed by a governance specialist. Stored as data so
a reviewer can edit a term without touching code.
"""

from __future__ import annotations

import json
from pathlib import Path

_CONFIG = Path(__file__).resolve().parents[2] / "config"
_DIR = _CONFIG / "keywords"


def load_keywords(directory: Path = _DIR) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in sorted(Path(directory).glob("*.json")):
        for cid, v in json.loads(f.read_text(encoding="utf-8")).items():
            out[cid] = list(v.get("ar", [])) + list(v.get("en", []))
    return out


def load_criteria(directory: Path = _CONFIG / "criteria") -> dict[str, dict]:
    """Per-specification acceptance criteria (met / partial / not_met / entity_specific)."""
    out: dict[str, dict] = {}
    for f in sorted(Path(directory).glob("*.json")):
        for cid, v in json.loads(f.read_text(encoding="utf-8")).items():
            if not cid.startswith("_"):
                out[cid] = v
    return out
