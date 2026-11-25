"""Indexa v2 assessment endpoint: entity-level, per-specification, evidence-cited.

POST /api/v2/assessments  (multipart)
    entity_name   display name of the entity
    domains       comma-separated catalogue files to assess, e.g. "6_Personal_Data_Protection,7_Data_Classification"
    files         the entity's evidence (PDF, DOCX, PPTX, XLSX, CSV)

Every file is parsed page by page, all files are pooled into one evidence index
for the entity, and each specification is judged once against that pool.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.indexa import Document, assess, load_controls
from src.indexa.documents import repair_text
from src.indexa.keywords import load_keywords
from src.indexa.llm import ChatClient

router = APIRouter(prefix="/api/v2/assessments", tags=["assessments"])

_FRAMEWORK_DIR = Path(__file__).resolve().parents[3] / "config" / "frameworks" / "NDI"
_DOMAIN_RE = re.compile(r"^[0-9]{1,2}_[A-Za-z_]+$")
_MAX_FILES = 60


def _pages_from_upload(path: Path) -> list[str]:
    """Page-level text; falls back to one page for formats without pages."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return [repair_text(p.extract_text() or "") for p in pdf.pages]
    from src.processing.file_dispatcher import extract_text_from_file  # existing parsers

    return [repair_text(extract_text_from_file(str(path), suffix) or "")]


@router.post("")
async def create_assessment(
    entity_name: str = Form(..., min_length=1, max_length=120),
    domains: str = Form("6_Personal_Data_Protection,7_Data_Classification"),
    files: list[UploadFile] = File(...),
) -> dict:
    names = [d.strip() for d in domains.split(",") if d.strip()]
    if not names or not all(
        _DOMAIN_RE.match(n) and (_FRAMEWORK_DIR / f"{n}.json").exists() for n in names
    ):
        raise HTTPException(400, "unknown domain")
    if not files or len(files) > _MAX_FILES:
        raise HTTPException(400, f"between 1 and {_MAX_FILES} files are required")

    controls = []
    for n in names:
        controls += load_controls(_FRAMEWORK_DIR / f"{n}.json", domain=n)
    keywords = load_keywords()
    for c in controls:
        c.keywords = keywords.get(c.control_id, [])

    documents = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(files):
            target = Path(tmp) / f"{i}{Path(f.filename or 'file').suffix.lower()}"
            target.write_bytes(await f.read())
            documents.append(
                Document(f"doc{i + 1}", f.filename or f"file {i + 1}", _pages_from_upload(target))
            )

    client = ChatClient()
    try:
        result = await assess(entity_name, controls, documents, client)
    finally:
        await client.aclose()

    return {
        "entity": entity_name,
        "scores": [s.__dict__ for s in result.scores],
        "gaps": result.gaps,
        "judgements": [j.to_dict() for j in result.judgements.values()],
        "notice": "Indicative readiness score weighted by NDMO priority; not the official NDI score.",
    }
