"""Indexa standalone service: the assessment engine plus a no-login demo page.

Light enough for a small container (no torch, no vector DB, no auth provider):

    uvicorn indexa.server:app --host 0.0.0.0 --port $PORT     (with PYTHONPATH=src)

Environment: INDEXA_LLM_BASE_URL, INDEXA_LLM_API_KEY, INDEXA_LLM_MODEL (see llm.py),
INDEXA_DEMO_TOKEN (optional: required to start a sample run), INDEXA_DATA_DIR (cache).
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .controls import load_controls
from .documents import Document, repair_text
from .engine import assess
from .keywords import load_criteria, load_keywords
from .llm import ChatClient
from .templates import load_template_index

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = ROOT / "config" / "frameworks" / "NDI"
DATA = Path(os.getenv("INDEXA_DATA_DIR", str(Path.home() / ".cache" / "indexa")))
DOMAINS = ["6_Personal_Data_Protection", "7_Data_Classification"]
MAX_FILES, MAX_BYTES = 20, 25 * 1024 * 1024

# Public policies used for the live sample (the same documents as the evaluation).
SAMPLES = json.loads((ROOT / "config" / "samples.json").read_text(encoding="utf-8"))

app = FastAPI(title="Indexa", version="0.2.0")
_runs: dict[str, dict] = {}
_templates = None


def _controls():
    kw, crit = load_keywords(), load_criteria()
    out = []
    for d in DOMAINS:
        for c in load_controls(FRAMEWORK / f"{d}.json", domain=d):
            c.keywords, c.criteria = kw.get(c.control_id, []), crit.get(c.control_id, {})
            out.append(c)
    return out


def _pdf_pages(data: bytes) -> list[str]:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return [repair_text(p.extract_text() or "") for p in pdf.pages]


def _report(result, started: float, model: str) -> dict:
    return {
        "entity": result.entity,
        "model": model,
        "seconds": round(time.time() - started),
        "scores": [s.__dict__ for s in result.scores],
        "gaps": result.gaps,
        "judgements": [j.to_dict() for j in result.judgements.values()],
        "notice": "درجة جاهزية مؤشرية مرجحة بأولويات NDMO، وليست الدرجة الرسمية للمؤشر.",
    }


async def _run(key: str, entity: str, docs: list[Document]) -> None:
    global _templates
    _runs[key] = {"status": "running", "started": time.time()}
    client = ChatClient()
    try:
        if _templates is None:
            _templates = await asyncio.to_thread(load_template_index)
        res = await assess(entity, _controls(), docs, client, concurrency=2, templates=_templates)
        report = _report(res, _runs[key]["started"], client.model)
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / f"{key}.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        _runs[key] = {"status": "done"}
    except Exception as e:  # surfaced to the page, never swallowed
        _runs[key] = {"status": "error", "error": str(e)[:300]}
    finally:
        await client.aclose()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model_configured": bool(os.getenv("INDEXA_LLM_API_KEY"))}


@app.get("/")
def page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/samples")
def samples() -> list[dict]:
    return [
        {"id": s["id"], "entity": s["entity"], "documents": [d["title"] for d in s["documents"]]}
        for s in SAMPLES
    ]


@app.post("/api/samples/{sample_id}/run")
async def run_sample(sample_id: str, x_demo_token: str | None = Header(default=None)) -> dict:
    token = os.getenv("INDEXA_DEMO_TOKEN")
    if token and x_demo_token != token:
        raise HTTPException(403, "demo token required")
    s = next((s for s in SAMPLES if s["id"] == sample_id), None)
    if not s:
        raise HTTPException(404, "unknown sample")
    if _runs.get(sample_id, {}).get("status") == "running":
        return _runs[sample_id]
    docs = []
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as http:
        for i, d in enumerate(s["documents"]):
            data = (await http.get(d["url"])).content
            docs.append(Document(f"doc{i + 1}", d["title"], _pdf_pages(data)))
    asyncio.create_task(_run(sample_id, s["entity"], docs))
    return {"status": "running"}


@app.get("/api/reports/{key}")
def report(key: str) -> JSONResponse:
    if not key.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "bad key")
    f = DATA / f"{key}.json"
    if f.exists():
        return JSONResponse(json.loads(f.read_text(encoding="utf-8")))
    return JSONResponse(_runs.get(key, {"status": "none"}), status_code=202)


@app.post("/api/assess")
async def assess_upload(
    entity_name: str = Form(..., max_length=120), files: list[UploadFile] = File(...)
) -> dict:
    if not files or len(files) > MAX_FILES:
        raise HTTPException(400, f"1 to {MAX_FILES} PDF files")
    docs = []
    for i, f in enumerate(files):
        data = await f.read()
        if len(data) > MAX_BYTES or not data.startswith(b"%PDF"):
            raise HTTPException(400, f"{f.filename}: PDF up to 25 MB")
        docs.append(Document(f"doc{i + 1}", f.filename or f"file {i + 1}", _pdf_pages(data)))
    key = f"upload-{int(time.time() * 1000)}"
    asyncio.create_task(_run(key, entity_name, docs))
    return {"key": key, "status": "running"}
