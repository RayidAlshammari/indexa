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
# Sample reports shipped with the code, so the demo answers at once on hosts without a disk.
BUNDLED = ROOT / "config" / "reports"
DOMAINS = ["6_Personal_Data_Protection", "7_Data_Classification"]
DOMAIN_NAMES = {
    "6_Personal_Data_Protection": "حماية البيانات الشخصية",
    "7_Data_Classification": "تصنيف البيانات",
}
MAX_FILES, MAX_BYTES = 20, 25 * 1024 * 1024
# Parallel model calls per assessment: keep low on rate-limited free tiers, raise on paid ones.
CONCURRENCY = int(os.getenv("INDEXA_CONCURRENCY", "2"))

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


TITLES_AR = {
    k: v
    for k, v in json.loads((FRAMEWORK / "titles_ar.json").read_text(encoding="utf-8")).items()
    if not k.startswith("_")
}


def _catalogue() -> dict:
    """What the page needs to show each specification: title, requirement, priority, domain."""
    return {
        c.control_id: {
            "title": TITLES_AR.get(c.control_id, c.title),
            "title_en": c.title,
            "criteria": {k: v for k, v in c.criteria.items() if k in ("met", "partial", "not_met")},
            "requirement": c.requirement,
            "priority": c.priority,
            "domain": c.domain,
            "domain_name": DOMAIN_NAMES.get(c.domain, c.domain),
            "evidence_kind": c.evidence_kind,
        }
        for c in _controls()
    }


def _overall(scores: list[dict]) -> float:
    """Indicative overall readiness: domain scores averaged (each domain counts once)."""
    return round(sum(s["score"] for s in scores) / len(scores), 1) if scores else 0.0


def _enrich(report: dict) -> dict:
    """Reports cached before these fields existed still render fully."""
    report.setdefault("index", {"id": "NDI", "name": "مؤشر البيانات الوطني (نضيء)"})
    report.setdefault("catalogue", _catalogue())
    report.setdefault("overall", _overall(report.get("scores", [])))
    for s in report.get("scores", []):
        s.setdefault("domain_name", DOMAIN_NAMES.get(s["domain"], s["domain"]))
    return report


def _report(result, started: float, model: str) -> dict:
    return {
        "entity": result.entity,
        "model": model,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - started),
        "scores": [s.__dict__ for s in result.scores],
        "gaps": result.gaps,
        "judgements": [j.to_dict() for j in result.judgements.values()],
        "notice": "درجة جاهزية مؤشرية مرجحة بأولويات NDMO، وليست الدرجة الرسمية للمؤشر.",
    }


async def _run(key: str, entity: str, docs: list[Document]) -> None:
    global _templates
    _runs[key] = {"status": "running", "started": time.time(), "done": 0, "total": len(_controls())}
    client = ChatClient()

    def progress(done: int, total: int) -> None:
        _runs[key].update(done=done, total=total)

    try:
        if _templates is None:
            _templates = await asyncio.to_thread(load_template_index)
        res = await assess(
            entity,
            _controls(),
            docs,
            client,
            concurrency=CONCURRENCY,
            templates=_templates,
            on_progress=progress,
        )
        failed = [
            j
            for j in res.judgements.values()
            if any(("model call failed" in r or "unparseable" in r) for r in j.review_reasons)
        ]
        if len(failed) * 2 > len(res.judgements):  # a mostly-failed run is not an assessment
            reason = failed[0].review_reasons[0].replace("model call failed: ", "")
            # e.g. "unparseable model output": the model answered, but not with a judgement
            raise RuntimeError(
                f"model provider refused {len(failed)} of {len(res.judgements)} requests ({reason[:160]})"
            )
        report = _enrich(_report(res, _runs[key]["started"], client.model))
        DATA.mkdir(parents=True, exist_ok=True)
        (DATA / f"{key}.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        _runs[key] = {"status": "done"}
    except Exception as e:  # surfaced to the page, never swallowed
        _runs[key] = {**_runs[key], "status": "error", "error": str(e)[:300]}
    finally:
        await client.aclose()


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model_configured": bool(os.getenv("INDEXA_LLM_API_KEY")),
        "templates": "not loaded yet"
        if _templates is None
        else len(getattr(_templates, "shingles", []) or []),
    }


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


@app.get("/api/samples/{sample_id}/inspect")
async def inspect_sample(sample_id: str) -> list[dict]:
    """What the server actually reads from a sample's public PDFs (pages, characters, opening text)."""
    s = next((s for s in SAMPLES if s["id"] == sample_id), None)
    if not s:
        raise HTTPException(404, "unknown sample")
    out = []
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as http:
        for d in s["documents"]:
            r = await http.get(d["url"])
            info = {
                "title": d["title"],
                "status": r.status_code,
                "type": r.headers.get("content-type", ""),
                "bytes": len(r.content),
                "is_pdf": r.content.startswith(b"%PDF"),
            }
            if info["is_pdf"]:
                pages = _pdf_pages(r.content)
                info.update(
                    pages=len(pages), chars=sum(map(len, pages)), start=" ".join(pages)[:300]
                )
            out.append(info)
    return out


@app.get("/api/reports/{key}")
def report(key: str) -> JSONResponse:
    if not key.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "bad key")
    for f in (DATA / f"{key}.json", BUNDLED / f"{key}.json"):
        if f.exists():
            return JSONResponse(_enrich(json.loads(f.read_text(encoding="utf-8"))))
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
