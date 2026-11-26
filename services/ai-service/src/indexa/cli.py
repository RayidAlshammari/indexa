"""Command line: assess an entity's evidence folder and write a JSON + Markdown report.

    python -m indexa.cli --entity "My Ministry" --domains 6_Personal_Data_Protection,7_Data_Classification \
        --out report evidence/*.pdf

Model settings come from INDEXA_LLM_BASE_URL / INDEXA_LLM_API_KEY / INDEXA_LLM_MODEL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .controls import load_controls
from .documents import Document, repair_text
from .engine import assess
from .keywords import load_criteria, load_keywords
from .llm import ChatClient

FRAMEWORK_DIR = Path(__file__).resolve().parents[2] / "config" / "frameworks" / "NDI"
LABEL_AR = {"met": "مستوفى", "partial": "مستوفى جزئياً", "not_met": "غير مستوفى"}


def read_pages(path: Path) -> list[str]:
    if path.suffix.lower() == ".pdf":
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return [repair_text(p.extract_text() or "") for p in pdf.pages]
    return [repair_text(path.read_text(encoding="utf-8", errors="ignore"))]


def to_markdown(result) -> str:
    lines = [
        f"# تقرير جاهزية: {result.entity}",
        "",
        "> درجة جاهزية مؤشرية مرجحة بأولويات NDMO، وليست الدرجة الرسمية للمؤشر.",
        "",
    ]
    for s in result.scores:
        lines.append(
            f"- **{s.domain}**: {s.score}% (مستوفى {s.met} · جزئي {s.partial} · غير مستوفى {s.not_met} · للمراجعة {s.needs_review})"
        )
    lines += ["", "## أولويات الإغلاق", ""]
    for g in result.gaps[:10]:
        flag = " ⚑ يحتاج مراجعة" if g["needs_review"] else ""
        lines.append(
            f"- **{g['control_id']}** ({g['priority']}) {g['title']}: {LABEL_AR[g['decision']]}{flag}"
        )
        for m in g["missing"][:3]:
            lines.append(f"  - ينقص: {m}")
    lines += ["", "## التفاصيل", ""]
    for j in result.judgements.values():
        lines.append(f"### {j.control_id}: {LABEL_AR[j.decision]} (ثقة {j.confidence:.2f})")
        lines.append(j.rationale or "-")
        for c in j.citations:
            mark = "✓" if c.verified else "✗ غير متحقق"
            lines.append(f"> «{c.quote}» ({c.doc_title}، ص {c.page}) {mark}")
        lines.append("")
    return "\n".join(lines)


async def main_async(args) -> None:
    controls = []
    for name in args.domains.split(","):
        controls += load_controls(FRAMEWORK_DIR / f"{name.strip()}.json", domain=name.strip())
    kw = load_keywords()
    crit = load_criteria()
    for c in controls:
        c.keywords = kw.get(c.control_id, [])
        c.criteria = crit.get(c.control_id, {})
    docs = [
        Document(f"doc{i + 1}", Path(f).stem, read_pages(Path(f))) for i, f in enumerate(args.files)
    ]
    client = ChatClient(model=args.model)
    try:
        result = await assess(args.entity, controls, docs, client, concurrency=args.concurrency)
    finally:
        await client.aclose()
    out = Path(args.out)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "entity": result.entity,
                "scores": [s.__dict__ for s in result.scores],
                "gaps": result.gaps,
                "judgements": [j.to_dict() for j in result.judgements.values()],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    out.with_suffix(".md").write_text(to_markdown(result), encoding="utf-8")
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")


def main() -> None:
    p = argparse.ArgumentParser(description="Indexa NDI readiness assessment")
    p.add_argument("--entity", required=True)
    p.add_argument("--domains", default="6_Personal_Data_Protection,7_Data_Classification")
    p.add_argument("--model", default=None)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--out", default="indexa_report")
    p.add_argument("files", nargs="+")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
