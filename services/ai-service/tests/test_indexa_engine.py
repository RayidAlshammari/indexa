"""Engine tests: no network, no model. A scripted fake LLM stands in for the provider."""

import asyncio
import json
import re
from pathlib import Path

from indexa.controls import Control, clean_description, load_controls
from indexa.documents import Document, chunk_document
from indexa.engine import assess
from indexa.judge import build_messages, parse_judgement
from indexa.retrieval import BM25Index
from indexa.scoring import compare_cycles, domain_scores, prioritised_gaps
from indexa.text import normalize, squash, tokenize

FRAMEWORK = Path(__file__).resolve().parents[1] / "config" / "frameworks" / "NDI"

POLICY = Document(
    doc_id="pdp",
    title="سياسة حماية البيانات الشخصية",
    pages=[
        "المقدمة\nتهدف هذه السياسة إلى حماية البيانات الشخصية لدى الوزارة.",
        "يجب على الوزارة إبلاغ الجهة المختصة خلال اثنتين وسبعين ساعة من اكتشاف تسرب البيانات الشخصية.\n"
        "يتم تعيين مسؤول حماية البيانات الشخصية ضمن مكتب إدارة البيانات.",
    ],
)


def ctrl(cid="PDP.1.1", title="Data Breach Notification", kw=("تسرب", "إبلاغ", "ساعة")):
    return Control(cid, title, "notify within 72 hours", "P1", "PDP", list(kw))


def test_normalize_folds_arabic_variants():
    assert normalize("إدارةُ البيانات") == normalize("ادارة البيانات")
    assert "ال" not in tokenize("البيانات")[0][:2]


def test_reversed_lam_alef_from_pdf_still_matches():
    from indexa.documents import repair_text

    extracted = repair_text("الجامعة اإلسالمية والحاالت")
    assert extracted.startswith("الجامعة الإس")
    assert tokenize("الجامعة الإسلامية والحالات") == tokenize(extracted)


def test_squash_ignores_spacing_and_punctuation():
    assert squash("خلال  اثنتين، وسبعين") == squash("خلال اثنتين وسبعين")


def test_chunks_keep_page_numbers():
    chunks = chunk_document(POLICY)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.chunk_id.startswith("pdp:p") for c in chunks)


def test_bm25_finds_breach_page():
    idx = BM25Index(chunk_document(POLICY))
    top = idx.search("تسرب البيانات إبلاغ ساعة", k=1)
    assert top and top[0][0].page == 2


def test_clean_description_strips_parser_noise():
    raw = "Data Breach Notification. The entity shall notify. (Priority: P1) Version History June 2020 Version 1.0"
    title, body, prio = clean_description(raw)
    assert title == "Data Breach Notification"
    assert "Version History" not in body and prio == "P1"


def test_catalogue_loads_all_pdp_and_dc():
    for name in ("6_Personal_Data_Protection", "7_Data_Classification"):
        cs = load_controls(FRAMEWORK / f"{name}.json")
        assert len(cs) >= 10
        assert all(c.priority in ("P1", "P2", "P3") for c in cs)
        assert not any("Version History" in c.requirement for c in cs)
        assert not any(re.search(r"\w{4,} [a-z]\b", c.title) for c in cs)


def test_verified_quote_and_wrong_excerpt_number_recovered():
    ex = [c for c, _ in BM25Index(chunk_document(POLICY)).search("تسرب إبلاغ", k=2)]
    raw = json.dumps(
        {
            "decision": "met",
            "confidence": 0.9,
            "rationale": "x",
            "evidence": [{"excerpt": "E9", "quote": "إبلاغ الجهة المختصة خلال اثنتين وسبعين ساعة"}],
            "missing": [],
        },
        ensure_ascii=False,
    )
    j = parse_judgement(ctrl(), raw, ex)
    assert j.decision == "met" and j.citations[0].verified and j.citations[0].page == 2
    assert not j.needs_review


def test_fabricated_quote_routes_to_review():
    ex = [c for c, _ in BM25Index(chunk_document(POLICY)).search("تسرب", k=2)]
    raw = '{"decision":"met","confidence":0.95,"rationale":"x","evidence":[{"excerpt":"E1","quote":"يتم إخطار الهيئة خلال 24 ساعة عبر البوابة"}]}'
    j = parse_judgement(ctrl(), raw, ex)
    assert j.needs_review and "no verifiable quote supports the decision" in j.review_reasons


def test_garbage_output_is_not_silently_scored():
    j = parse_judgement(ctrl(), "<think>hmm</think> I think it is met", [])
    assert j.needs_review and j.confidence == 0.0


def test_prompt_contains_requirement_and_page():
    ex = chunk_document(POLICY)[:1]
    msgs = build_messages(ctrl(), ex)
    assert "PDP.1.1" in msgs[1]["content"] and "page 1" in msgs[1]["content"]


def test_scoring_weights_by_priority_and_ranks_gaps():
    cs = [ctrl("A", kw=()), Control("B", "b", "", "P3", "PDP"), Control("C", "c", "", "P1", "PDP")]
    from indexa.judge import Judgement

    js = {
        "A": Judgement("A", "met", 1, ""),
        "B": Judgement("B", "not_met", 1, ""),
        "C": Judgement("C", "partial", 1, ""),
    }
    (s,) = domain_scores(cs, js)
    assert s.score == round(100 * (3 + 1.5) / 7, 1)
    assert [g["control_id"] for g in prioritised_gaps(cs, js)] == ["C", "B"]


def test_compare_cycles():
    d = compare_cycles({"A": "partial", "B": "met"}, {"A": "met", "B": "partial", "C": "met"})
    assert d == {"improved": ["A"], "regressed": ["B"], "unchanged": [], "new": ["C"]}


def test_end_to_end_with_fake_model():
    async def fake_llm(messages):
        content = messages[1]["content"]
        if "اثنتين وسبعين" in content:
            return '{"decision":"met","confidence":0.9,"rationale":"ok","evidence":[{"excerpt":"E1","quote":"إبلاغ الجهة المختصة خلال اثنتين وسبعين ساعة"}]}'
        return '{"decision":"not_met","confidence":0.8,"rationale":"none","evidence":[]}'

    cs = [ctrl(), ctrl("PDP.9.9", "Cookie consent banner", ("كوكيز",))]
    a = asyncio.run(assess("demo", cs, [POLICY], fake_llm))
    assert a.judgements["PDP.1.1"].decision == "met"
    assert a.judgements["PDP.9.9"].decision == "not_met"
    assert a.gaps[0]["control_id"] == "PDP.9.9"


def test_provider_failure_is_flagged():
    async def broken(_):
        raise RuntimeError("503")

    a = asyncio.run(assess("demo", [ctrl()], [POLICY], broken))
    assert a.judgements["PDP.1.1"].needs_review


def test_metrics_and_original_baseline_parsing():
    from indexa.evaluation import aggregate_files, cohen_kappa, parse_original, summarize

    gold = ["met", "partial", "not_met", "not_met"]
    assert cohen_kappa(gold, gold) == 1.0
    s = summarize(gold, ["met", "met", "met", "not_met"])
    assert s["over_claim"] == 2 and s["critical_over_claim"] == 1 and s["accuracy"] == 0.5
    f1 = parse_original(
        '```json\n{"control_decisions":[{"control_id":"PDP.3.1","decision":"Compliant"},{"control_id":"DC.1.2","decision":"Not Compliant"}]}\n```'
    )
    f2 = {"DC.1.2": "met"}
    assert aggregate_files([f1, f2], ["PDP.3.1", "DC.1.2", "X"]) == {
        "PDP.3.1": "met",
        "DC.1.2": "met",
        "X": "not_met",
    }


def test_keywords_cover_pdp_and_dc():
    from indexa.keywords import load_keywords

    kw = load_keywords()
    for name in ("6_Personal_Data_Protection", "7_Data_Classification"):
        for c in load_controls(FRAMEWORK / f"{name}.json"):
            assert len(kw.get(c.control_id, [])) >= 10, c.control_id


def test_quote_matching_grades():
    from indexa.judge import match_quote

    src = "يجب على مكتب إدارة البيانات إبالغ الجهات التنظيمية فورا ً ودون تأخير، بما ال يتجاوز 72 ساعة، من وقوع"
    assert match_quote("إبالغ الجهات التنظيمية فورا ودون تأخير", src) == "exact"
    # model silently drops a conjunction and repairs the broken PDF glyphs
    assert (
        match_quote("إبلاغ الجهات التنظيمية فورًا دون تأخير، بما لا يتجاوز 72 ساعة", src) == "near"
    )
    # elided quote: every fragment must be in the source
    assert match_quote("يجب على مكتب إدارة البيانات ... بما ال يتجاوز 72 ساعة", src) == "exact"
    assert (
        match_quote("يجب على مكتب إدارة البيانات ... خلال أربع وعشرين ساعة عبر البوابة", src)
        == "none"
    )
    # fabricated statement built from common words does not pass
    assert match_quote("تلتزم الجهة بإجراء تدقيق داخلي سنوي على الامتثال", src) == "none"
