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


def test_restatement_and_deferral_are_flagged():
    from indexa.judge import defers_to_regulator

    iu = "إبالغ الجهات التنظيمية فورا ً ودون تأخير، بما ال يتجاوز 72 ساعة، وفقا ً لآلليات والإجراءات التي تحددها الجهات التنظيمية."
    assert defers_to_regulator(iu)
    assert not defers_to_regulator(
        "يتولى مسؤول حماية البيانات إبلاغ سدايا عبر منصة الإبلاغ خلال 72 ساعة"
    )
    ex = chunk_document(Document("d", "t", [iu]))
    raw = json.dumps(
        {
            "decision": "met",
            "confidence": 0.9,
            "rationale": "x",
            "evidence": [
                {
                    "excerpt": "E1",
                    "quote": "بما ال يتجاوز 72 ساعة، وفقا ً لآلليات والإجراءات التي تحددها الجهات التنظيمية",
                }
            ],
            "restates_requirement": True,
            "entity_specific_details": [],
        },
        ensure_ascii=False,
    )
    j = parse_judgement(ctrl(), raw, ex)
    assert j.restates_requirement and j.defers_to_regulator and j.needs_review


def test_criteria_load_and_reach_the_prompt():
    from indexa.keywords import load_criteria

    crit = load_criteria()
    assert len(crit) == 20 and all({"met", "partial", "not_met"} <= set(v) for v in crit.values())
    c = ctrl()
    c.criteria = crit["PDP.3.1"]
    assert "Acceptance criteria" in build_messages(c, [])[1]["content"]


def test_template_copy_downgrades_copied_evidence():
    from indexa.judge import apply_template_rule
    from indexa.templates import TemplateIndex

    template = "يجب على الجهة إبلاغ الجهات التنظيمية فورا ودون تأخير بما لا يتجاوز 72 ساعة من وقوع أو اكتشاف أي حادثة تسريب"
    idx = TemplateIndex([template])
    copied = "يجب على الجامعة إبالغ الجهات التنظيمية فورا ً ودون تأخير، بما ال يتجاوز 72 ساعة، من وقوع أو اكتشاف أي حادثة تسريب"
    own = "يرفع مركز العمليات الأمنية بلاغ الحادثة عبر منصة سدايا ويعتمده وكيل الجامعة للتحول الرقمي خلال ست ساعات"
    assert idx.copy_ratio(copied) >= 0.7 and idx.copy_ratio(own) < 0.2
    ex = chunk_document(Document("d", "t", [copied + "\n" + own]))
    raw = json.dumps(
        {
            "decision": "met",
            "confidence": 0.9,
            "rationale": "x",
            "evidence": [{"excerpt": "E1", "quote": copied[:80]}],
        },
        ensure_ascii=False,
    )
    j = apply_template_rule(parse_judgement(ctrl(), raw, ex), idx)
    assert j.decision == "not_met" and j.decision_before_template_rule == "met"
    raw2 = json.dumps(
        {
            "decision": "met",
            "confidence": 0.9,
            "rationale": "x",
            "evidence": [{"excerpt": "E1", "quote": copied[:80]}, {"excerpt": "E1", "quote": own}],
        },
        ensure_ascii=False,
    )
    j2 = apply_template_rule(parse_judgement(ctrl(), raw2, ex), idx)
    assert j2.decision == "met"


def test_server_serves_page_and_samples():
    from fastapi.testclient import TestClient

    from indexa.server import app

    c = TestClient(app)
    assert c.get("/health").json()["ok"]
    assert "Indexa" in c.get("/").text
    ids = [s["id"] for s in c.get("/api/samples").json()]
    assert ids == ["mewa", "iu"]
    assert c.get("/api/reports/none-yet").status_code == 202
    assert (
        c.post(
            "/api/assess", data={"entity_name": "x"}, files=[("files", ("a.txt", b"hello"))]
        ).status_code
        == 400
    )


def test_improvement_plan_is_parsed_and_gaps_carry_points():
    from indexa.judge import Judgement

    raw = json.dumps(
        {
            "decision": "not_met",
            "confidence": 0.9,
            "rationale": "x",
            "evidence": [],
            "actions": ["أعدّ خطة", "اعتمدها", "تابعها", "زائد"],
            "evidence_to_prepare": ["الخطة المعتمدة"],
            "owner": "مكتب إدارة البيانات",
        },
        ensure_ascii=False,
    )
    j = parse_judgement(ctrl(), raw, [])
    assert len(j.actions) == 3 and j.owner == "مكتب إدارة البيانات"
    cs = [Control("A", "a", "", "P1", "PDP"), Control("B", "b", "", "P3", "PDP")]
    js = {"A": j, "B": Judgement("B", "met", 1, "")}
    (g,) = prioritised_gaps(cs, js)
    assert g["points"] == 75.0 and g["actions"][0] == "أعدّ خطة"


def test_server_serves_bundled_sample_report(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from indexa import server

    monkeypatch.setattr(server, "DATA", tmp_path / "cache")
    monkeypatch.setattr(server, "BUNDLED", tmp_path)
    (tmp_path / "mewa.json").write_text('{"entity": "x"}', encoding="utf-8")
    c = TestClient(server.app)
    body = c.get("/api/reports/mewa").json()
    assert body["entity"] == "x" and body["index"]["id"] == "NDI"
    assert body["catalogue"]["PDP.1.1"]["domain_name"] == "حماية البيانات الشخصية"
    assert body["catalogue"]["PDP.3.1"]["title"] == "الإبلاغ عن تسرب البيانات"
    assert body["catalogue"]["PDP.3.1"]["title_en"] == "Data Breach Notification"
    assert c.get("/api/reports/iu").status_code == 202


def test_engine_shrinks_evidence_when_request_too_large():
    import asyncio

    from indexa.controls import Control
    from indexa.documents import Document
    from indexa.engine import assess
    from indexa.llm import RequestTooLarge

    sizes = []

    async def llm(messages):
        n = messages[1]["content"].count("[E")
        sizes.append(n)
        if n > 3:
            raise RequestTooLarge("Request too large for model")
        return '{"decision": "not_met", "confidence": 0.9, "rationale": "x", "evidence": []}'

    pages = [f"سياسة حماية البيانات الشخصية البند {i} " * 30 for i in range(8)]
    doc = Document("doc1", "سياسة", pages)
    c = Control("PDP.1.1", "t", "حماية البيانات الشخصية", "P1", "6_PDP", ["البيانات"], "process", {})
    res = asyncio.run(assess("e", [c], [doc], llm))
    j = res.judgements["PDP.1.1"]
    assert sizes[0] > 3 and sizes[-1] <= 3
    assert not any("model call failed" in r for r in j.review_reasons)
