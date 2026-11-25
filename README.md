# Indexa

**مستشار جاهزية المؤشرات الحكومية** · *Government index readiness advisor* (الاسم مبدئي)

Indexa reads an entity's own evidence (policies, procedures, registers, reports) and tells it, specification by specification, where it stands against the **NDMO Data Management and Personal Data Protection Standards** that the **National Data Index (NDI)** assesses. Every finding cites the exact sentence and page it rests on, the citation is checked against the source by code, and anything the model is unsure of goes to a human reviewer instead of into the score.

> Indexa gives an **indicative readiness score**. The official NDI methodology is not public, so the score is clearly labelled as indicative and never presented as a prediction of the official result.

---

## بالعربي باختصار

- **المشكلة:** الجهات الحكومية تستعد لقياس مؤشر البيانات الوطني يدوياً، فتراجع عشرات الوثائق مقابل 191 مواصفة. يستغرق ذلك أسابيع من وقت المستشارين، والنتيجة تختلف من مقيّم لآخر.
- **الحل:** يقيّم Indexa كل مواصفة على **كل** وثائق الجهة مجتمعة. يخرج بقرار (مستوفى / جزئي / غير مستوفى)، ومعه اقتباس حرفي ورقم الصفحة يتحقق منهما الكود، وما ينقص لاستيفاء المواصفة. ثم يرتب الفجوات حسب أولوية NDMO، ويقارن بين دورات القياس.
- **الحوكمة أولاً:** أي قرار بلا اقتباس متحقق منه، أو بثقة منخفضة، يُحال لمراجع بشري. هكذا لا يدخل في النتيجة شيء لا يمكن تتبعه.
- **السيادة:** طبقة النماذج تعمل مع أي خدمة متوافقة مع OpenAI: Groq، أو HUMAIN، أو Azure، أو خادم vLLM داخل الجهة. البحث عن الأدلة (BM25) يعمل محلياً على المعالج دون تنزيل نماذج، وبلا قاعدة بيانات متجهات.

---

## How an assessment works

```
Entity evidence (PDF/DOCX/...)                NDMO catalogue (191 specifications)
        │ page-level text, Arabic repair                │ cleaned text, priority, evidence type,
        ▼                                               ▼ bilingual retrieval vocabulary
  page-anchored chunks ──► BM25 index ◄── one query per specification
                                │ top excerpts with doc + page
                                ▼
                 LLM judge (one specification at a time)
                 JSON: decision · confidence · rationale (AR) · quotes · missing elements
                                │
                                ▼
         quote verification (normalised substring match against the excerpt)
         routing: unverified quote / low confidence / parse failure → human review
                                │
                                ▼
   indicative score (P1=3, P2=2, P3=1 · met=1, partial=0.5) · gaps ranked by score gain
   cycle-over-cycle comparison · JSON + Markdown report · API
```

**Evidence types.** Some specifications ask for a *process* (a documented, approved procedure is enough). Others ask for an *artifact*: an assessment, a plan, a register, delivered training, an audit report. For those, a policy that only *mandates* the artifact is at most "partial". This one rule is the biggest difference between a readiness tool and a keyword matcher, and it is the commonest way entities over-estimate their readiness.

## What changed from the original Governance-Agent

Indexa started as a fork of [Governance-Agent](https://github.com/AFAskar/Governance-Agent) (MIT; history and notice kept). The evaluation core was rebuilt after a review found design issues that a readiness tool cannot have:

| Original | Indexa |
|---|---|
| Each file judged on its own, so a policy is marked "not compliant" for things another document covers | Each specification judged once against **all** of the entity's evidence |
| Only the first 15,000 characters of each file are read | Whole documents, chunked by page |
| All of a domain's controls answered in one reply, capped at 25 per domain | One focused judgement per specification, with no cap |
| Vector RAG over the *framework*, none over the *evidence* | Retrieval over the evidence, with page citations |
| Binary Compliant / Not compliant | met · partial · not met, plus the missing elements |
| No citations, no verification | Verbatim quotes, checked against the source by code |
| No score, no gap ranking, no comparison between cycles | Indicative priority-weighted score, ranked gaps, cycle diff |
| Groq + OpenRouter + WorkOS + an 8 GB server (torch, embeddings, Qdrant) | Engine in pure Python; any OpenAI-compatible model, including in-Kingdom or on-premise |
| Control text polluted by PDF table residue ("Version History …") | Cleaned catalogue with priority and evidence type |

The original web portal (TanStack Start, tRPC, Drizzle, WorkOS) and its v1 API are still in the repository. The new engine is exposed alongside them as `POST /api/v2/assessments` and as a CLI.

## Evaluation

<!-- EVALUATION-SUMMARY -->

The full method, per-model results, failure analysis and the reviewable gold labels are in [`docs/EVALUATION.md`](docs/EVALUATION.md) and [`services/ai-service/eval/`](services/ai-service/eval).

## Quick start (engine only)

```bash
cd services/ai-service
pip install httpx pdfplumber
export INDEXA_LLM_BASE_URL=https://api.groq.com/openai/v1   # or HUMAIN / Azure / your vLLM
export INDEXA_LLM_API_KEY=...                               # never commit keys
export INDEXA_LLM_MODEL=openai/gpt-oss-120b

PYTHONPATH=src python -m indexa.cli --entity "My Entity" \
  --domains 6_Personal_Data_Protection,7_Data_Classification \
  --out report evidence/*.pdf            # writes report.json and report.md

PYTHONPATH=src pytest -q tests/test_indexa_engine.py -o addopts=""   # no network, no model
```

API: run the service (`uv run python run.py`) and `POST /api/v2/assessments` with `entity_name`, `domains` and `files` (multipart). It needs the same `INDEXA_LLM_*` variables and the existing `AI_SERVICE_KEY` header.

## Repository map

```
services/ai-service/
├── src/indexa/              # the assessment engine (pure Python)
│   ├── text.py              # Arabic/English normalisation, tokenisation
│   ├── documents.py         # page-anchored chunks, PDF Arabic repair
│   ├── retrieval.py         # BM25 evidence retrieval
│   ├── controls.py          # catalogue loading, cleaning, priority, evidence type
│   ├── judge.py             # prompt, JSON parsing, quote verification, review routing
│   ├── scoring.py           # indicative score, ranked gaps, cycle comparison
│   ├── engine.py            # entity assessment
│   ├── llm.py               # OpenAI-compatible provider client
│   ├── evaluation.py        # agreement metrics + original-design baseline
│   └── cli.py
├── config/frameworks/NDI/   # NDMO catalogue (12 domains)
├── config/keywords/         # bilingual retrieval vocabularies (reviewable data)
├── eval/                    # gold labels, runs, scoring, engine fingerprint
└── src/api/routers/assessments.py   # POST /api/v2/assessments
apps/, packages/             # original web portal (unchanged)
docs/EVALUATION.md           # model comparison report
```

## Limits

- Covers the PDP and DC domains end to end today. The other ten domains load from the same catalogue, but their evidence types and vocabularies still need specialist review.
- The gold labels are a first draft by Indexa, under review by a governance specialist with SDAIA experience. They were committed before any model ran.
- Policies alone rarely prove "met" for artifact specifications. Indexa says so instead of guessing.

## License

MIT. See [LICENSE](LICENSE). Original work © Governance Agent Contributors; Indexa changes © Indexa contributors.
