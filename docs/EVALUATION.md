# Indexa model evaluation, v1 (November 2026)

**Status: draft.** The gold labels were written by Indexa (Claude) and are under review by a governance specialist with SDAIA experience. Every number below will be recomputed after that review, from the stored model outputs. No new model calls are needed for that.

## Result in one paragraph

On 40 entity-level items (20 NDMO specifications × 2 Saudi government entities), **Qwen3.8-27B** agreed with the gold labels on 85% of items and **gpt-oss-120b** on 75%. On 40 items that difference is not statistically significant (exact McNemar p = 0.39). Neither model ever called a missing requirement "met". The bigger effect is the design: **on the same model, the original Governance-Agent design reached 33% and the Indexa design 83%** on the 30 items both could run (p = 0.003). The operating recommendation is two independent judges, with disagreements and unverifiable quotes routed to a reviewer. On this set that left 25 of 40 items automated at 92% agreement.

## Set-up

| | |
|---|---|
| Entities | Ministry of Environment, Water and Agriculture (PDP policy v3.3, Data Classification policy v2.2, Privacy Notice v1.0; 95 pages)<br>Islamic University of Madinah (PDP policy v1.0, Data Classification policy v1.0; 25 pages) |
| Specifications | All of PDP (10) and DC (10) |
| Gold labels | Read in full and labelled before any model ran (commit `3344e1d`): 5 met / 25 partial / 10 not met; 31 clear, 9 borderline |
| Rubric | *Process* specifications: a documented policy with the minimum elements is "met". *Artifact* specifications (plans, registers, assessments, delivered training, KPIs): a policy that only mandates the artifact is at most "partial". |
| Same for every model | Same 6 BM25 excerpts (≤ 650 characters each, with page), same prompt, temperature 0 |
| Retrieval | A gold evidence page was among the 6 excerpts for 26 of 30 items that have evidence (87%) |

## Results

| Model (Indexa design) | Agreement (95% CI) | Weighted κ | Over-claims (critical) | Under-claims | Verified quotes | Routed to review | Median latency | Cost, 40 judgements |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | 85% (71–93%) | 0.72 | 5 (0) | 1 | 68 / 70 | 2 | 0.9 s | $0.089 |
| gpt-oss-120b | 75% (60–86%) | 0.60 | 1 (0) | 9 | 52 / 54 | 2 | 1.8 s | $0.024 |
| gpt-oss-20b | 62.5% (47–76%) | 0.43 | 0 (0) | 15 | 38 / 39 | 3 | 1.2 s | $0.013 |
| ALLaM-2-7B | no parseable output (0 / 40) | — | — | — | — | 40 | 1.1 s | — |
| *Original design, Qwen3.8-27B (30 items)* | *33% (19–51%)* | *0.20* | *2 (0)* | *18* | *no citations* | — | *2.9 s* | — |

- An **over-claim** is a decision above the gold label. A **critical** over-claim is "met" where gold is "not met".
- Costs use Groq list prices: gpt-oss-120b $0.15 / $0.60 and Qwen3.8-27B $0.80 / $4.00 per million input / output tokens. A full 191-specification assessment of one entity comes to about $0.43 with Qwen and $0.12 with gpt-oss-120b.
- Two judges combined (Qwen first, gpt-oss-120b second): the judges agreed on 28 of 40 items, with 93% agreement with gold on those. Routing disagreements and unverified quotes to review leaves 25 items automated at 92%.

## Why the original design fails

The original design was reproduced faithfully: same instructions, one file per request against its domain's controls, text cut at 15,000 characters, and one binary reply.

1. **Truncation removes evidence.**
   - The University's 72-hour breach clause starts at character 15,921, so PDP.3.1 was judged "not compliant".
   - The Ministry's copies of the same clause sit at characters 18,285 and 36,492.
   - The University's classification steps (inventory, impact assessment, review) sit between characters 19,219 and 23,192.
2. **There is no "partial".** Of the 19 items whose gold label is partial, 17 became "not compliant" and 2 became "compliant". The design marks 36 of 40 items "not compliant".
3. **Evidence is never pooled across files.**
4. **There are no citations,** so a reviewer cannot check any decision.
5. **It does not fit the free tier.** The Ministry's PDP policy and its notice could not even be sent: one request exceeds the 8,000 tokens-per-minute limit.

## Failure patterns in the Indexa design

- **Retrieval misses:** 4 of 30 items. They explain 3 of gpt-oss-120b's 10 errors. The retriever was deliberately not tuned on this set.
- **Qwen leans generous:** all 5 of its over-claims are "partial" for a "not met" item. Example: it read a personal-data classification matrix as evidence of DC.1.2 prioritisation.
- **gpt-oss leans strict:** it marks policy-mandated artifacts "not met" instead of "partial". This accounts for 13 of gpt-oss-20b's 15 errors.
- **Rationale adds facts not in the evidence.** Example: one Qwen rationale named SDAIA where the text says "the responsible authorities in the Kingdom". Quote verification checks quotes, not rationales.
- **Valid but uninformative quotes,** such as a document's title line.
- **Format failures.**
  - gpt-oss-20b produced 2 unparseable replies. They were routed to review and scored as "not met".
  - ALLaM-2-7B looped until the token limit on all 40 items and invented excerpt numbers. Its 4,096-token context leaves little room for the task. Samples re-tried at 2,400 tokens and with JSON mode failed the same way.
- **PDF text quality.** The University's PDFs store lam-alef reversed. The engine repairs and normalises this for retrieval and verification.

## Not tested

- **llama-3.3-70b-versatile** (the original model) is not available on this Groq account; Groq now offers it to enterprise customers only.
- **Azure models:** no key.
- **The original design on gpt-oss-120b:** blocked by the free-tier 8,000 tokens-per-minute limit.

## Reproducibility

- **Data:**
  - `services/ai-service/eval/gold/labels_v1.json`: gold labels.
  - `eval/runs/predictions_v1.json` (rebuilt by `eval/build_predictions.py`): every decision.
  - `eval/runs/metrics_v1.json`: metrics, from `python eval/score.py eval/runs/predictions_v1.json`.
- **Where the runs happened:** the build container could not reach the model API, so the runs used the same engine inside a browser sandbox (Pyodide).
- **Code identity:** `eval/runs/engine_fingerprint_v1.txt` holds the AST fingerprints of the code that ran. They match the repository; check them with `python eval/engine_fingerprint.py`.
- **Quote verifier v2:** added after the first run (it accepts near-exact matches and elided quotes) and re-applied to the stored outputs. Decisions did not change. For gpt-oss-120b, items routed to review fell from 11 to 2.
- **Limits:**
  - 40 items, 2 entities, policies only, and one labeller (draft).
  - One run per model.
  - Qwen3.8-27B is a preview model on Groq.
