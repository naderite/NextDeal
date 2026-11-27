# Implementation status — NextDeal

## Generalized evaluation

See the [external benchmark guide](EXTERNAL_BENCHMARKS.md) for the source files, commands, and dataset adapters. Current results are recorded under `reports/external_*.json`.

- The advisor ranker exceeds both HitRate@3 targets on UCI CoIL's consumer reconstruction holdout.
- The public Spanish motor data exceeds the 12-month persistence threshold among matched observations. Attrition remains a limit of that result.
- The Spanish motor replay clears the renewal timing target under scripted approvals.
- The configurable intent classifier exceeds the macro-F1 target on CLINC150's in-scope holdout.
- The coverage-addition model meets the 60% suggestion target. Its 1.08× lift is below target, and contract eligibility needs separate evidence.
- The public renewal-payment proxy clears the recall target. Its 0.341 PR-AUC is below target, and the label does not identify unpaid invoices.
- The independent InsuranceQA semantic benchmark reaches 45.5% Recall@5 and 79.3% Recall@100. Faithfulness, judge agreement, and advisor acceptance still require separate evidence.

## Delivery phases

| Phase | Delivered | Verification |
|---|---|---|
| A | Local environment, persistent API, Ollama tools and advisor workflow | Live end-to-end scenarios and startup smoke test |
| B | All 15 objective implementations and evaluation protocols | Objective register with real/synthetic provenance and missed targets |
| C | Production build, tests, notebook reruns, offline artifacts, demo documentation | 17 API tests; 3 browser tests; 4 notebooks; 50 live tool scenarios; 3 live full workflows |

## Historical BH Assurance objective status

| Objective | Status | Evidence type |
|---|---|---|
| 1.1 — Next-product propensity | target not met | real reconstruction benchmark |
| 1.2 — Client segmentation | target met | real portfolio |
| 1.3 — Recommendation agent | target met | actual local Ollama calls; isolated scenario database |
| 2.1 — Non-payment risk | target not met | real retrospective + separate synthetic benchmark |
| 2.2 — Guarantee upselling | target not met | real held-out renewal contracts |
| 2.3 — Renewal agent | target not met | historical simulation |
| 3.1 — Intent classifier | awaiting evidence | synthetic template benchmark completed; independent English labels pending |
| 3.2 — Grounded retrieval | awaiting evidence | 86.6% English Recall@5 on development questions; independent labels and human faithfulness ratings pending |
| 3.3 — Conversation orchestrator | awaiting evidence | live application; human review pending |
| 4.1 — Value and loyalty | awaiting evidence | synthetic simulation completed; real dated holdings pending |
| 4.2 — LLM quality judge | awaiting evidence | live model; human comparison pending |
| 4.3 — Monitoring agent | awaiting evidence | actual Ollama report over observed test activity |
| 5.1 — Data anomalies | target met | synthetic fault injection |
| 5.2 — Identity deduplication | target met | synthetic same-type duplicate pairs |
| 5.3 — Pipeline agent | target met | synthetic controlled incident suite |

## Evidence still requiring people or unavailable data

- **Advisor workflow.** Use a dedicated runtime to record genuine approvals, edits, escalations, rejections, and simulated renewal dispatches. Rate grounding on reviewed model drafts so the evaluator can measure faithfulness and agreement.
- **Independent English evaluation.** Use `--intent-labels` with at least 300 independently labeled messages and `--retrieval-labels` with independent questions and relevant passage IDs. The development retrieval benchmark reached 86.6% English Recall@5.
- **Guarantee eligibility.** Ask the insurer for exact product and guarantee mappings with policy references. The first two candidate pairs would cover 199 of 320 contracts if confirmed. Use `--guarantee-catalogue` to measure confirmed candidate coverage. The rules still require held-out lift and coverage.
- **Real outcomes.** Payment-risk validation needs dated invoices and payment outcomes available after a defined prediction date. Twelve-month holdings validation needs dated snapshots for the same clients one calendar year apart. Use `--holdings-snapshots` for the persistence baseline.
- **Recommendations.** The current ranker reached 66.4% overall and 25.7% on long-tail products in the repeated company-grouped benchmark. Targets are 70% and 40%.

## Demo entry points

- Run `./scripts/start.sh`, then open http://127.0.0.1:3000.
- Follow `docs/DEMO_GUIDE.md`.
- Open `reports/Validation2_Review.html` for an offline review handout.
- Inspect `reports/live_demo_evidence.json` and `reports/validation2_evaluation.json` for detailed evidence.

All objective workflows are implemented. This status does not assert that every academic metric or real business target has been achieved.
