# Implementation status — NextDeal

## Delivery phases

| Phase | Delivered | Verification |
|---|---|---|
| A | Local environment, persistent API, Ollama tools and advisor workflow | Live end-to-end scenarios and startup smoke test |
| B | All 15 objective implementations and evaluation protocols | Objective register with real/synthetic provenance and missed targets |
| C | Production build, tests, notebook reruns, offline artifacts, demo documentation | 17 API tests; 3 browser tests; 4 notebooks; 50 live tool scenarios; 3 live full workflows |

## Measured objective status

| Objective | Status | Evidence type |
|---|---|---|
| 1.1 — Next-product propensity | target not met | real reconstruction benchmark |
| 1.2 — Client segmentation | target met | real portfolio |
| 1.3 — Recommendation agent | target met | actual local Ollama calls; isolated scenario database |
| 2.1 — Non-payment risk | target not met | real retrospective + separate synthetic benchmark |
| 2.2 — Guarantee upselling | target not met | real held-out renewal contracts |
| 2.3 — Renewal agent | target not met | historical simulation |
| 3.1 — Intent classifier | target met | 300 synthetic English messages; grouped template CV |
| 3.2 — Grounded retrieval | awaiting evidence | real public pages + curated question/phrase benchmark |
| 3.3 — Conversation orchestrator | awaiting evidence | live application; human review pending |
| 4.1 — Value and loyalty | target not met | synthetic longitudinal simulation |
| 4.2 — LLM quality judge | awaiting evidence | live model; human comparison pending |
| 4.3 — Monitoring agent | awaiting evidence | actual Ollama report over observed test activity |
| 5.1 — Data anomalies | target met | synthetic fault injection |
| 5.2 — Identity deduplication | target met | synthetic same-type duplicate pairs |
| 5.3 — Pipeline agent | target met | synthetic controlled incident suite |

## Evidence still requiring people or unavailable data

- Genuine human draft acceptance, answer faithfulness review, and judge agreement are not manufactured from scripted tests or model labels.
- Independent human-written English intent evaluation remains needed; the synthetic rule/ML benchmark is not evidence of general performance.
- Recommendation, English retrieval, real payment-risk, guarantee coverage, and synthetic value targets remain below target where shown in the register.
- Authentic contractual PDFs and real 12-month outcome history remain unavailable.
- Timely renewal contact is measured only after advisor-approved simulated dispatch, not from the existence of scheduled actions.

## Demo entry points

- Run `./scripts/start.sh`, then open http://127.0.0.1:3000.
- Follow `docs/DEMO_GUIDE.md`.
- Open `reports/Validation2_Review.html` for an offline review handout.
- Inspect `reports/live_demo_evidence.json` and `reports/validation2_evaluation.json` for detailed evidence.

All objective workflows are implemented. This status does not assert that every academic metric or real business target has been achieved.
