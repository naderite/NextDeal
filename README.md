# NextDeal — Local advisor application

An integrated university demonstration covering **all 15 objectives from the five-member PDF**. English interface and responses, actual local Ollama calls, persistent advisor approval, and simulated communication.

## Start

From this directory:

```bash
./scripts/start.sh
```

Open **http://127.0.0.1:3000**. API docs: **http://127.0.0.1:8000/docs**. Stop with Ctrl+C. Persistent state: `runtime/nextdeal.sqlite3`.

For a fresh installation, run `./scripts/setup.sh`, start `ollama serve`, and pull `qwen3:4b-instruct` before startup. Override the model with `OLLAMA_MODEL=<installed-tag> ./scripts/start.sh`. The startup script removes AppImage variables that can otherwise break Python environment detection. It starts Ollama if necessary and verifies the selected model and application ports.

Libraries and models require internet during initial setup. Ordinary demo operation is local after artifacts have been cached. Use the instruction model rather than the thinking-only `qwen3:4b` tag tested during development.

## Advisor workflow

1. Search Clients, inspect holdings, persona, outreach score, and recommendations.
2. Start a conversation or create a case in Inbox.
3. Run the local assistant. Actual tool calls, supporting passages, model, and timing appear in the evidence panel.
4. Edit, approve, reject with reason, or escalate the draft.
5. Editing creates a new version and invalidates approval. Approval is required before simulated dispatch.
6. Inspect Outbox and add a simulated client reply to continue.
7. Review individual contracts in Renewals using the historical clock, initially **2025-08-01**.

Manual drafts remain available when inference fails and are labelled manual. No real email is sent. Recorded evidence is not silently substituted for live generation.

## Objective coverage

| Group | Features |
|---|---|
| BO1 / DSO1.1–1.3 | Product ranking, seven client personas, live recommendation tools |
| BO2 / DSO2.1–2.3 | Exploratory payment risk, association-rule guarantees, historical renewal decisions |
| BO3 / DSO3.1–3.3 | English intent classification, source-grounded answers, LangGraph conversation orchestration |
| BO4 / DSO4.1–4.3 | Explainable priority score, structured local judge and human-review capture, event-based monitoring |
| BO5 / DSO5.1–5.3 | Quality gate, within-type duplicate review, quarantine/retry, controlled artifact refresh |

Implementation does not imply every numerical target is met. Evaluation shows actual metrics, provenance, and limitations. Human acceptance and agreement require genuine human labels.

## Verification

```bash
./scripts/check.sh
# With API/frontend running against a separate test runtime:
cd frontend
npx playwright test --reporter=line
cd ..
# Actual Ollama scenario evidence:
env -u APPIMAGE -u APPDIR -u LD_PRELOAD .venv/bin/python scripts/live_validation.py
# Fresh baseline notebook execution:
env -u APPIMAGE -u APPDIR -u LD_PRELOAD .venv/bin/python scripts/run_notebooks.py
```

The notebook runner needs the `nextdeal` kernel, installed with `.venv/bin/python -m ipykernel install --user --name nextdeal --display-name NextDeal`. API tests use temporary databases. For browser checks, use a separate `NEXTDEAL_RUNTIME` directory so scripted approvals are not counted as actual advisor evidence. Browser tests use local Google Chrome headlessly.

The running app checks canonical client metadata once per day, requests a gated model refresh after a passing check, and materialises due renewal cases every minute using the demo clock. It never auto-approves or sends drafts. The daily guard persists across restarts; failed checks are logged. Disable this worker with `NEXTDEAL_MAINTENANCE=0` for scripted test environments.

Optional multilingual retrieval setup: install CPU PyTorch, then `pip install -r requirements-retrieval.txt` in the application environment and run `scripts/prepare_retrieval.py` with that Python. The encoder is saved under `models/e5-small` for local reuse. Both the encoder and cached passages are already prepared on this machine.

## Artifacts and evidence

- `IMPLEMENTATION_PLAN.md`: agreed scope and acceptance criteria.
- `docs/DEMO_GUIDE.md`: demonstration scenarios and reviewer checklist.
- `reports/validation2_evaluation.json`: all 15 objectives and measured results.
- `reports/validation2_ranking_predictions.csv`: company-grouped predictions.
- `reports/validation2_intent_predictions.csv`: template-grouped English intent results.
- `reports/validation2_retrieval_predictions.json`: question-level retrieval evidence.
- `reports/live_demo_evidence.json`: actual local LLM validation results.
- `reports/notebook_runs/execution_results.json`: fresh notebook results.

## Governance and limitations

Raw analytical extracts are unchanged. Accepted demonstration imports are recorded in a separate curated identity table and do not silently enter model training. Duplicate confirmations preserve source records and canonical mappings. Corporate entities and individual people are never merged across types.

Controlled refresh trains a candidate on analytical data, checks a fixed company holdout against popularity, verifies artifact reload, and atomically promotes only if the operational gate passes. Rejected candidates preserve the last valid artifact. This gate is not independent final model-selection evidence.

Recommendation scores rank products; they are not calibrated future purchase probabilities. Updated evaluations use raw profile labels, constant unknowns, and no history features of uncertain timing. Portfolio reconstruction does not establish future sales.

Only 12 of 320 historical contracts are unpaid. Payment-risk results are exploratory and feature timing is unverified. There are no real 12-month value outcomes or conversion-uplift measurements. Synthetic benchmarks are explicitly separated from real results.

Retrieval uses saved official French public product pages. Authentic general-condition PDFs were not located in the searched folders, package archive, original current source tree, or online searches. Public product pages do not establish contractual exclusions or guarantees. English responses cite the original French passages and must defer unsupported claims to an advisor.

The generator and judge use the same local model. Human approval and human-labelled evaluation remain necessary. Pipeline incident handling includes safe quarantine, not guaranteed automatic repair of arbitrary data.

## Structure

`backend/`: API, database, model tools, agent, governance. `frontend/`: Next.js UI and browser tests. `src/`: original analytical modules. `scripts/`: setup, startup and evaluation. `tests/`: workflow tests. `data/`: original, processed and synthetic datasets. `models/`: artifacts. `runtime/`: ignored database and logs. `reports/`: evidence. `docs/`: guides and baseline documentation.

The original README is retained as `docs/VALIDATION1_README.md`. Baseline notebooks and the four-member HTML presentation remain historical artifacts; the five-member PDF defines the implemented scope.
