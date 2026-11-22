# NextDeal — Final Implementation Plan

## 1. Goal and deadlines

Build a local university demonstration of NextDeal covering **all 15 data-science objectives in the five-member PDF presentation**. The application must connect the existing analytical modules to an actual LLM-powered workflow and a simple advisor interface.

- **Work review:** Monday, November 23, 2026.
- **Final validation:** Saturday, November 28, 2026.
- **Primary deliverable:** an integrated, reproducible demonstration with evidence for each objective.
- **Success standard:** functioning workflows and honest evaluations, with achieved, missed, simulated, and blocked targets clearly distinguished.

This document is a plan only. Implementation, installations, model downloads, and application changes have not started.

## 2. Confirmed decisions

| Area | Decision |
|---|---|
| Objective scope | All 15 objectives from the PDF, grouped under five business objectives |
| Interface language | English |
| Generated responses | English, including when retrieved sources are French |
| LLM hosting | Local Ollama models; install a suitable model if needed |
| Dependencies | Libraries needed for implementation may be installed |
| Email | Simulated inbox and outbox |
| Advisor control | Review, edit, approve, reject, or escalate generated drafts |
| Synthetic data | Permitted, with provenance and results separated from real data |
| Identity matching | Deduplicate within corporate clients and within individual clients; do not merge companies with people |
| Framework | No mandatory framework has been specified; use the proposed stack below |

## 3. Existing baseline and gaps

Baseline package:

`/home/naderite/Downloads/Folders/NextDeal-BH-Assurance_Validation1`

Existing assets include:

- Four notebooks with saved successful executions and no recorded errors; a fresh execution is still required.
- Corporate portfolio data: 299 companies, 440 reconstruction episodes, and a 46-product catalogue with positive examples for 34 products.
- Individual renewal data: 320 contracts, including only 12 unpaid examples.
- Evaluated recommendation baseline: HitRate@3 approximately 0.686 overall and 0.329 on long-tail products.
- Implemented but incompletely evaluated hybrid recommendation, segmentation, guarantee-rule, intent-classification, and retrieval modules.
- Saved BH Assurance public product pages and a local retrieval embedding cache.

Missing integration includes actual LLM tool calls, conversation orchestration, persistent workflow state, advisor approval, simulated dispatch, monitoring, and data-governance workflows.

Known evaluation limitations include imputation learned before cross-validation, uncertain availability of contract-history features, missing longitudinal outcomes, limited payment-risk positives, and incomplete labelled NLP benchmarks.

Preserve this package as the validation baseline. Create a version-controlled working copy for implementation, with original raw extracts retained unchanged.

## 4. Proposed architecture

| Component | Choice | Responsibility |
|---|---|---|
| Advisor interface | Next.js | Client lookup, case review, renewals, approvals, and reports |
| Backend | FastAPI and Pydantic | Typed endpoints, model tools, workflow actions, and validation |
| Orchestration | LangGraph | Routing, bounded tool execution, approval pauses, and resumption |
| Generative inference | Ollama | Local tool selection, grounded drafting, and structured judging |
| Persistence | SQLite | Clients, contracts, messages, draft versions, decisions, events, and pipeline runs |
| Retrieval | Existing BM25 and multilingual embedding modules | Local evidence retrieval with source metadata |
| Model artifacts | Versioned local files | Fitted preprocessors, models, indexes, thresholds, and metadata |
| Evaluation | Existing notebooks plus scripts and reports | Reproducible benchmarks and objective-level evidence |

Keep all agents in one backend application. Use separate agent roles and explicit workflow states rather than separate services.

Move reusable preparation, training, and inference logic out of notebooks. Application startup must load prepared artifacts rather than rerun notebook training.

### Main workflow

1. Receive a simulated message or renewal trigger.
2. Resolve the client and load relevant context.
3. Classify the message and choose the cross-sell, renewal, or question-answering workflow.
4. Let the LLM request validated tools.
5. Execute tools and return their actual results to the LLM.
6. Retrieve relevant supporting passages.
7. Generate an English draft with evidence references.
8. Run structured quality assessment and deterministic checks.
9. Pause for advisor review.
10. Record approval, edit, rejection, or escalation.
11. Dispatch approved messages to the simulated outbox.
12. Persist events for monitoring and evaluation.

The agent trace should show tool names, arguments, results, timing, and citations. It need not expose internal model reasoning.

## 5. Ollama and hardware plan

Observed hardware:

- NVIDIA RTX 3050 Ti Laptop GPU: 4 GB VRAM.
- Approximately 23 GB system RAM; available memory varies with other applications.
- Sufficient disk space for compact models and artifacts.
- Ollama CLI installed, but its server was stopped during inspection.
- Installed model inventory could not be confirmed: the user model store was absent and the service model directory was unreadable.

Implementation steps:

1. Start Ollama and inspect the available models through its API or CLI.
2. Reuse an appropriate installed model if its performance is acceptable.
3. If necessary, install a quantized `qwen3:4b` as the initial tool-capable candidate.
4. Benchmark tool selection, valid arguments, English drafting, French-source comprehension, grounded answers, and structured judging.
5. Select the model based on measured local quality, latency, and memory use; reduce model size or context if necessary.
6. Keep one generative model loaded at a time and run generation and judging sequentially.
7. Persist the model identifier, quantization, prompts, settings, and benchmark results.

Use bounded context, limited tool rounds, explicit timeouts, validated schemas, and a controlled retry policy. No silent substitution of prerecorded responses for live generation.

If inference fails, preserve the case, display a recoverable error, and allow retry or manual drafting. Recorded backup demonstrations must be labelled as recordings.

References:

- [Qwen3:4b model listing](https://ollama.com/library/qwen3:4b)
- [Ollama tool calling](https://github.com/ollama/ollama/blob/main/docs/capabilities/tool-calling.mdx)
- [Ollama structured outputs](https://ollama.com/blog/structured-outputs)

## 6. Data foundation and document acquisition

### Canonical records

- Keep corporate and individual client identifiers in separate namespaces.
- Retain source identifiers, source type, real/synthetic status, import batch, and transformation version.
- Recover contract dates, guarantee baskets, and available financial fields from raw JSON when processed tables omit them.
- Store workflow events separately from original analytical extracts.
- Use a controllable demonstration clock for historical renewal scenarios.
- Preserve original product labels while providing readable English descriptions in the interface.

### Reproducible preparation

- Extract notebook preparation into reusable functions.
- Fit learned imputers, encoders, and statistics within training folds.
- Audit whether each feature would be available at prediction time.
- Persist preparation outputs, artifacts, checksums, and training metadata.
- Maintain separate real-data and synthetic-benchmark datasets and reports.

### General-condition documents

Search completed during planning:

- Checked Downloads, Documents, Projects, and Desktop for identifiable insurance PDFs and archives.
- Checked the current original Next-BH repository tree for PDFs.
- Searched online for BH Assurance general conditions and attempted direct access to its official website.

**Result:** no authentic general-condition PDFs covering the promised six document families were located. Direct website requests failed. Existing saved public product pages are available, but they are not contractual general conditions.

Implementation approach:

1. Use the saved official product-page snapshots for the initial retrieval workflow.
2. Label their source type explicitly as product information.
3. Continue targeted document discovery, including relevant local archives and official product/download links.
4. Verify insurer, product, document type, and edition before admitting any PDF to the corpus.
5. Preserve source URL, retrieval date, checksum, page numbers, product association, and document version.
6. Extract PDF text, using OCR only when required, and create page-aware chunks.
7. Link catalogue products to verified documents; expose missing documentation.

Until contractual documents are available, limit answers to supported public information and abstain on unsupported exclusions, prices, or contractual guarantees. English answers should cite original passages, even when those passages are French.

## 7. Objective-by-objective implementation

### BO1 — Cross-selling

| Objective | Missing implementation | Evidence and target |
|---|---|---|
| DSO1.1 Next-product propensity | Serve the baseline, evaluate the existing hybrid ranker, exclude held products, and apply documented suitability/diversification rules | Company-grouped CV; HitRate@3 >= 0.70 overall and >= 0.40 long-tail; coverage and company-bootstrap intervals |
| DSO1.2 Client segmentation | Fit 4–8 segments, produce understandable personas, and show client membership and segment characteristics | Silhouette >= 0.25; segment sizes, stability, and business interpretation |
| DSO1.3 Recommendation agent tool | Connect real LLM calls to recommendation and retrieval tools; generate sourced proposals | At least 50 scenarios; >= 90% valid tool calls; verify that explanations use actual tool outputs |

Recommendation scores must be described as relative ranking scores unless calibration against an appropriate outcome is demonstrated. Portfolio reconstruction is not proof of future conversion.

### BO2 — Renewals and collection

| Objective | Missing implementation | Evidence and target |
|---|---|---|
| DSO2.1 Non-payment risk | Implement a cautious baseline using features available before the outcome; expose uncertainty and an operational prioritisation fallback | Recall >= 0.70 and PR-AUC >= 0.40; evaluate the small real dataset separately from synthetic payment-history benchmarks |
| DSO2.2 Guarantee upsell | Connect existing association rules to contracts and display the triggering rule and suggested guarantee | Lift >= 1.5 and coverage >= 60%; assess on held-out contracts or masked guarantees |
| DSO2.3 Renewal agent | Choose reminder, offer, or escalation; generate drafts and scheduled actions with the demonstration clock | >= 95% simulated contact at least 30 days before expiry; contact requires approved simulated dispatch, not merely a generated draft |

Do not use the current unpaid label or post-outcome information as predictive features. Split simulated histories by client and time. The real dataset's 12 unpaid examples limit the strength of conclusions.

### BO3 — Conversation handling

| Objective | Missing implementation | Evidence and target |
|---|---|---|
| DSO3.1 Intent and objections | Train and serve the six-intent classifier; add uncertainty handling and escalation | Macro-F1 >= 0.80 on a 300-message labelled benchmark with a held-out test partition; confusion matrix and per-intent scores |
| DSO3.2 Grounded retrieval and answers | Connect retrieval to English answers, source references, product filtering, and insufficient-evidence handling | Recall@5 >= 0.85 and faithfulness >= 0.90 on labelled questions and reviewed supporting passages |
| DSO3.3 Conversation orchestrator | Persist context, select validated tools, route specialised workflows, and pause/resume around advisor review | Complete scenario tests; >= 70% of drafts approved without edits by human reviewers |

Train/test separation must keep paraphrases or generated variants of the same scenario together. Report synthetic-message performance separately from independently written human examples.

### BO4 — Value, quality, and monitoring

| Objective | Missing implementation | Evidence and target |
|---|---|---|
| DSO4.1 Value and loyalty | Implement a transparent prioritisation score and expose its contributing factors, separately from segmentation | Spearman >= 0.5 against future holdings in a synthetic longitudinal benchmark; real 12-month validation remains unavailable |
| DSO4.2 LLM-as-judge quality | Implement a structured rubric for grounding, relevance, tone, unsupported claims, and advisor escalation | Cohen's kappa >= 0.6 against human reviewers; disclose shared-model bias when generator and judge use the same model |
| DSO4.3 Monitoring agent | Calculate KPIs from stored events and generate an on-demand weekly insight report through tools | Reconcile reported numbers with database queries; distinguish observed demo activity from simulated business outcomes |

The quality judge assists the advisor; it does not replace human approval. Future business outcomes must not be manufactured from the same score being evaluated.

### BO5 — Data governance

| Objective | Missing implementation | Evidence and target |
|---|---|---|
| DSO5.1 Data anomalies | Detect schema errors, missing fields, invalid dates, duplicate records, and suspicious values | Injected-fault recall >= 90%, precision >= 80%; also report checks on original data |
| DSO5.2 Identity deduplication | Normalize records, generate candidates within each client type, score duplicate pairs, and expose uncertain matches for review | Precision >= 95%, recall >= 85% on labelled duplicate benchmarks; entity-separated train/test data |
| DSO5.3 Data-pipeline agent | Inspect quality failures, quarantine invalid batches, retry recoverable failures, and trigger controlled refresh/retraining | >= 95% of defined injected incidents handled without manual rerun; verify idempotence and preservation of the last valid artifacts |

Do not merge corporate entities with individual clients. Keep uncertain duplicate pairs as review candidates and retain source-to-canonical mappings. Retraining creates candidate artifacts; promotion requires validation rather than overwriting a working model immediately.

## 8. Advisor interface

### Design direction

A restrained desktop-oriented English workspace with clear tables, readable status labels, a central case panel, and minimal navigation. Prioritise functionality and rapid case review over decorative presentation.

The distinctive element is an expandable **evidence panel** showing tool calls, returned recommendations, supporting passages, quality findings, and model/version metadata.

### Screens and actions

| Screen | Basic advisor functionality |
|---|---|
| Overview | Pending drafts, upcoming renewals, escalations, recent activity, backend and model status |
| Clients | Search/filter by client type, sector, or segment; inspect profile, holdings, persona, value score, and recommendations |
| Inbox and case workspace | Read a simulated message, enter a client reply, run the assistant, inspect conversation history and agent activity |
| Draft review | Edit, approve, reject with reason, regenerate, or escalate; inspect citations and quality findings |
| Renewals | Sort by expiry, inspect risk and guarantee suggestions, generate reminders or offers, and review scheduled actions |
| Outbox | Inspect approved simulated messages and continue scenarios with simulated client replies |
| Data and evaluation | Inspect quarantined batches, review duplicate candidates, run controlled refreshes, and view objective-level evidence |

### State and persistence

Primary lifecycle:

**Received -> Processing -> Awaiting review -> Approved -> Simulated dispatch**

Additional explicit states: failed, rejected, and escalated.

- Editing creates a new draft version and invalidates previous approval.
- Rejecting or escalating records the reason and timestamp.
- Sending uses the approved version only.
- Repeated actions must not create duplicate dispatches or imports.
- Conversations, decisions, drafts, and activity survive application restarts.
- Loading, empty, unavailable-model, invalid-input, and retry states must be understandable.
- A resettable demonstration dataset must not delete original analytical data.

## 9. Implementation phases and schedule

### Phase A — Foundation and first integrated workflow

**Target: before the Monday, November 23 review.**

1. Create the working copy and isolated Python environment; establish dependency versions.
2. Start Ollama, inventory models, and select a workable local model.
3. Establish SQLite schemas, seed clients/contracts, and define event records.
4. Package baseline recommendation and retrieval as backend tools.
5. Implement actual tool-calling, grounded drafting, and bounded failure handling.
6. Build client lookup, inbox/case review, draft editing/approval, and simulated outbox.
7. Add one historical renewal scenario and basic activity counters.
8. Rehearse a full client-message-to-approved-outbox scenario.

**Exit criterion:** a live LLM calls actual tools; the advisor reviews a sourced draft; approved simulated dispatch and history persist after restart. Demonstrate one renewal path and provide an honest status matrix for all 15 objectives.

### Phase B — Objective implementation and initial evaluation

**Monday after review:** incorporate professor feedback, confirm objective definitions, and freeze benchmark splits and metric definitions.

**Tuesday, November 24:** complete recommendation, segmentation, guarantee, and payment-risk integrations; produce initial evaluations.

**Wednesday, November 25:** complete intent/retrieval benchmarks, conversation routing, draft judging, and human-review collection.

**Thursday, November 26:** complete value scoring, monitoring, anomaly detection, deduplication, and pipeline incident handling.

**Exit criterion:** all 15 objectives have executable functionality, representative outputs, and an evaluation or explicitly documented data limitation.

### Phase C — Stabilisation and validation

**Friday, November 27:**

- Run the complete benchmark and integration suite.
- Fix blocking failures and retest affected paths.
- Verify a clean installation/startup procedure and offline operation.
- Reconcile all displayed metrics with stored evidence.
- Update README, objective mapping, team names, and presentation to all 15 objectives.
- Prepare a short demonstration script and labelled recording backup.
- Freeze new features and rehearse the final presentation.

**Saturday, November 28:** verify the environment, warm the model, reset demonstration scenarios, and run the startup checklist before validation.

### Priority under time pressure

Preserve the integrated advisor workflow, coverage of all 15 objectives, and credible evaluation evidence. Reduce cosmetic polish and extensive hyperparameter searches first. Do not replace missing functionality with unlabelled scripted outputs or claim numerical targets were achieved without measurement.

## 10. Evaluation and verification strategy

- Freeze datasets and splits before final model comparison.
- Keep company/client groups intact across train/test partitions.
- Fit learned preprocessing within training folds and audit feature availability.
- Separate real, synthetic, and human-reviewed evidence in reports and interface labels.
- Store dataset versions, seeds, model parameters, prompts, and artifact identifiers.
- Use paired company-bootstrap intervals for recommendation comparisons.
- Measure local inference latency and tool-call validity before committing to a model.
- Collect human labels for intent examples, retrieval relevance, draft acceptance, and quality judgments.
- State denominators and metric definitions explicitly, including guarantee coverage, timely contact, incident handling, and approval-without-edit rate.
- Retain unsuccessful cases and missed targets in the final report.

Meaningful integration checks should cover invalid tool arguments, unsupported questions, refusal handling, missing clients, Ollama unavailability, approval invalidation after edits, duplicate dispatch prevention, persistence after restart, quarantine, retry, and rejected artifact promotion.

## 11. Final demonstration scenarios

1. **Corporate cross-sell:** select a company, inspect its persona and holdings, request recommendations, generate a sourced proposal, edit/approve it, and inspect the outbox.
2. **Individual renewal:** advance the demonstration clock, identify a due contract, inspect the renewal decision and guarantee suggestion, and approve a reminder or escalation.
3. **Grounded question:** submit an English question, retrieve relevant evidence, and display an English answer with original source references.
4. **Refusal or insufficient evidence:** demonstrate respectful refusal handling or advisor escalation without invented contractual claims.
5. **Data incident:** import a faulty batch, detect anomalies, quarantine it, show the pipeline agent's decision, and recover without corrupting active data.
6. **Duplicate review:** display same-type duplicate candidates with evidence and retain an auditable mapping after confirmation.
7. **Monitoring:** generate an insight report from actual demonstration events and inspect the underlying counts.

## 12. Final acceptance criteria

- One documented startup procedure launches the local application.
- The application operates offline after models, libraries, and source documents are cached.
- The selected Ollama model performs live tool calls and grounded drafting on this laptop.
- All 15 objectives have an implementation, visible output, and evidence/status record.
- Advisor approval is required before simulated dispatch.
- Editing invalidates approval; repeated actions do not duplicate messages or imports.
- Conversations, draft versions, decisions, and events persist across restarts.
- Historical scenarios use an explicit demonstration clock.
- Data provenance and real/synthetic status remain visible.
- Model failures and unsupported questions have recoverable outcomes.
- The interface exposes sources and makes product-information limitations clear.
- Numerical targets are reported as achieved, missed, or unevaluable with reasons.
- Documentation and presentation consistently describe the five-member, 15-objective scope.

## 13. Remaining dependency and limitations

The main unresolved external dependency is access to authentic BH Assurance general-condition documents. Continue discovery during implementation, but do not block the initial workflow on their availability. Public product information can support the initial demonstration while contractual coverage remains explicitly limited.

Real longitudinal validation of customer value, strong payment-risk conclusions from only 12 unpaid examples, and real business conversion/renewal improvements cannot be established by this deadline from the current extracts. Implement the corresponding workflows and clearly separated synthetic benchmarks, and report these limits directly.
