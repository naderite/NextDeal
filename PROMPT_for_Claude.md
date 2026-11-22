# Reusable prompt for the next milestone

Attach this ZIP (and the professor's PDF) to a new Claude conversation and paste the prompt below.
Replace the bracketed parts.

---

You are helping our team (4 Data Science master students, FST – Université de Tunis El Manar) with the
"Agentic AI Project" supervised by Dr Olfa Layouni. Our project is **NextDeal**, an agentic AI sales
assistant for **BH Assurance** (Tunisian insurer). The attached ZIP is our validation 1 package:
4 executed notebooks (business understanding, data understanding, data preparation, modeling baseline),
raw and processed data, shared baseline helpers plus newer recommendation/segmentation/rules/NLP modules, 16 figures, and an HTML presentation.

Course rules to respect:
- CRISP-DM phases: business understanding → data acquisition & understanding → modeling → deployment.
- Each member owns ≥ 3 distinct data-science objectives (DSOs); ≥ 1 of them must include an agentic
  component (tool-calling, decision-making or orchestration). Our 12 DSOs are in notebook 01 – keep them.
- Everything (notebooks, slides, videos) must be in English. Work is shared on GitHub.

Key facts already established (do not contradict them without new evidence):
- Dataset A: 299 companies, 69 % mono-product, 1.47 products per company on average.
- The prototype score is leaky (`Bought` ≡ candidate == product held); we use leave-one-out episodes instead.
- DSO1.1 baseline: logistic regression HitRate@3 0.686 vs popularity 0.654; long-tail HitRate@3 0.33 vs 0.11
  (5-fold GroupKFold by company × 3 repeats). Targets: 0.70 overall, 0.40 long-tail.
- Dataset B: 320 individual contracts expiring in Sept. 2025, only 12 unpaid → DSO2.1 needs more history.

Task for [validation 2 / date]: [describe what the professor asks, e.g. "finish modeling for all DSOs,
build the agent with tool-calling, a dashboard and a demo"].

Please:
1. Keep the existing folder structure and coding style (shared helpers in `src/`, figures saved to `figures/`,
   reports to `reports/`), and execute every notebook so outputs are visible.
2. For each DSO, implement the approach listed in notebook 04 §6, evaluate it with client-grouped CV and the
   metric written in its success criterion, and report honestly (mean ± std, confidence intervals).
3. Turn the models into agent tools (JSON schema + FastAPI endpoint) and show one end-to-end agent run with
   human validation.
4. Update the HTML presentation with the same design system (navy #0F2A47, blue #2A78D6, orange #EB6834,
   Inter Display / Inter / Lora) and real figures; keep HTML as the current presentation. Update the PDF only if requested.
5. Deliver everything as one ZIP and list what changed since validation 1.

Current clarification: the old architecture and API contracts are a baseline, not constraints. New modules are implemented but not evaluated in notebooks 01–04. The ranking benchmark reconstructs hidden owned products, not future purchases; scores are not calibrated purchase probabilities. Company-bootstrap intervals are +0.1 to +6.5 points overall and +15.8 to +29.6 long-tail, conditional on current preprocessing and model selection. The PDF is a previous export.
