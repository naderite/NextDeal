# NextDeal — Demo guide

## Preflight

Start with `./scripts/start.sh`; confirm the model is connected. Keep the historical clock at 2025-08-01. Run one warm-up case before presenting. Keep dependencies, models and evidence cached for offline use.

## Corporate cross-sell

Search Clients for `pm:6958`. Inspect the automobile portfolio, persona and ranked suggestions. Explain relative scores and the preference for a new insurance line. Start a conversation, run the assistant, inspect tool calls and sources, edit if needed, approve, simulate dispatch and open Outbox. Confirm that approval and message history persist.

## Renewal

Open Renewals and inspect an individual contract. Generate a renewal case and inspect the action, guarantee suggestions and payment status. Client `pp:28563` has a rule-based guarantee suggestion available for a concrete upsell demonstration. Advance the demo clock to demonstrate reminders and overdue escalation. A schedule is not contact evidence; only approved simulated dispatch counts.

## Grounding and refusal

Ask about a product described in the saved pages and inspect the original passages. Ask an unsupported exact-price or exclusion question and demonstrate advisor deferral. Add a reply saying “No thank you. Please stop contacting me.” and verify the assistant does not continue selling.

## Governance

Submit a batch with an empty label or negative age and inspect quarantine. Correct it and import twice to demonstrate idempotence. Run the transient recovery scenario. Compare same-type duplicate records and confirm or reject the mapping. A person/company pair must remain distinct.

Trigger controlled model refresh and inspect the promotion gate. Explain that the previous model remains active if validation fails.

## Monitoring and objective register

Generate an insight report and compare numbers with stored counts. Open all 15 objective results. Explain missed targets and real/synthetic provenance. Record a genuine human draft-quality assessment in the evidence panel.

## Human evidence still needed

Collect independently written English messages, human-reviewed question relevance and factual support, positive/negative draft-quality labels, and real advisor decisions on model drafts. Scripted test approvals are not genuine advisor-acceptance evidence.

## Recovery

If Ollama fails, preserve and retry the case or write a labelled manual draft through the same approval process. Use a saved recording only as a clearly labelled backup.
