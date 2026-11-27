"""Reproducible objective register. No synthetic result is labelled real."""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, f1_score, precision_score, recall_score, silhouette_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.config import ARTIFACT_VERSION, CUSTOM_DATASET, ROOT
from backend.db import connect, init, now
from backend.governance import match_pair, pipeline, quality_check
from backend.review_metrics import draft_acceptance_metrics, human_review_metrics
from backend.services import INTENT_EXAMPLES, engine, safe_profiles, english_intent, intent_training_data, split_guarantees
from src.features import build_preprocessor
from src.nlp import IntentClassifier, fold_text
from src.recommender import HybridRecommender
from src.rules import baskets_from_contracts, mine_rules, observed_guarantees_by_product, recommend
from src.synthetic import RAG_QUESTIONS
from src.utils import wilson_ci


def objective(id, name, target, metrics, provenance, met=None, limitation="", implementation="implemented"):
    return {"id": id, "name": name, "target": target, "metrics": metrics, "provenance": provenance,
            "status": "awaiting_evidence" if met is None else "target_met" if met else "target_not_met",
            "limitation": limitation, "implementation": implementation}


def recommendation_eval(e, fast=False):
    data = safe_profiles(e.episodes)
    y = np.array([e.products.index(p) for p in data.hidden_product])
    groups = data.REF_pm.to_numpy()
    all_rows = []
    for seed in range(1 if fast else 3):
        # Shuffle group labels reproducibly; each company stays in one fold.
        rng = np.random.default_rng(seed)
        unique = np.unique(groups); shuffled = rng.permutation(unique)
        mapping = dict(zip(unique, shuffled)); split_groups = np.array([mapping[g] for g in groups])
        for fold, (train, test) in enumerate(GroupKFold(5).split(data, groups=split_groups)):
            print(f"Recommendation repeat {seed + 1}, fold {fold + 1}", flush=True)
            tr, te = data.iloc[train], data.iloc[test]
            models = {"logistic": make_pipeline(build_preprocessor(e.products), LogisticRegression(C=.3, max_iter=3000)).fit(tr, y[train])}
            if not fast:
                models['hybrid'] = HybridRecommender(e.products, e.branches, w_lr=.7).fit(tr, y[train], groups[train])
            counts = np.bincount(y[train], minlength=len(e.products))
            popular = set(np.argsort(-counts)[:3])
            scores = {'popularity': np.tile(counts.astype(float), (len(test), 1))}
            for name, m in models.items():
                if name == 'hybrid':
                    scores[name] = m.predict_scores(te)
                else:
                    s = np.zeros((len(test), len(e.products)))
                    s[:, m.classes_] = m.predict_proba(te)
                    scores[name] = s
            for name, s in scores.items():
                for row_index, (idx, (_, episode)) in enumerate(zip(test, te.iterrows())):
                    held = {e.products.index(p) for p in episode.visible_portfolio.split(';') if p in e.products}
                    candidates = [int(j) for j in np.argsort(-s[row_index], kind='stable') if j not in held]
                    if name == 'logistic':
                        held_lines = {e.branches.get(e.products[j]) for j in held}
                        candidates.sort(
                            key=lambda j: (
                                round(float(s[row_index, j]), 4),
                                e.branches.get(e.products[j]) not in held_lines,
                            ),
                            reverse=True,
                        )
                    rank = candidates.index(int(y[idx])) + 1
                    all_rows.append({'repeat': seed, 'fold': fold, 'episode': int(idx), 'company': int(groups[idx]), 'model': name,
                                     'rank': rank, 'long_tail': int(y[idx]) not in popular, 'top3': candidates[:3]})
    frame = pd.DataFrame(all_rows)
    frame.drop(columns=['top3']).to_csv(ROOT / 'reports/validation2_ranking_predictions.csv', index=False)
    summary = {}
    for name, sub in frame.groupby('model'):
        tail = sub[sub.long_tail]
        summary[name] = {'HitRate@3': float((sub['rank'] <= 3).mean()), 'HitRate@1': float((sub['rank'] == 1).mean()),
                         'long_tail_HitRate@3': float((tail['rank'] <= 3).mean()), 'MRR': float((1 / sub['rank']).mean()),
                         'coverage': len({p for v in sub.top3 for p in v})}
    # Company bootstrap: average repeats per episode, then resample complete company clusters.
    pivot = frame.assign(hit=frame['rank'] <= 3).pivot_table(index=['company', 'episode'], columns='model', values='hit', aggfunc='mean')
    rng = np.random.default_rng(0)
    for name in ('logistic', 'hybrid'):
        if name in pivot:
            dif = (pivot[name] - pivot['popularity']).groupby(level='company').agg(['sum', 'count']).to_numpy()
            boots = []
            for _ in range(2000):
                sample = dif[rng.integers(0, len(dif), len(dif))]
                boots.append(sample[:, 0].sum() / sample[:, 1].sum())
            summary[name]['gain_95pct_company_CI'] = np.quantile(boots, [.025, .975]).tolist()
    return summary


def intent_eval():
    rows = []
    for label, examples in INTENT_EXAMPLES.items():
        for i, text in enumerate(examples):
            for prefix in ('', 'Hello, ', 'For our account: ', 'Good morning. ', 'Regarding your offer: '):
                rows.append({'text': prefix + text, 'label': label, 'template': f'{label}-{i}'})
    data = pd.DataFrame(rows)
    predicted = np.empty(len(data), dtype=object)
    for tr, te in GroupKFold(5).split(data, groups=data.template):
        classifier = IntentClassifier().fit(data.iloc[tr].text, data.iloc[tr].label)
        predicted[te] = classifier.predict(data.iloc[te].text)
    learned_prediction = predicted.copy()
    predicted = np.array([english_intent(t) or p for t, p in zip(data.text, predicted)])
    data['prediction'] = predicted
    data['learned_prediction'] = learned_prediction
    data.to_csv(ROOT / 'reports/validation2_intent_predictions.csv', index=False)
    (ROOT / 'data/synthetic').mkdir(exist_ok=True)
    data.drop(columns=['prediction']).to_csv(ROOT / 'data/synthetic/english_intents.csv', index=False)
    return {'messages': len(data), 'templates': data.template.nunique(), 'macro_F1': float(f1_score(data.label, predicted, average='macro')),
            'learned_only_macro_F1': float(f1_score(data.label, learned_prediction, average='macro')),
            'per_class': classification_report(data.label, predicted, output_dict=True, zero_division=0), 'split': '5 folds grouped by original template'}


def heldout_intent_eval(e, path):
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"text", "label"} <= set(rows.columns):
        raise ValueError("Intent benchmark must contain text and label columns")
    texts = rows["text"].str.strip()
    labels = rows["label"].str.strip()
    known = sorted(INTENT_EXAMPLES)
    normalized = texts.map(lambda t: " ".join(fold_text(t).split()))
    training = {" ".join(fold_text(t).split()) for t in intent_training_data()[0]}
    if len(rows) < 300 or set(labels) != set(known):
        raise ValueError("Provide at least 300 labeled messages spanning all supported intents")
    if normalized.eq("").any() or normalized.duplicated().any() or normalized.isin(training).any():
        raise ValueError("Use non-empty, unique messages absent from the training examples")
    predictions = [e.classify(t)["intent"] for t in texts]
    return {"messages": len(rows), "macro_F1": float(f1_score(labels, predictions, labels=known, average="macro")),
            "per_class": classification_report(labels, predictions, labels=known, output_dict=True, zero_division=0),
            "provenance": "operator-supplied labeled dataset; confirm independent authorship"}


def retrieval_eval(e):
    rows = []
    for french, key, english in RAG_QUESTIONS:
        for language, question in [('English', english), ('French', french)]:
            relevant = [str(r.chunk_id) for r in e.corpus.itertuples() if fold_text(key) in fold_text(r.text)]
            result = e.search(question, 5)
            retrieved = [p['chunk_id'] for p in result['passages']]
            hit = bool(set(relevant) & set(retrieved))
            recall = len(set(relevant) & set(retrieved)) / len(relevant) if relevant else None
            rows.append({'question': question, 'language': language, 'key_phrase': key, 'answerable': bool(relevant), 'hit_at_5': hit,
                         'recall_at_5': recall, 'relevant_ids': relevant, 'retrieved_ids': retrieved})
    (ROOT / 'reports/validation2_retrieval_predictions.json').write_text(json.dumps(rows, indent=2))
    report = {}
    for language in ('English', 'French'):
        sub = [r for r in rows if r['language'] == language and r['answerable']]
        report[language] = {'questions': 50, 'answerable': len(sub), 'Recall@5': float(np.mean([r['recall_at_5'] for r in sub])) if sub else 0,
                            'HitRate@5': float(np.mean([r['hit_at_5'] for r in sub])) if sub else 0}
    report['faithfulness'] = 'Requires human review of generated answers; retrieval scores alone do not establish faithfulness'
    return report


def heldout_retrieval_eval(e, path):
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Retrieval benchmark must be a non-empty JSON array")
    chunk_ids = set(e.corpus["chunk_id"].astype(str))
    seen = {" ".join(fold_text(q).split()) for fr, _, en in RAG_QUESTIONS for q in (fr, en)}
    recalls = []
    abstentions = []
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("question"), str)
                or not isinstance(row.get("relevant_ids"), list)
                or any(not isinstance(id_, str) for id_ in row["relevant_ids"])):
            raise ValueError("Each retrieval question needs question and relevant_ids fields")
        question = row["question"].strip()
        normalized = " ".join(fold_text(question).split())
        relevant = set(row["relevant_ids"])
        if not normalized or normalized in seen or not relevant <= chunk_ids:
            raise ValueError("Use unique questions outside the development set and valid passage IDs")
        seen.add(normalized)
        hits = set(str(p["chunk_id"]) for p in e.search(question, 5)["passages"])
        if relevant:
            recalls.append(len(relevant & hits) / len(relevant))
        else:
            abstentions.append(not hits)
    if not recalls:
        raise ValueError("Include at least one answerable question")
    return {"questions": len(rows), "answerable": len(recalls), "Recall@5": float(np.mean(recalls)),
            "unanswerable": len(abstentions),
            "abstention_rate": float(np.mean(abstentions)) if abstentions else None}


def payment_eval(e):
    x = e.pp[['contract_age', 'n_guarantees']].astype(float)
    y = e.pp.unpaid.astype(int).to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', random_state=0, max_iter=1000))
    groups = e.pp.user_id.to_numpy()
    splitter = StratifiedGroupKFold(3, shuffle=True, random_state=0)
    folds = list(splitter.split(x, y, groups))
    p = cross_val_predict(model, x, y, cv=folds, method='predict_proba')[:, 1]
    flagged = p >= .5
    positives = int(y.sum())
    caught = int(np.count_nonzero(flagged & (y == 1)))
    model.fit(x.to_numpy(), y)
    joblib.dump({'model': model, 'features': list(x.columns), 'provenance': 'real historical retrospective sample; feature timing unverified'}, ROOT / 'models/app_payment.joblib')
    rng = np.random.default_rng(23)
    latent = rng.normal(size=(1200, 3)); probability = 1 / (1 + np.exp(-(-2 + latent @ np.array([1., .7, -.8]))))
    sy = rng.binomial(1, probability)
    sp = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced')), latent, sy,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
    return {'real': {'n': len(y), 'positives': positives, 'detected_positives': caught,
                     'policyholders': len(set(groups)), 'threshold': .5, 'flagged': int(flagged.sum()),
                     'Recall': float(recall_score(y, flagged)), 'recall_95pct_wilson': wilson_ci(caught, positives),
                     'precision': float(precision_score(y, flagged, zero_division=0)),
                     'PR_AUC': float(average_precision_score(y, p)), 'prevalence': float(y.mean()),
                     'features': list(x.columns), 'split': 'three stratified folds grouped by policyholder'},
            'synthetic': {'n': len(sy), 'Recall': float(recall_score(sy, sp >= .5)), 'PR_AUC': float(average_precision_score(sy, sp))}}


def source_guarantee_candidates():
    rows = json.loads((ROOT / "data/raw/pp_contract_renewals.json").read_text())
    by_family = {}
    shortlist = defaultdict(set)
    for index, row in enumerate(rows):
        profile = row["profile"]
        held = {fold_text(value) for value in split_guarantees(profile.get("Lib_Garantie"))}
        candidates = {fold_text(value): value for value in split_guarantees(profile.get("Candidate_Garanties"))}
        unheld = set(candidates) - held
        counts = by_family.setdefault(profile["LIB_SOUS_BRANCHE"], {"contracts": 0, "with_unheld_candidates": 0})
        counts["contracts"] += 1
        counts["with_unheld_candidates"] += int(bool(unheld))
        for name in unheld:
            shortlist[(profile["LIB_PRODUIT"], candidates[name])].add(index)
    # Propose review order by incremental coverage. Eligibility still needs confirmation.
    remaining = dict(shortlist)
    covered = set()
    priority = []
    while remaining:
        pair = min(remaining, key=lambda key: (-len(remaining[key] - covered), key))
        contracts = remaining.pop(pair)
        incremental = len(contracts - covered)
        if not incremental:
            break
        covered.update(contracts)
        priority.append({"product": pair[0], "guarantee": pair[1], "additional_contracts": incremental,
                         "cumulative_contracts_if_approved": len(covered)})
    return {
        "contracts": len(rows),
        "with_unheld_candidates": sum(r["with_unheld_candidates"] for r in by_family.values()),
        "by_family": by_family,
        "review_shortlist": [
            {"product": product, "guarantee": guarantee, "contracts": len(contracts)}
            for (product, guarantee), contracts in sorted(shortlist.items(), key=lambda item: (-len(item[1]), item[0]))
        ],
        "review_priority_if_approved": priority,
        "provenance": "source Candidate_Garanties field; eligibility and method undocumented",
    }


def verified_guarantee_coverage(path):
    catalogue = json.loads(path.read_text())
    if not isinstance(catalogue, list) or not catalogue:
        raise ValueError("Verified guarantee catalogue must be a non-empty JSON array")
    approved = {}
    for row in catalogue:
        fields = ("product", "guarantee", "source")
        if not isinstance(row, dict) or any(
            not isinstance(row.get(field), str) or not row[field].strip() for field in fields
        ):
            raise ValueError("Each approved pair needs product, guarantee and source fields")
        product = fold_text(row["product"].strip())
        guarantee = fold_text(row["guarantee"].strip())
        approved.setdefault(product, set()).add(guarantee)
    rows = json.loads((ROOT / "data/raw/pp_contract_renewals.json").read_text())
    covered = 0
    for row in rows:
        profile = row["profile"]
        held = {fold_text(value) for value in split_guarantees(profile.get("Lib_Garantie"))}
        proposed = {fold_text(value) for value in split_guarantees(profile.get("Candidate_Garanties"))}
        allowed = approved.get(fold_text(profile["LIB_PRODUIT"].strip()), set())
        covered += int(bool((proposed & allowed) - held))
    return {"contracts": len(rows), "covered": covered, "coverage": covered / len(rows) if rows else None,
            "approved_pairs": sum(map(len, approved.values())),
            "provenance": "operator-provided approved catalogue intersected with source candidates"}


def guarantee_eval(e):
    baskets = baskets_from_contracts(e.pp)
    groups = e.pp.user_id.to_numpy()
    folds = []
    by_family = {}
    heldout_rule_checks = []
    for train, test in GroupKFold(5).split(baskets, groups=groups):
        training = [baskets[i] for i in train]
        rules = mine_rules(training)
        observed = observed_guarantees_by_product(training)
        test_baskets = [baskets[i] for i in test]
        for rule in rules.itertuples():
            consequent = f"gar::{rule.consequent}"
            antecedent = set(rule.antecedent)
            matching = [basket for basket in test_baskets if antecedent <= basket]
            support = sum(consequent in basket for basket in matching)
            prevalence = sum(consequent in basket for basket in test_baskets) / len(test_baskets)
            if len(matching) >= 3 and prevalence > 0:
                lift = support / len(matching) / prevalence
                heldout_rule_checks.append(lift)
        fold_covered = 0
        for index in test:
            basket = baskets[index]
            product = next(item for item in basket if item.startswith("prod::"))
            family = str(e.pp.iloc[index].sub_branch)
            allowed = observed.get(product, set())
            held = {item[5:] for item in basket if item.startswith("gar::")}
            available = allowed - held
            covered = bool(recommend(basket, rules, allowed=allowed))
            row = by_family.setdefault(family, {"contracts": 0, "covered": 0, "observed_alternative": 0})
            row["contracts"] += 1
            row["covered"] += int(covered)
            row["observed_alternative"] += int(bool(available))
            fold_covered += int(covered)
        folds.append({"test_contracts": len(test), "covered": fold_covered, "rules": len(rules),
                      "minimum_rule_lift": float(rules.lift.min()) if len(rules) else None})
    lifts = [fold["minimum_rule_lift"] for fold in folds if fold["minimum_rule_lift"] is not None]
    return {"test_contracts": len(baskets), "coverage": sum(f["covered"] for f in folds) / len(baskets),
            "observed_alternative_coverage": sum(r["observed_alternative"] for r in by_family.values()) / len(baskets),
            "minimum_rule_lift": min(lifts, default=0),
            "heldout_rule_lift": {"supported_checks": len(heldout_rule_checks),
                                  "checks_at_or_above_target": sum(lift >= 1.5 for lift in heldout_rule_checks),
                                  "minimum": min(heldout_rule_checks, default=None)},
            "folds": folds, "by_family": by_family,
            "split": "five folds grouped by policyholder; same-product observational coverage"}


def governance_eval():
    rows, expected = [], []
    for i in range(200):
        r = {'id': str(i), 'kind': 'corporate' if i % 2 else 'individual', 'label': f'Evaluation {i}', 'email': f'eval{i}@example.test', 'age': 35}
        corrupt = i % 2 == 0
        if corrupt:
            fault = i % 8
            if fault == 0: r['label'] = ''
            elif fault == 2: r['age'] = -10
            elif fault == 4: r['email'] = 'invalid'
            else: r['expiry_date'] = 'broken-date'
        rows.append(r); expected.append(corrupt)
    flagged = {f['row'] for f in quality_check(rows)}
    pred = [i in flagged for i in range(len(rows))]
    pairs, truth, matches = [], [], []
    for i in range(120):
        a = {'id': f'a{i}', 'kind': 'corporate' if i % 2 else 'individual', 'label': f'Client {i} Trading', 'email': f'client{i}@example.test'}
        same = i % 3 == 0
        b = {**a, 'id': f'b{i}', 'label': f'Client {i} Trding'} if same else {**a, 'id': f'b{i}', 'email': f'other{i}@example.test', 'label': f'Different {i} Partners'}
        if i % 5 == 0 and not same: b['kind'] = 'individual' if a['kind'] == 'corporate' else 'corporate'
        result = match_pair(a, b)
        truth.append(same); matches.append(result['decision'] == 'duplicate')
        pairs.append({'left': a, 'right': b, 'duplicate': same, 'prediction': result})
    (ROOT / 'data/synthetic/identity_benchmark.json').write_text(json.dumps(pairs, indent=2))
    incidents = []
    for i in range(20):
        records = [{'id': f'incident-{i}', 'kind': 'corporate', 'label': '' if i % 2 == 0 else f'Valid {i}'}]
        result = pipeline(records, transient_failure=i % 2 == 1)
        expected_status = 'quarantined' if i % 2 == 0 else 'accepted'
        incidents.append({'scenario': i, 'expected': expected_status, 'actual': result['status'], 'handled': result['status'] == expected_status})
    return ({'rows': len(rows), 'injected_fault_rows': sum(expected), 'recall': float(recall_score(expected, pred)), 'precision': float(precision_score(expected, pred))},
            {'pairs': len(pairs), 'precision': float(precision_score(truth, matches)), 'recall': float(recall_score(truth, matches)), 'split': 'held-out entity identifiers; fixed heuristic, no fitting'},
            {'scenarios': incidents, 'handled_rate': float(np.mean([r['handled'] for r in incidents]))})


def longitudinal_eval(path):
    rows = pd.read_csv(path, dtype={"client_id": str})
    required = {"client_id", "as_of", "holdings"}
    if not required <= set(rows.columns):
        raise ValueError("Holdings snapshots need client_id, as_of and holdings columns")
    rows = rows[list(sorted(required))].copy()
    rows["as_of"] = pd.to_datetime(rows["as_of"], format="%Y-%m-%d", errors="raise")
    counts = pd.to_numeric(rows["holdings"], errors="raise")
    if (rows["client_id"].fillna("").str.strip().eq("").any()
            or rows["as_of"].isna().any() or not np.isfinite(counts).all()
            or (counts < 0).any() or (counts % 1 != 0).any()
            or rows.duplicated(["client_id", "as_of"]).any()):
        raise ValueError("Provide unique dated client snapshots with nonnegative integer holdings")
    rows["holdings"] = counts
    rows["outcome_date"] = rows["as_of"] + pd.DateOffset(years=1)
    pairs = rows.merge(
        rows[["client_id", "as_of", "holdings"]].rename(
            columns={"as_of": "outcome_date", "holdings": "future_holdings"}
        ),
        on=["client_id", "outcome_date"],
        validate="many_to_one",
    )
    # Use the latest complete calendar cohort so a client contributes only once.
    cohort = pairs[pairs["as_of"] == pairs["as_of"].max()]
    baseline = None
    if cohort["holdings"].nunique() > 1 and cohort["future_holdings"].nunique() > 1:
        baseline = float(spearmanr(cohort["holdings"], cohort["future_holdings"]).statistic)
    return {"paired_clients": len(cohort),
            "cohort_date": cohort["as_of"].iloc[0].date().isoformat() if len(cohort) else None,
            "persistence_baseline_spearman": baseline,
            "provenance": "operator-supplied dated snapshots; confirm source independently"}


def run(fast=False, intent_labels=None, holdings_snapshots=None, retrieval_labels=None, guarantee_catalogue=None):
    if CUSTOM_DATASET:
        raise SystemExit("The historical evaluator requires the bundled dataset. Use scripts/external_benchmarks.py for public benchmarks.")
    init(); e = engine(); objectives = []
    heldout_intents = heldout_intent_eval(e, intent_labels) if intent_labels is not None else None
    real_value = longitudinal_eval(holdings_snapshots) if holdings_snapshots is not None else None
    live_path = ROOT / "reports/live_demo_evidence.json"
    live = json.loads(live_path.read_text()) if live_path.exists() else {}
    historical_live = {"finished_at": live.get("finished_at"), "summary": live.get("summary"),
                       "provenance": "saved live validation; rerun to verify current code"}
    report = {'generated_at': now(), 'version': ARTIFACT_VERSION, 'live_llm': historical_live,
              'datasets': {'corporate': 'real historical pseudonymised portfolio', 'renewal': 'real historical pseudonymised contracts',
                           'intent': 'synthetic English templates', 'identity': 'synthetic within-type duplicates', 'longitudinal_value': 'synthetic independent outcome simulation'}, 'objectives': objectives}
    def save():
        (ROOT / 'reports/validation2_evaluation.json').write_text(json.dumps(report, indent=2, default=str, allow_nan=False))
    ranks = recommendation_eval(e, fast)
    lr = ranks['logistic']  # The application currently serves logistic scores with new lines breaking score ties.
    objectives.append(objective('1.1', 'Next-product propensity', 'HitRate@3 >= 0.70 overall / 0.40 long-tail', ranks, 'real reconstruction benchmark', lr['HitRate@3'] >= .7 and lr['long_tail_HitRate@3'] >= .4,
                                 'Raw profile labels, constant UNKNOWNs, history disabled. Reconstructs held products; not future sales. Logistic scores include the UI new-line tiebreaker; hybrid results are experimental.'))
    sil = float(silhouette_score(e.segment_features, e.segmenter.labels_))
    objectives.append(objective('1.2', 'Client segmentation', '4–8 segments; silhouette >= 0.25', {'segments': 7, 'silhouette': sil, 'personas': e.personas.reset_index().to_dict('records'), 'selection': 'Compared k=4..8 on descriptive training data; chose 7 as the smallest k exceeding 0.25; not held-out stability evidence'}, 'real portfolio', sil >= .25))
    tool_path = ROOT / "reports/validation2_live_tool_scenarios.json"
    tool_run = json.loads(tool_path.read_text()) if tool_path.exists() else {}
    hashes = tool_run.get("source_sha256", {})
    expected_sources = {"backend/agent.py", "backend/services.py", "scripts/live_validation.py"}
    current_tool_run = set(hashes) == expected_sources and all(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
        for name, digest in hashes.items()
    )
    tool_metrics = {"protocol": "Live local tool trace plus 50-scenario tool suite",
                    "saved_live_run": historical_live}
    if current_tool_run:
        tool_metrics["current_tool_run"] = {k: v for k, v in tool_run.items() if k != "results"}
    objectives.append(objective(
        '1.3', 'Recommendation agent', '>= 90% valid tool calls on >= 50 scenarios',
        tool_metrics, 'live Ollama tool scenarios against current code in an isolated runtime'
        if current_tool_run else 'live Ollama evaluation pending',
        tool_run["scenarios"] >= 50 and tool_run["valid_scenario_rate"] >= .9 if current_tool_run else None,
        limitation='Do not infer tool validity from schema definitions.',
    ))
    save()
    print('Evaluating payment and guarantee tools', flush=True)
    risk = payment_eval(e)
    objectives.append(objective('2.1', 'Non-payment risk', 'Recall >= 0.70; PR-AUC >= 0.40', risk, 'real retrospective + separate synthetic benchmark', risk['real']['Recall'] >= .7 and risk['real']['PR_AUC'] >= .4,
                                 'Only 12 unpaid examples. Timing of real features unverified. Synthetic history is functional evidence only.'))
    guarantees = guarantee_eval(e)
    guarantees["source_candidates_for_review"] = source_guarantee_candidates()
    if guarantee_catalogue is not None:
        guarantees["verified_catalogue_coverage"] = verified_guarantee_coverage(guarantee_catalogue)
    objectives.append(objective('2.2', 'Guarantee upselling', 'Lift >= 1.5; coverage >= 60%', guarantees, 'real held-out renewal contracts', guarantees['coverage'] >= .6 and (guarantees['heldout_rule_lift']['minimum'] or 0) >= 1.5,
                                 'Coverage is measured out of fold. Training lift is the rule-mining threshold. Held-out lift is reported for fold-rule checks with at least three antecedent matches and a nonzero consequent prevalence. Same-product observation is historical evidence and eligibility requires advisor verification.'))
    renewal = e.renewal_dispatch_metrics()
    coverage = renewal["approved_dispatch_coverage"]
    objectives.append(objective('2.3', 'Renewal agent', '>= 95% dispatched >= 30 days before expiry',
                                 renewal, 'historical simulation; local outbox events',
                                 coverage >= .95 if coverage is not None else None))
    intents = intent_eval()
    if heldout_intents is not None:
        intents["heldout"] = heldout_intents
    objectives.append(objective('3.1', 'Intent classifier', 'Macro-F1 >= 0.80 on 300 labelled messages',
                                 intents, 'synthetic template benchmark and optional operator-supplied labels',
                                 heldout_intents["macro_F1"] >= .8 if heldout_intents is not None else None,
                                 'Final validation requires independently authored English messages with human labels.'))
    reviews = human_review_metrics()
    retrieval = retrieval_eval(e)
    retrieval["local_human_reviews"] = reviews
    recall = None
    if retrieval_labels is not None:
        retrieval["heldout"] = heldout_retrieval_eval(e, retrieval_labels)
        recall = retrieval["heldout"]["Recall@5"]
    faithfulness = reviews["faithfulness"]
    retrieval_met = (recall >= .85 and faithfulness >= .90
                     if recall is not None and faithfulness is not None else None)
    objectives.append(objective('3.2', 'Grounded retrieval', 'Recall@5 >= 0.85; faithfulness >= 0.90',
                                 retrieval, 'Public passages, optional held-out labels, and recorded human reviews',
                                 retrieval_met,
                                 'Both independently labeled retrieval and human reviews of generated drafts are required.'))
    acceptance = draft_acceptance_metrics()
    approval_rate = acceptance["approval_without_edit_rate"]
    objectives.append(objective('3.3', 'Conversation orchestrator', '>= 70% drafts approved without edit',
                                 acceptance, 'local advisor workflow records',
                                 approval_rate >= .7 if approval_rate is not None else None,
                                 'Use a dedicated runtime with genuine advisor decisions to establish acceptance.'))
    rng = np.random.default_rng(42)
    n = rng.integers(1, 5, 500); tenure = rng.uniform(0, 10, 500)
    scores = np.minimum(n / 4, 1) * 40 + np.minimum(tenure / 10, 1) * 30 + np.maximum(0, 4 - n) / 4 * 30
    latent_loyalty = rng.beta(2, 3, 500)
    future = np.clip(n + rng.poisson(.12 * tenure + latent_loyalty) - rng.binomial(n, .15), 0, 10)
    rho = float(spearmanr(scores, future).statistic)
    pd.DataFrame({'n_products': n, 'tenure': tenure, 'priority_score': scores, 'future_holdings': future, 'provenance': 'synthetic'}).to_csv(ROOT / 'data/synthetic/value_longitudinal.csv', index=False)
    objectives.append(objective('4.1', 'Value and loyalty', 'Spearman >= 0.5 against 12-month holdings', {'synthetic_spearman': rho, 'n': 500, 'real_persistence_baseline': real_value}, 'synthetic simulation and optional dated holdings baseline',
                                 limitation='The synthetic result is exploratory. The dated baseline needs real snapshots and does not establish a validated prediction model.'))
    quality_kappa = reviews["quality_kappa"]
    objectives.append(objective('4.2', 'LLM quality judge', "Cohen's kappa >= 0.6 against human reviewers",
                                 {'rubric': ['grounded', 'relevant', 'professional', 'unsupported_claims'],
                                  'human_kappa': quality_kappa, 'local_human_reviews': reviews},
                                 'Model groundedness compared with recorded human groundedness labels',
                                 quality_kappa >= .6 if quality_kappa is not None else None,
                                 'Groundedness agreement uses drafts with consensus human labels and both label classes.'))
    objectives.append(objective('4.3', 'Monitoring agent', 'Report accurate event KPIs and weekly insights', {'metrics': e.metrics(), 'report_endpoint': '/api/monitoring/report'}, 'observed local application activity', limitation='Live report evidence stored separately. No real conversion or sales claim.'))
    anomaly, identity, incidents = governance_eval()
    objectives.append(objective('5.1', 'Data anomalies', 'Recall >= 90%; precision >= 80%', anomaly, 'synthetic fault injection', anomaly['recall'] >= .9 and anomaly['precision'] >= .8,
                                 'Covers declared schema/date/range/email faults, not arbitrary unknown anomalies.'))
    objectives.append(objective('5.2', 'Identity deduplication', 'Precision >= 95%; recall >= 85%', identity, 'synthetic same-type duplicate pairs', identity['precision'] >= .95 and identity['recall'] >= .85,
                                 'No real identity ground truth. Shared contacts can be ambiguous; advisor review remains mandatory.'))
    objectives.append(objective('5.3', 'Pipeline agent', '>= 95% incidents handled without manual rerun', incidents, 'synthetic controlled incident suite', incidents['handled_rate'] >= .95,
                                 'Handled means safe quarantine or successful bounded recovery; quarantined data is not silently repaired or promoted.'))
    save()
    print('Saved reports/validation2_evaluation.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--fast', action='store_true')
    parser.add_argument('--intent-labels', type=Path, help='Independent English benchmark CSV with text and label columns')
    parser.add_argument("--holdings-snapshots", type=Path, help="Dated CSV with client_id, as_of and holdings columns")
    parser.add_argument("--retrieval-labels", type=Path, help="Independent benchmark JSON with question and relevant_ids")
    parser.add_argument("--guarantee-catalogue", type=Path, help="Approved product, guarantee and source entries in JSON")
    args = parser.parse_args()
    run(args.fast, args.intent_labels, args.holdings_snapshots, args.retrieval_labels, args.guarantee_catalogue)
