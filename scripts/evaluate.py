"""Reproducible objective register. No synthetic result is labelled real."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, f1_score, precision_score, recall_score, silhouette_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.config import ARTIFACT_VERSION, ROOT
from backend.db import connect, init, now
from backend.governance import match_pair, pipeline, quality_check
from backend.services import INTENT_EXAMPLES, engine, safe_profiles, english_intent
from src.features import build_preprocessor
from src.nlp import IntentClassifier, fold_text
from src.recommender import HybridRecommender
from src.rules import baskets_from_contracts, mine_rules, recommend
from src.synthetic import RAG_QUESTIONS


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


def payment_eval(e):
    x = e.pp[['contract_age', 'n_guarantees']].astype(float)
    y = e.pp.unpaid.astype(int).to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', random_state=0, max_iter=1000))
    p = cross_val_predict(model, x, y, cv=StratifiedKFold(3, shuffle=True, random_state=0), method='predict_proba')[:, 1]
    model.fit(x.to_numpy(), y)
    joblib.dump({'model': model, 'features': list(x.columns), 'provenance': 'real historical retrospective sample; feature timing unverified'}, ROOT / 'models/app_payment.joblib')
    rng = np.random.default_rng(23)
    latent = rng.normal(size=(1200, 3)); probability = 1 / (1 + np.exp(-(-2 + latent @ np.array([1., .7, -.8]))))
    sy = rng.binomial(1, probability)
    sp = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced')), latent, sy,
                           cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
    return {'real': {'n': len(y), 'positives': int(y.sum()), 'Recall': float(recall_score(y, p >= .5)), 'PR_AUC': float(average_precision_score(y, p)), 'features': list(x.columns)},
            'synthetic': {'n': len(sy), 'Recall': float(recall_score(sy, sp >= .5)), 'PR_AUC': float(average_precision_score(sy, sp))}}


def guarantee_eval(e):
    baskets = baskets_from_contracts(e.pp)
    train, test = baskets[:220], baskets[220:]
    rules = mine_rules(train)
    suggestions = [recommend(b, rules) for b in test]
    return {'train_contracts': len(train), 'test_contracts': len(test), 'rules': len(rules),
            'coverage': float(np.mean([bool(s) for s in suggestions])), 'minimum_rule_lift': float(rules.lift.min()) if len(rules) else 0,
            'split': 'fixed held-out contract partition; no guarantee recommendation selected using test baskets'}


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


def run(fast=False):
    init(); e = engine(); objectives = []
    report = {'generated_at': now(), 'version': ARTIFACT_VERSION, 'live_llm': 'See live_demo_evidence.json; human review remains separate',
              'datasets': {'corporate': 'real historical pseudonymised portfolio', 'renewal': 'real historical pseudonymised contracts',
                           'intent': 'synthetic English templates', 'identity': 'synthetic within-type duplicates', 'longitudinal_value': 'synthetic independent outcome simulation'}, 'objectives': objectives}
    def save():
        (ROOT / 'reports/validation2_evaluation.json').write_text(json.dumps(report, indent=2, default=str, allow_nan=False))
    ranks = recommendation_eval(e, fast)
    lr = ranks['logistic']
    objectives.append(objective('1.1', 'Next-product propensity', 'HitRate@3 >= 0.70 overall / 0.40 long-tail', ranks, 'real reconstruction benchmark', lr['HitRate@3'] >= .7 and lr['long_tail_HitRate@3'] >= .4,
                                 'Raw profile labels, constant UNKNOWNs, history disabled. Reconstructs held products; not future sales. UI applies additional diversification rules.'))
    sil = float(silhouette_score(e.segment_features, e.segmenter.labels_))
    objectives.append(objective('1.2', 'Client segmentation', '4–8 segments; silhouette >= 0.25', {'segments': 7, 'silhouette': sil, 'personas': e.personas.reset_index().to_dict('records'), 'selection': 'Compared k=4..8 on descriptive training data; chose 7 as the smallest k exceeding 0.25; not held-out stability evidence'}, 'real portfolio', sil >= .25))
    objectives.append(objective('1.3', 'Recommendation agent', '>= 90% valid tool calls on >= 50 scenarios', {'protocol': 'Live local tool trace plus 50-scenario tool suite; see live evidence'}, 'live Ollama evaluation pending', limitation='Do not infer tool validity from schema definitions.'))
    save()
    print('Evaluating payment and guarantee tools', flush=True)
    risk = payment_eval(e)
    objectives.append(objective('2.1', 'Non-payment risk', 'Recall >= 0.70; PR-AUC >= 0.40', risk, 'real retrospective + separate synthetic benchmark', risk['real']['Recall'] >= .7 and risk['real']['PR_AUC'] >= .4,
                                 'Only 12 unpaid examples. Timing of real features unverified. Synthetic history is functional evidence only.'))
    guarantees = guarantee_eval(e)
    objectives.append(objective('2.2', 'Guarantee upselling', 'Lift >= 1.5; coverage >= 60%', guarantees, 'real held-out renewal contracts', guarantees['minimum_rule_lift'] >= 1.5 and guarantees['coverage'] >= .6,
                                 'Coverage means held-out contracts receiving at least one new-guarantee rule suggestion; it does not measure purchase.'))
    contracts = e.contracts()
    day = min(c['data']['expiry_date'] for c in contracts)
    objectives.append(objective('2.3', 'Renewal agent', '>= 95% dispatched >= 30 days before expiry', {'contracts': len(contracts), 'first_expiry': day, 'demo_clock': 'configurable; initial 2025-08-01',
                                 'scheduled_contacts': len(contracts), 'approved_dispatch_coverage': 0}, 'historical simulation', False,
                                 'Scheduling is implemented. Timely contact only counts approved dispatch; no contact success claimed from a schedule.'))
    intents = intent_eval()
    objectives.append(objective('3.1', 'Intent classifier', 'Macro-F1 >= 0.80 on 300 labelled messages', intents, '300 synthetic English messages; grouped template CV', intents['macro_F1'] >= .8,
                                 'Human-written independent evaluation remains required; template labels are synthetic.'))
    retrieval = retrieval_eval(e)
    objectives.append(objective('3.2', 'Grounded retrieval', 'Recall@5 >= 0.85; faithfulness >= 0.90', retrieval, 'real public pages + curated question/phrase benchmark', limitation='No general-condition PDFs. English questions are glosses of the inherited French benchmark. Generated-answer faithfulness needs human review.'))
    objectives.append(objective('3.3', 'Conversation orchestrator', '>= 70% drafts approved without edit', {'workflow': ['context', 'tools_and_draft', 'quality', 'advisor_review', 'simulated_dispatch']}, 'live application; human review pending', limitation='Advisor acceptance must be measured from genuine review decisions.'))
    rng = np.random.default_rng(42)
    n = rng.integers(1, 5, 500); tenure = rng.uniform(0, 10, 500)
    scores = np.minimum(n / 4, 1) * 40 + np.minimum(tenure / 10, 1) * 30 + np.maximum(0, 4 - n) / 4 * 30
    latent_loyalty = rng.beta(2, 3, 500)
    future = np.clip(n + rng.poisson(.12 * tenure + latent_loyalty) - rng.binomial(n, .15), 0, 10)
    rho = float(spearmanr(scores, future).statistic)
    pd.DataFrame({'n_products': n, 'tenure': tenure, 'priority_score': scores, 'future_holdings': future, 'provenance': 'synthetic'}).to_csv(ROOT / 'data/synthetic/value_longitudinal.csv', index=False)
    objectives.append(objective('4.1', 'Value and loyalty', 'Spearman >= 0.5 against 12-month holdings', {'synthetic_spearman': rho, 'n': 500, 'real_longitudinal_result': None}, 'synthetic longitudinal simulation', rho >= .5,
                                 'Future outcomes simulated with independent randomness and latent loyalty. Does not establish real future value.'))
    objectives.append(objective('4.2', 'LLM quality judge', "Cohen's kappa >= 0.6 against human reviewers", {'rubric': ['grounded', 'relevant', 'professional', 'unsupported_claims'], 'human_kappa': None}, 'live model; human comparison pending', limitation='Generator and judge share a model. Human labels cannot be replaced by another model and called human agreement.'))
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
    run(parser.parse_args().fast)
