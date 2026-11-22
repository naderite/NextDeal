"""Actual Ollama calls; isolated DB; resumable evidence. Not human acceptance evidence."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('NEXTDEAL_RUNTIME', str(ROOT / 'runtime/live_validation'))

from backend.agent import TOOLS, RecoArgs, SearchArgs, chat, run_case
from backend.config import OLLAMA_MODEL
from backend.db import connect, init, now
from backend.services import engine


def run(scenarios, full_cases, rerun_full=False):
    init(); e = engine()
    path = ROOT / 'reports/live_demo_evidence.json'
    evidence = json.loads(path.read_text()) if path.exists() else {'generated_at': now(), 'model': OLLAMA_MODEL, 'tool_scenarios': [], 'full_cases': []}
    if evidence.get('model') != OLLAMA_MODEL:
        evidence = {'generated_at': now(), 'model': OLLAMA_MODEL, 'tool_scenarios': [], 'full_cases': []}
    def save():
        path.write_text(json.dumps(evidence, indent=2, default=str))
    for i in range(len(evidence['tool_scenarios']), scenarios):
        corporate = i % 2 == 0
        rows = e.companies
        cid = f"pm:{int(rows.iloc[(i * 7) % len(rows)].REF_pm)}" if corporate else e.contracts()[i % len(e.contracts())]['client_id']
        instruction = ["Recommend additional business insurance products", "Find next insurance products not already held", "Suggest cross-sell candidates", "Rank additional protection", "What else could this company consider?"][i % 5] if corporate else ["Review my renewal and payment priority", "Decide the next renewal action", "Check upcoming policy expiry", "Help with an individual renewal reminder", "Inspect this policyholder's renewal options"][i % 5]
        start = time.monotonic(); expected = 'recommend_products' if corporate else 'renewal_action'
        record = {'scenario': i, 'client_id': cid, 'request': instruction, 'expected_tool': expected}
        try:
            response = chat([
                {'role': 'system', 'content': 'Call exactly one relevant tool for this client. Corporate cross-sell uses recommend_products. Individual renewal uses renewal_action. Do not answer with prose. The current client is bound by the application.'},
                {'role': 'user', 'content': json.dumps({'client_id': cid, 'kind': 'corporate' if corporate else 'individual', 'request': instruction})},
            ], tools=[t for t in TOOLS if t['function']['name'] in ('recommend_products', 'renewal_action')])
            calls = response.get('tool_calls') or []
            outputs = []
            for call in calls:
                f = call['function']; args = f.get('arguments', {})
                if isinstance(args, str): args = json.loads(args)
                if f['name'] == 'recommend_products':
                    parsed = RecoArgs.model_validate(args); output = e.recommendations(cid, parsed.k)
                elif f['name'] == 'renewal_action':
                    if args: raise ValueError('Renewal tool accepts no arguments')
                    output = e.renewal(cid)
                else: raise ValueError('Unexpected tool')
                outputs.append({'name': f['name'], 'arguments': args, 'executed': True, 'output_keys': list(output)})
            record.update({'calls': outputs, 'valid': bool(outputs), 'correct_route': bool(outputs) and all(t['name'] == expected for t in outputs)})
        except Exception as exc:
            record.update({'valid': False, 'correct_route': False, 'error': str(exc)})
        record['elapsed_seconds'] = round(time.monotonic() - start, 2)
        evidence['tool_scenarios'].append(record); save()
        print(f"Tool scenario {i + 1}/{scenarios}: valid={record['valid']} route={record['correct_route']} ({record['elapsed_seconds']}s)", flush=True)
    messages = [
        (f"pm:{int(e.companies.iloc[0].REF_pm)}", 'Corporate cross-sell', 'What additional business protection could suit our company? Please use recommendations and supporting product information.'),
        (e.contracts()[0]['client_id'], 'Individual renewal', 'Please review my upcoming renewal, payment status and relevant additional guarantees.'),
        (f"pm:{int(e.companies.iloc[0].REF_pm)}", 'Refusal handling', 'No thank you. Please stop contacting me. I decline this offer.'),
    ]
    for cid, subject, body in messages[:full_cases]:
        if not rerun_full and any(c['subject'] == subject and c.get('status') == 'awaiting_review' for c in evidence['full_cases']): continue
        with connect() as db:
            cursor = db.execute('INSERT INTO cases(client_id,subject,created_at) VALUES(?,?,?)', (cid, subject, now()))
            case_id = cursor.lastrowid
            db.execute("INSERT INTO messages(case_id,role,body,created_at) VALUES(?,'client',?,?)", (case_id, body, now()))
            db.execute("UPDATE cases SET status='processing' WHERE id=?", (case_id,))
        print('Full live workflow:', subject, flush=True); run_case(case_id)
        with connect() as db:
            case = dict(db.execute('SELECT * FROM cases WHERE id=?', (case_id,)).fetchone())
            draft = db.execute('SELECT * FROM drafts WHERE case_id=? ORDER BY version DESC LIMIT 1', (case_id,)).fetchone()
        record = {**case, 'request': body}
        if draft:
            record.update({'draft': draft['body'], 'evidence': json.loads(draft['evidence']), 'quality': json.loads(draft['quality'])})
        previous = [c for c in evidence['full_cases'] if c['subject'] == subject]
        evidence.setdefault('previous_attempts', []).extend(previous)
        evidence['full_cases'] = [c for c in evidence['full_cases'] if c['subject'] != subject] + [record]
        save()
    metrics = e.metrics()
    try:
        answer = chat([{'role': 'system', 'content': 'Summarize these local demonstration counts in English. Use only the supplied numbers. Do not claim real sales or business uplift.'}, {'role': 'user', 'content': json.dumps(metrics)}])
        evidence['monitoring'] = {'metrics': metrics, 'report': answer['content'], 'live': True}
    except Exception as exc:
        evidence['monitoring'] = {'error': str(exc)}
    results = evidence['tool_scenarios']
    evidence['summary'] = {'scenarios': len(results), 'valid_scenario_rate': sum(r['valid'] for r in results) / max(1, len(results)),
                           'correct_route_rate': sum(r['correct_route'] for r in results) / max(1, len(results)),
                           'full_workflows_passed': sum(r['status'] == 'awaiting_review' for r in evidence['full_cases']),
                           'human_acceptance_evidence': False}
    evidence['finished_at'] = now(); save()
    register = ROOT / 'reports/validation2_evaluation.json'
    if register.exists():
        report = json.loads(register.read_text()); report['live_llm'] = evidence['summary']
        for o in report['objectives']:
            if o['id'] == '1.3':
                o.update({'metrics': evidence['summary'], 'provenance': 'actual local Ollama calls; isolated scenario database',
                          'status': 'target_met' if len(results) >= 50 and evidence['summary']['valid_scenario_rate'] >= .9 else 'target_not_met'})
            if o['id'] == '3.3':
                o['metrics']['live_workflows'] = [{'subject': r['subject'], 'status': r['status']} for r in evidence['full_cases']]
            if o['id'] == '4.3':
                o.update({'metrics': evidence.get('monitoring', {}), 'provenance': 'actual Ollama report over observed test activity', 'status': 'awaiting_evidence'})
        register.write_text(json.dumps(report, indent=2))
    print(json.dumps(evidence['summary'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--scenarios', type=int, default=50); parser.add_argument('--full-cases', type=int, default=3); parser.add_argument('--rerun-full', action='store_true')
    args = parser.parse_args(); run(args.scenarios, args.full_cases, args.rerun_full)
