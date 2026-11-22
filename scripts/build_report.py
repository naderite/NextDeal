"""Build an offline review handout from measured evidence; no invented success claims."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / 'reports/validation2_evaluation.json').read_text())
esc = lambda value: html.escape(str(value))
groups = [('1', 'Cross-selling'), ('2', 'Renewals and collection'), ('3', 'Conversation handling'), ('4', 'Value, quality and monitoring'), ('5', 'Data governance')]
pages = ['''<section><div class="eyebrow">NEXTDEAL · BH ASSURANCE</div><h1>A local assistant.<br>A complete advisor workflow.</h1><p>Five business objectives · Fifteen data-science objectives</p><p>English workspace · Ollama tool calls · Evidence retrieval · Advisor approval · Simulated outbox</p><div class="note">Implementation and measured performance are separate. This review preserves missed targets, synthetic provenance and pending human evidence.</div></section>''']
for group, title in groups:
    cards = []
    for o in report['objectives']:
        if not o['id'].startswith(group + '.'): continue
        cards.append(f'''<article><span class="badge">DSO {esc(o['id'])} · {esc(o['status'])}</span><h3>{esc(o['name'])}</h3><p>{esc(o['target'])}</p><p class="muted">{esc(o['provenance'])}</p><details><summary>Measured evidence</summary><pre>{esc(json.dumps(o['metrics'], indent=2))}</pre></details><p class="note">{esc(o['limitation'])}</p></article>''')
    pages.append(f'<section><div class="eyebrow">BUSINESS OBJECTIVE {group}</div><h2>{title}</h2><div class="cards">'+''.join(cards)+'</div></section>')
pages.append('''<section><div class="eyebrow">LIVE DEMONSTRATION</div><h2>Follow a case from message to decision</h2><ol><li>Select Company 6958 and inspect its existing portfolio.</li><li>Run the assistant and inspect actual tool calls and citations.</li><li>Edit or approve the draft, then simulate dispatch.</li><li>Review a renewal using the historical clock.</li><li>Demonstrate refusal, a quarantined batch and duplicate review.</li><li>Compare observed activity with the monitoring report.</li></ol><div class="note">General-condition PDFs and real longitudinal outcomes remain unavailable. Public product information cannot establish contractual exclusions. Human acceptance, faithfulness and judge agreement require genuine reviewer labels.</div></section>''')
document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NextDeal · Validation 2 Review</title><style>body{margin:0;background:#eef3f4;color:#183841;font:17px/1.6 Arial,sans-serif}section{min-height:90vh;padding:60px;box-sizing:border-box;border-bottom:1px solid #cbdcdf;max-width:1500px;margin:auto}h1{font:56px/1.2 Georgia,serif;max-width:900px}h2{font:38px Georgia,serif}h3{font-size:21px}.eyebrow{color:#126b65;letter-spacing:2px;font-size:13px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}article{background:white;border:1px solid #d0dfe1;border-radius:8px;padding:24px}.badge{font-size:12px;background:#e6f2ed;color:#245c50;padding:6px}.muted{color:#607b82;font-size:14px}.note{font-size:15px;padding:18px;background:#e1eeea;margin-top:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:400px;overflow:auto}summary{cursor:pointer}@media(max-width:850px){section{padding:25px}.cards{grid-template-columns:1fr}h1{font-size:36px}}@media print{section{break-after:page;min-height:0;padding:25px}.cards{gap:10px}article{padding:12px}details{display:none}}</style>'''+''.join(pages)+f'<footer style="padding:30px">Generated from measured evidence: {esc(report["generated_at"])}</footer></html>'
(ROOT / 'reports/Validation2_Review.html').write_text(document)
print('Saved reports/Validation2_Review.html')
