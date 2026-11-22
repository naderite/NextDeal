"""Execute baseline notebooks into reports, preserving the original delivered files."""
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / 'reports/notebook_runs'
output.mkdir(exist_ok=True)
results = []
for path in sorted((ROOT / 'notebooks').glob('0[1-4]*.ipynb')):
    print('Executing', path.name, flush=True)
    nb = nbformat.read(path, as_version=4)
    # Start the kernel using this environment's Python, not an unrelated saved kernelspec.
    nb.metadata['kernelspec'] = {'name': 'nextdeal', 'display_name': 'NextDeal', 'language': 'python'}
    try:
        NotebookClient(nb, timeout=900, kernel_name='nextdeal', resources={'metadata': {'path': str(ROOT / 'notebooks')}}).execute()
        results.append({'notebook': path.name, 'status': 'passed'})
    except Exception as exc:
        results.append({'notebook': path.name, 'status': 'failed', 'error': str(exc)[:3000]})
    nbformat.write(nb, output / path.name)
(output / 'execution_results.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
raise SystemExit(any(r['status'] == 'failed' for r in results))
