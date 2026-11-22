#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset APPIMAGE APPDIR LD_PRELOAD
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:4b-instruct}"
export LANGCHAIN_TRACING_V2=false
export LANGSMITH_TRACING=false
export NEXT_TELEMETRY_DISABLED=1
mkdir -p runtime
if [[ ! -x .venv/bin/python || ! -d frontend/node_modules ]]; then
  echo 'Run ./scripts/setup.sh first.' >&2
  exit 1
fi
if [[ ! -d frontend/.next ]]; then npm run build --prefix frontend; fi
owned_ollama=''
cleanup() {
  kill "${backend_pid:-}" "${frontend_pid:-}" 2>/dev/null || true
  if [[ -n "$owned_ollama" ]]; then kill "$owned_ollama" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  ollama serve >runtime/ollama.log 2>&1 &
  owned_ollama=$!
  for attempt in {1..30}; do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi
    sleep 1
  done
fi
.venv/bin/python - <<'PY'
import httpx,os,socket
models=[m['name'] for m in httpx.get('http://127.0.0.1:11434/api/tags').json()['models']]
if os.environ['OLLAMA_MODEL'] not in models:
    raise SystemExit('Missing local model. Run: ollama pull '+os.environ['OLLAMA_MODEL'])
for port in (8000,3000):
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',port)) == 0:
            raise SystemExit(f'Port {port} is in use. Stop the previous NextDeal instance first.')
PY
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 >runtime/backend.log 2>&1 &
backend_pid=$!
npm run start --prefix frontend >runtime/frontend.log 2>&1 &
frontend_pid=$!
echo 'NextDeal advisor: http://127.0.0.1:3000'
echo 'API documentation: http://127.0.0.1:8000/docs'
echo 'Logs: runtime/backend.log and runtime/frontend.log'
wait -n "$backend_pid" "$frontend_pid"
