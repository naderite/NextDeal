#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# AppImage launchers can otherwise report themselves as Python's executable.
unset APPIMAGE APPDIR LD_PRELOAD
export NEXT_TELEMETRY_DISABLED=1
if [[ ! -x .venv/bin/python ]]; then
  /usr/bin/python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements-app.txt
npm ci --prefix frontend
npm run build --prefix frontend
if ! command -v ollama >/dev/null; then
  echo 'Install Ollama, then rerun setup. https://ollama.com/download' >&2
  exit 1
fi
if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; then
  echo 'Start ollama serve in another terminal, then run the model pull below.'
fi
echo 'Model setup: ollama pull qwen3:4b-instruct'
echo 'Optional multilingual retrieval: .venv/bin/pip install -r requirements-retrieval.txt'
echo 'Then: ./scripts/start.sh'
