#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset APPIMAGE APPDIR LD_PRELOAD
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m pytest -q
npm run build --prefix frontend
.venv/bin/python scripts/evaluate.py
