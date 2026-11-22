import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("NEXTDEAL_RUNTIME", ROOT / "runtime"))
RUNTIME.mkdir(parents=True, exist_ok=True)
DB_PATH = RUNTIME / "nextdeal.sqlite3"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b-instruct")
ARTIFACT_VERSION = "nextdeal-v2.0"
