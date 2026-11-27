import hashlib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NEXTDEAL_DATA_DIR", ROOT / "data")).resolve()
CUSTOM_DATASET = DATA_DIR != (ROOT / "data").resolve()
dataset_key = hashlib.sha256(os.fsencode(DATA_DIR)).hexdigest()[:16]
default_runtime = ROOT / "runtime" / dataset_key if CUSTOM_DATASET else ROOT / "runtime"
RUNTIME = Path(os.environ.get("NEXTDEAL_RUNTIME", default_runtime)).resolve()
RUNTIME.mkdir(parents=True, exist_ok=True)
MODEL_DIR = RUNTIME / "models" if CUSTOM_DATASET else ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_PAGES = DATA_DIR / "raw" / ("source_pages" if CUSTOM_DATASET else "bh_public_pages")
DB_PATH = RUNTIME / "nextdeal.sqlite3"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b-instruct")
ARTIFACT_VERSION = "nextdeal-v2.0"
