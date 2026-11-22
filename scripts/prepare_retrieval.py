"""Download a multilingual encoder once, save it locally, and cache corpus embeddings."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sentence_transformers import SentenceTransformer
from src.nlp import DenseRetriever, build_corpus

path = ROOT / 'models/e5-small'
print('Preparing multilingual-e5-small on CPU', flush=True)
model = SentenceTransformer(str(path) if (path / 'modules.json').exists() else 'intfloat/multilingual-e5-small', device='cpu')
model.save(str(path))
corpus = build_corpus(ROOT / 'data/raw/bh_public_pages')
dense = DenseRetriever(ROOT / 'models/rag_dense_cache.npz')
dense.model_ = model
dense.fit(corpus.indexed_text)
print('Cached', len(corpus), 'passages at', path, flush=True)
