"""NLP building blocks for DSO3.1 (intent & objection classifier) and DSO3.2 (retrieval).

* `prototype_intent` ports the regex rules of the NextDeal prototype
  (`agent/intent_analysis/config.py` of github.com/naderite/Next-BH) and is used as the baseline.
* `IntentClassifier` = TF-IDF (word 1-2 grams + character 2-5 grams) + logistic regression.
* The retrieval corpus is built from a snapshot of BH Assurance's **public product pages**
  (`data/raw/bh_public_pages/`), chunked by paragraph. Retrievers: BM25, dense
  multilingual-e5-small (same family as the prototype's e5-large), and their reciprocal-rank fusion.
"""
from __future__ import annotations

import hashlib
import html as _html
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

INTENTS = ["interest", "hesitation", "already_insured", "comparison", "quote", "refusal"]


def fold_text(text: str) -> str:
    """Lower-case, strip accents, normalise apostrophes and spaces."""
    t = unicodedata.normalize("NFKD", str(text).lower().replace("’", "'"))
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------- DSO3.1 baseline (prototype rules)
_PROTOTYPE_PATTERNS = [   # (intent, regex) in the prototype's priority order, French labels mapped to ours
    ("refusal", r"(pas\s+d['e]\s*(interet|besoin)|\baucun(e)?\b|pas\s*(du tout|interess[eé])|(ça|cela)\s+ne\s+m['’]interess(e|e pas)|"
                r"\bnon\s+merci\b|\brefus[eé]?\b|\b(j['’]en veux pas|je n['’]ai pas besoin|inutile)\b)"),
    ("hesitation", r"(pas\s*s[uû]r(e)?|h[eé]site(r|z)?|je\s*(vais|veux)?\s*y\s*r[eé]fl[eé]chir|je\s*(vais|voudrais)?\s*d[eé]cider plus tard|"
                   r"(je|on)\s+verra|je\s*(vais|voudrais)?\s*y penser|(besoin|donner)\s+du\s+temps|\bj'y pense encore\b)"),
    ("already_insured", r"(d[eé]j[aà]|(j['’]ai|nous avons)\s+une\s+autre\s+assurance|assur[eé]\s+ailleurs|je\s+suis\s+couvert|"
                        r"(j['’]ai|nous avons)\s+une\s+couverture|(mon|ma)\s+(banque|employeur)\s+me\s+couvre|assur[eé]\s+autrement)"),
    ("comparison", r"(diff[eé]rence(s)?|comparer|comparaison|(faire|avoir)\s+un\s+comparatif|(quelle|quelles)\s+sont\s+les\s+options|"
                   r"(qu['’]est-ce qui)\s+change)"),
    ("quote", r"(devis|tarif(s)?|prix|cout|co[uû]t|combien\s+(ça|cela)\s+co[uû]te|estimation|montant|offre\s+de\s+prix)"),
]


def prototype_intent(text: str) -> str:
    """Prototype behaviour: first matching regex wins, otherwise 'interest' (its 'intérêt global' fallback)."""
    t = fold_text(text)
    for intent, pattern in _PROTOTYPE_PATTERNS:
        if re.search(pattern, t):
            return intent
    return "interest"


class IntentClassifier:
    """TF-IDF word + char n-grams -> multinomial logistic regression, with a confidence score."""

    def __init__(self, C: float = 8.0):
        self.C = C

    def fit(self, texts, labels) -> "IntentClassifier":
        feats = FeatureUnion([
            ("word", TfidfVectorizer(preprocessor=fold_text, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("char", TfidfVectorizer(preprocessor=fold_text, analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ])
        self.pipe_ = make_pipeline(feats, LogisticRegression(C=self.C, max_iter=5000))
        self.pipe_.fit(list(texts), list(labels))
        self.classes_ = list(self.pipe_.classes_)
        return self

    def predict(self, texts) -> list[str]:
        return list(self.pipe_.predict(list(texts)))

    def predict_proba(self, texts) -> np.ndarray:
        return self.pipe_.predict_proba(list(texts))

    def classify(self, text: str) -> dict:
        p = self.predict_proba([text])[0]
        order = np.argsort(-p)
        return {"intent": self.classes_[order[0]], "confidence": round(float(p[order[0]]), 3),
                "alternatives": [{"intent": self.classes_[i], "p": round(float(p[i]), 3)} for i in order[1:3]]}


# --------------------------------------------------------------------------- DSO3.2 corpus
_STAFF_MARKERS = {"Responsable"}
_START_MARKERS = ["Demander un devis", "Contacter l'un de nos partenaires"]      # end of navigation
_END_MARKERS = ("Les Experts de", "Leader de l'assurance digitale")


def _page_lines(raw_html: str) -> list[str]:
    s = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw_html)
    s = re.sub(r"(?is)<br\s*/?>", "\n", s)
    s = re.sub(r"(?s)<[^>]+>", "\n", s)
    s = _html.unescape(s).replace("\xa0", " ")
    return [re.sub(r"\s+", " ", l).strip() for l in s.split("\n") if l.strip()]


def extract_product_page(raw_html: str) -> tuple[str, list[str]]:
    """Title + content lines of a product page (navigation, staff names and footer removed)."""
    lines = _page_lines(raw_html)
    title = lines[0].split("|")[0].strip()
    start = next((lines.index(m) + 1 for m in _START_MARKERS if m in lines), 0)
    end = next((i for i in range(start, len(lines)) if lines[i].startswith(_END_MARKERS)), len(lines))
    body, skip = [], False
    for l in lines[start:end]:
        if skip:                      # the line after "Responsable" is a staff member's name
            skip = False
            continue
        if l in _STAFF_MARKERS:
            skip = True
            continue
        body.append(l)
    return title, body


def extract_faq(raw_html: str) -> list[tuple[str, str]]:
    m = re.search(r"(?is)<main[^>]*>(.*?)</main>", raw_html)
    lines = _page_lines(m.group(1) if m else raw_html)
    qa, q, a = [], None, []
    for l in lines:
        if l.endswith("?") and len(l) > 15:
            if q:
                qa.append((q, " ".join(a)))
            q, a = l, []
        elif q:
            a.append(l)
    if q:
        qa.append((q, " ".join(a)))
    return qa


def build_corpus(pages_dir: Path, max_words: int = 110) -> pd.DataFrame:
    """Chunk every page: paragraphs merged up to ~max_words, each chunk prefixed with its product title."""
    manifest = json.loads((pages_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for page in manifest:
        raw = (pages_dir / page["file"]).read_text(encoding="utf-8", errors="replace")
        slug = page["file"].removesuffix(".html")
        segment = "professional" if slug.startswith("professionnels") else ("individual" if slug.startswith("particuliers") else "general")
        if slug.startswith("faq"):
            for j, (q, a) in enumerate(extract_faq(raw)):
                rows.append({"chunk_id": f"{slug}#{j}", "page": slug, "title": "FAQ", "segment": segment,
                             "url": page["url"], "text": f"{q} {a}".strip()})
            continue
        title, body = extract_product_page(raw)
        buf, j = [], 0
        for l in body:
            buf.append(l)
            if sum(len(b.split()) for b in buf) >= max_words:
                rows.append({"chunk_id": f"{slug}#{j}", "page": slug, "title": title, "segment": segment,
                             "url": page["url"], "text": " ".join(buf)})
                buf, j = [], j + 1
        if buf:
            rows.append({"chunk_id": f"{slug}#{j}", "page": slug, "title": title, "segment": segment,
                         "url": page["url"], "text": " ".join(buf)})
    df = pd.DataFrame(rows)
    df["indexed_text"] = df["title"] + " — " + df["text"]
    return df


# --------------------------------------------------------------------------- retrievers
_STOP = set("""a au aux avec ce ces cette d de des du elle en est et etre il ils je l la le les leur lui ma mais me mes mon
ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes ton tu un une vos votre vous y c s n j m t
est-ce quel quelle quels quelles comment dans plus tout tous est sont ai as avez avons a-t-il bh assurance""".split())


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", fold_text(text)) if w not in _STOP and len(w) > 1]


class BM25:
    def __init__(self, k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs) -> "BM25":
        self.docs_ = [Counter(tokens(d)) for d in docs]
        self.len_ = np.array([sum(d.values()) for d in self.docs_], dtype=float)
        self.avg_ = self.len_.mean()
        df = Counter(w for d in self.docs_ for w in d)
        n = len(self.docs_)
        self.idf_ = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        return self

    def scores(self, query: str) -> np.ndarray:
        q = tokens(query)
        out = np.zeros(len(self.docs_))
        for i, d in enumerate(self.docs_):
            norm = self.k1 * (1 - self.b + self.b * self.len_[i] / self.avg_)
            out[i] = sum(self.idf_.get(w, 0) * d[w] * (self.k1 + 1) / (d[w] + norm) for w in q if w in d)
        return out


class DenseRetriever:
    """multilingual-e5-small sentence embeddings (cosine)."""

    MODEL = "intfloat/multilingual-e5-small"

    def __init__(self, cache: Path | None = None):
        self.cache = cache

    def _model(self):
        if not hasattr(self, "model_"):
            from sentence_transformers import SentenceTransformer
            self.model_ = SentenceTransformer(self.MODEL)
        return self.model_

    def fit(self, docs) -> "DenseRetriever":
        docs = list(docs)
        key = hashlib.md5("␞".join(docs).encode("utf-8")).hexdigest()
        if self.cache is not None and self.cache.exists():
            z = np.load(self.cache, allow_pickle=False)
            if str(z["key"]) == key:
                self.emb_ = z["emb"]
                return self
        self.emb_ = self._model().encode([f"passage: {d}" for d in docs], normalize_embeddings=True, batch_size=32)
        if self.cache is not None:
            np.savez(self.cache, emb=self.emb_, key=key)
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self._model().encode([f"query: {query}"], normalize_embeddings=True)[0]
        return self.emb_ @ q


class HybridRetriever:
    """Reciprocal-rank fusion of BM25 and dense rankings (k = 60, no weight to tune)."""

    def __init__(self, corpus: pd.DataFrame, dense: DenseRetriever | None = None, rrf_k: int = 60):
        self.corpus = corpus.reset_index(drop=True)
        self.bm25 = BM25().fit(self.corpus["indexed_text"])
        self.dense = dense.fit(self.corpus["indexed_text"]) if dense is not None else None
        self.rrf_k = rrf_k

    def rank(self, query: str, method: str = "hybrid") -> np.ndarray:
        if method == "bm25" or self.dense is None:
            return np.argsort(-self.bm25.scores(query), kind="stable")
        if method == "dense":
            return np.argsort(-self.dense.scores(query), kind="stable")
        rb = np.argsort(np.argsort(-self.bm25.scores(query), kind="stable"))
        rd = np.argsort(np.argsort(-self.dense.scores(query), kind="stable"))
        return np.argsort(-(1 / (self.rrf_k + rb + 1) + 1 / (self.rrf_k + rd + 1)), kind="stable")

    def search(self, query: str, k: int = 5, method: str = "hybrid", segment: str | None = None) -> list[dict]:
        order = self.rank(query, method)
        if segment:
            order = [i for i in order if self.corpus.at[i, "segment"] in (segment, "general")]
        cols = ["chunk_id", "title", "url", "text"]
        return [{**self.corpus.loc[i, cols].to_dict(), "rank": r + 1} for r, i in enumerate(order[:k])]


# --------------------------------------------------------------------------- grounded (extractive) answers
def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?:])\s+", text) if len(s.strip()) > 20]


def extractive_answer(question: str, passages: list[dict], n_sentences: int = 2) -> dict:
    """Pick the passage sentences sharing most content words with the question; each keeps its citation."""
    q = set(tokens(question))
    cands = []
    for p in passages:
        for s in split_sentences(p["text"]):
            overlap = len(q & set(tokens(s)))
            cands.append((overlap, -p["rank"], s, p["url"], p["chunk_id"]))
    cands.sort(reverse=True)
    picked = cands[:n_sentences]
    return {"answer": " ".join(c[2] for c in picked), "citations": sorted({c[4] for c in picked}),
            "sentences": [{"text": c[2], "url": c[3], "chunk_id": c[4]} for c in picked]}


def support_rate(answer_sentences: list[str], context: str, threshold: float = 0.8) -> float:
    """Share of answer sentences whose content words are >= `threshold` present in the context.

    A reusable, model-free faithfulness proxy: it is trivially ~1 for extractive answers, and becomes
    informative when an LLM rewrites the text.
    """
    ctx = set(tokens(context))
    ok = [len(set(tokens(s)) & ctx) / max(len(set(tokens(s))), 1) >= threshold for s in answer_sentences]
    return float(np.mean(ok)) if ok else float("nan")
