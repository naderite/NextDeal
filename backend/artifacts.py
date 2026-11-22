"""Controlled refresh: train a candidate, check it, then atomically promote."""
import json
import os
import threading
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from .config import ARTIFACT_VERSION, ROOT, RUNTIME
from .db import event, now
from .services import engine, safe_profiles
from src.features import build_preprocessor

REFRESH_LOCK = threading.Lock()
MANIFEST = RUNTIME / 'artifact_manifest.json'


def promotion_gate(candidate_hit, baseline_hit, corpus_chunks):
    return corpus_chunks > 0 and np.isfinite(candidate_hit) and candidate_hit >= baseline_hit - .05


def refresh_artifacts():
    if not REFRESH_LOCK.acquire(blocking=False):
        return {"status": "already_running"}
    manifest = {"status": "training", "started_at": now(), "version": ARTIFACT_VERSION}
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    try:
        e = engine()
        data = safe_profiles(e.episodes)
        tr, te = next(GroupKFold(5).split(data, groups=data.REF_pm))
        model = make_pipeline(build_preprocessor(e.products), LogisticRegression(C=.3, max_iter=3000))
        model.fit(data.iloc[tr], data.iloc[tr].hidden_product)
        p = model.predict_proba(data.iloc[te]); classes = model.classes_
        popular = data.iloc[tr].hidden_product.value_counts().index.tolist()
        hit, base = [], []
        for j, (_, row) in enumerate(data.iloc[te].iterrows()):
            held = set(row.visible_portfolio.split(';'))
            ranked = [classes[k] for k in np.argsort(-p[j]) if classes[k] not in held]
            hit.append(row.hidden_product in ranked[:3])
            base.append(row.hidden_product in [x for x in popular if x not in held][:3])
        candidate_hit, baseline_hit = float(np.mean(hit)), float(np.mean(base))
        passed = promotion_gate(candidate_hit, baseline_hit, len(e.corpus))
        manifest.update({"candidate_HitRate@3": candidate_hit, "popularity_HitRate@3": baseline_hit,
                         "gate": "non-inferior within 5 percentage points on fixed company holdout; non-empty retrieval corpus",
                         "limitation": "Operational promotion gate, not independent final model-selection evidence"})
        if not passed:
            manifest['status'] = 'rejected; previous artifacts retained'
        else:
            model.fit(data, data.hidden_product)
            candidate = RUNTIME / 'candidate.joblib'
            joblib.dump({"version": ARTIFACT_VERSION, "recommender": model, "intent": e.intent}, candidate)
            # Verify artifact reload before replacing the known-working file.
            loaded = joblib.load(candidate)
            loaded['recommender'].predict_proba(data.iloc[:1])
            os.replace(candidate, ROOT / 'models/app_baseline.joblib')
            manifest['status'] = 'promoted'
            engine.cache_clear()
        manifest['finished_at'] = now()
    except Exception as exc:
        manifest.update({"status": "failed; previous artifacts retained", "error": str(exc), "finished_at": now()})
    finally:
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        event('artifact_refresh', manifest)
        REFRESH_LOCK.release()
    return manifest
