"""Application tools over the original datasets, with explicit provenance."""
import hashlib
import json
import math
import re
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.features import build_preprocessor
from src.nlp import BM25, IntentClassifier, build_corpus, fold_text
from src.nlp import DenseRetriever
from src.rules import baskets_from_contracts, mine_rules, observed_guarantees_by_product, recommend as guarantee_recommend
from src.segments import company_features, describe_segments
from src.utils import normalize_label

from .config import ARTIFACT_VERSION, CUSTOM_DATASET, DATA_DIR, MODEL_DIR, ROOT, SOURCE_PAGES
from .db import connect, decode, demo_date, event, now
from .review_metrics import draft_acceptance_metrics

RENEWAL_PREPARATION_LEAD_DAYS = 45

INTENT_EXAMPLES = {
    "interest": ["I am interested in this policy", "Please tell me more about the coverage", "We would like to proceed with insurance", "This product sounds useful for our company", "I want to learn about your protection", "Can you explain the benefits", "I would like to subscribe", "We need insurance for our business", "Please share more information", "Let us discuss this offer"],
    "hesitation": ["I need time to think about it", "I am not sure yet", "Let me discuss it with my family", "I will decide later", "Maybe next month", "I am hesitant to commit now", "Can we wait before deciding", "I have doubts about subscribing", "I would rather think first", "Please give us some time"],
    "already_insured": ["I already have insurance elsewhere", "Our company is covered by another insurer", "My employer already provides coverage", "We have an existing policy with a competitor", "I am already insured", "My bank covers this risk", "I do not need a second policy because I am covered", "Our current insurer handles this", "We already bought protection", "Another provider insures our vehicles"],
    "comparison": ["How does this compare with other policies", "What is the difference between these products", "Please compare the coverage options", "Which option has better benefits", "Can you provide a comparison", "How is your offer different", "Compare motor and property protection", "What distinguishes these two plans", "Which policy suits us better", "Show me the differences"],
    "quote": ["Please send me a quote", "How much does this insurance cost", "What is the price", "Can I get an estimate", "I would like a quotation for my car", "What premium would I pay", "Please calculate the annual rate", "Send your pricing details", "What is the total cost", "We need a price proposal"],
    "refusal": ["No thank you, I am not interested", "Please stop contacting me", "I refuse this offer", "Do not send me more proposals", "I do not want this policy", "Remove me from this campaign", "We decline your offer", "I have no interest in subscribing", "Please cancel this sales conversation", "Do not call me again"],
}


def intent_training_data():
    texts, labels = [], []
    for label, examples in INTENT_EXAMPLES.items():
        for example in examples:
            for prefix in ("", "Hello, ", "For our account: "):
                texts.append(prefix + example)
                labels.append(label)
    return texts, labels


def english_intent(text):
    """Explicit high-precision English rules, alongside the learned intent baseline."""
    t = text.lower()
    patterns = [
        ('refusal', r'\b(refuse|decline|cancel|stop contacting|remove me|not interested|no thank|do not (want|send|call)|no interest)\b'),
        ('already_insured', r'\b(already|existing policy|current insurer|another insurer|another provider|bank covers|employer.*coverage|competitor|am covered)\b'),
        ('quote', r'\b(quote|quotation|price|pricing|premium|cost|estimate|annual rate)\b'),
        ('comparison', r'\b(compare|comparison|difference|different|distinguishes|better|suits us)\b'),
        ('hesitation', r'\b(think|not sure|later|wait|hesitant|doubts|time|maybe|discuss.*family)\b'),
        ('interest', r'\b(interested|subscribe|proceed|learn|benefits|more information|more about|useful|discuss.*offer|need insurance)\b'),
    ]
    for label, pattern in patterns:
        if re.search(pattern, t):
            return label
    return None


def split_guarantees(value):
    return [v.strip() for v in re.split(r"[;,|]", str(value or "")) if v.strip()]


def safe_profiles(episodes):
    """Use raw profile labels and constant unknowns, not whole-dataset learned imputations."""
    raw = pd.read_csv(DATA_DIR / "raw/pm_product_portfolio.csv", sep=";")
    profile = raw.drop_duplicates("REF_pm").set_index("REF_pm")
    out = episodes.copy()
    for dest, source in (("sector", "LIB_SECTEUR_ACTIVITE"), ("activity", "LIB_ACTIVITE"), ("category", "category")):
        out[dest] = out["REF_pm"].map(profile[source].map(normalize_label)).fillna("UNKNOWN")
    out["sector_imputed"] = (out["sector"] == "UNKNOWN").astype(int)
    out[["has_history", "max_contract_age"]] = 0
    return out


class Engine:
    def __init__(self):
        self.companies = pd.read_csv(DATA_DIR / "processed/pm_companies_clean.csv").fillna("")
        self.episodes = pd.read_csv(DATA_DIR / "processed/pm_episodes.csv").fillna("")
        self.catalogue = json.loads((DATA_DIR / "processed/catalogue.json").read_text())
        self.products = self.catalogue["products"]
        self.branches = self.catalogue.get("product_branch", {})
        if self.products and not self.branches:
            raw = pd.read_csv(DATA_DIR / "raw/pm_product_portfolio.csv", sep=";")
            self.branches = raw.dropna(subset=["LIB_PRODUIT", "BRANCHE"]).drop_duplicates("LIB_PRODUIT").set_index("LIB_PRODUIT")["BRANCHE"].to_dict()
        self.corpus = build_corpus(SOURCE_PAGES)
        self.bm25 = BM25().fit(self.corpus["indexed_text"])
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                               strip_accents="unicode", sublinear_tf=True)
        self.char_passages = self.char_vectorizer.fit_transform(self.corpus["indexed_text"])
        self.dense = None
        self.dense_error = None
        local_embedding = ROOT / 'models/e5-small'
        if (local_embedding / 'modules.json').exists():
            try:
                self.dense = DenseRetriever(MODEL_DIR / 'rag_dense_cache.npz')
                self.dense.MODEL = str(local_embedding)
                self.dense.fit(self.corpus['indexed_text'])
            except Exception as exc:
                self.dense_error = str(exc)
                self.dense = None
        model_path = MODEL_DIR / "app_baseline.joblib"
        if model_path.exists():
            bundle = joblib.load(model_path)
            self.recommender, self.intent = bundle["recommender"], bundle["intent"]
        else:
            # Remove history inputs with uncertain hidden-product availability.
            self.recommender = None
            if not self.episodes.empty:
                train = safe_profiles(self.episodes)
                self.recommender = make_pipeline(build_preprocessor(self.products), LogisticRegression(C=.3, max_iter=3000))
                self.recommender.fit(train, train["hidden_product"])
            texts, labels = intent_training_data()
            self.intent = IntentClassifier().fit(texts, labels)
            joblib.dump({"version": ARTIFACT_VERSION, "recommender": self.recommender, "intent": self.intent}, model_path)
        labels_path = DATA_DIR / "processed" / "intent_training.json"
        self.custom_intents = CUSTOM_DATASET and labels_path.is_file()
        if self.custom_intents:
            training_bytes = labels_path.read_bytes()
            key = hashlib.sha256(training_bytes).hexdigest()
            artifact = MODEL_DIR / "custom_intent.joblib"
            intent_bundle = joblib.load(artifact) if artifact.is_file() else {}
            if intent_bundle.get("training_hash") == key:
                self.intent = intent_bundle["intent"]
            else:
                rows = json.loads(training_bytes)
                if not isinstance(rows, list) or not all(
                    isinstance(row, list) and len(row) == 2
                    and all(isinstance(value, str) and value.strip() for value in row)
                    for row in rows
                ) or len({row[1] for row in rows}) < 2:
                    raise ValueError("intent_training.json needs [message, label] pairs for at least two intents")
                self.intent = IntentClassifier().fit(*zip(*rows))
                joblib.dump({"training_hash": key, "intent": self.intent}, artifact)
        persona_features = company_features(self.companies)
        self.segment_category_weight = float(self.catalogue.get("segmentation_category_weight", 1.0))
        if not 0.0 < self.segment_category_weight <= 1.0:
            raise ValueError("segmentation_category_weight must be greater than zero and at most one")
        self.segment_features = persona_features.copy()
        category_columns = [name for name in persona_features if name.startswith("cat::")]
        self.segment_features[category_columns] *= self.segment_category_weight
        self.segmenter = None
        self.company_segment = {}
        self.personas = pd.DataFrame(index=pd.Index([], name="segment"))
        if not self.companies.empty:
            self.segmenter = KMeans(n_clusters=min(7, len(self.companies)), n_init=20, random_state=0).fit(self.segment_features)
            self.personas = describe_segments(persona_features, self.segmenter.labels_, self.companies)
            self.company_segment = {int(ref): int(seg) for ref, seg in zip(self.companies.REF_pm, self.segmenter.labels_)}
        pp = pd.read_csv(DATA_DIR / "processed/pp_renewals_clean.csv").fillna("")
        pp["guarantee_list"] = pp.apply(lambda r: [c[5:] for c in pp.columns if c.startswith("gar::") and r[c] == 1], axis=1)
        self.pp = pp
        baskets = baskets_from_contracts(pp)
        self.rules = mine_rules(baskets)
        self.observed_guarantees = observed_guarantees_by_product(baskets)
        self.seed()

    def seed(self):
        with connect() as db:
            for c in self.companies.to_dict("records"):
                ref = int(c["REF_pm"])
                c.setdefault("provenance", "real extract; identifiers pseudonymised")
                c.update({"segment": self.company_segment[ref],
                          "persona": self.personas.loc[self.company_segment[ref], "persona"]})
                db.execute("INSERT INTO clients VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data", (f"pm:{ref}", "corporate", c.get("display_label") or f"Company {ref}", json.dumps(c, default=str)))
            raw = json.loads((DATA_DIR / "raw/pp_contract_renewals.json").read_text())
            for item in raw:
                p = item["profile"]
                uid = str(item["user_id"])
                cid = f"pp:{uid}"
                data = {"product": p.get("LIB_PRODUIT"), "sector": p.get("Lib_Secteur_Activite") or "Unknown",
                        "category": p.get("Category"), "provenance": "real renewal extract; identifiers pseudonymised"}
                db.execute("INSERT OR IGNORE INTO clients VALUES(?,?,?,?)", (cid, "individual", f"Individual {uid}", json.dumps(data)))
                contract = {"product": p.get("LIB_PRODUIT"), "branch": p.get("LIB_BRANCHE"), "sub_branch": p.get("LIB_SOUS_BRANCHE"),
                            "effective_date": str(p.get("Effet_Contrat", ""))[:10], "expiry_date": str(p.get("Date_Expiration", ""))[:10],
                            "payment_status": p.get("statut_paiement"), "guarantees": split_guarantees(p.get("Lib_Garantie")),
                            "contract_age": p.get("Age_Contrat"), "provenance": p.get("provenance", "real historical extract")}
                contract_id = str(p.get("Num_Contrat"))
                db.execute("INSERT OR IGNORE INTO contracts VALUES(?,?,?)", (contract_id, cid, json.dumps(contract)))
                try:
                    contact_date = str(date.fromisoformat(contract['expiry_date']) - pd.Timedelta(days=RENEWAL_PREPARATION_LEAD_DAYS))[:10]
                    db.execute("INSERT INTO renewal_schedule(contract_id,contact_date) VALUES(?,?) ON CONFLICT(contract_id) DO UPDATE SET contact_date=excluded.contact_date WHERE renewal_schedule.status='scheduled'", (contract_id, contact_date))
                except ValueError:
                    pass
            if not self.companies.empty and not db.execute("SELECT 1 FROM cases LIMIT 1").fetchone():
                first = f"pm:{int(self.companies.iloc[0].REF_pm)}"
                cursor = db.execute("INSERT INTO cases(client_id,subject,status,created_at) VALUES(?,?,'received',?)", (first, "Explore additional business protection", now()))
                db.execute("INSERT INTO messages(case_id,role,body,created_at) VALUES(?,'client',?,?)", (cursor.lastrowid, "We are interested in protecting more of our business. What other insurance could suit our company?", now()))

    def client(self, client_id):
        with connect() as db:
            row = db.execute("SELECT * FROM clients WHERE id=?", (client_id,)).fetchone()
        if row is None:
            raise ValueError("Client not found")
        result = decode(row)
        result["value"] = self.value(result)
        return result

    def value(self, client):
        d = client["data"]
        depth = int(d.get("n_products", 1))
        history = float(d.get("max_contract_age", 0))
        factors = {"portfolio_depth": min(depth / 4, 1) * 40, "relationship_tenure": min(history / 10, 1) * 30,
                   "cross_sell_opportunity": max(0, 4 - depth) / 4 * 30}
        return {"score": round(sum(factors.values()), 1), "factors": factors,
                "type": "transparent outreach priority heuristic", "limitation": "Not validated against real future value; not a monetary CLV estimate"}

    def recommendations(self, client_id, k=3):
        c = self.client(client_id)
        if c["kind"] != "corporate":
            return {"recommendations": [], "reason": "Corporate recommendation model does not apply to individual policyholders"}
        d = c["data"]
        if "portfolio" not in d:
            return {"recommendations": [], "held": [], "reason": "Imported identity has no verified portfolio; advisor follow-up required"}
        held = split_guarantees(d["portfolio"])
        if self.recommender is None:
            return {"recommendations": [], "held": held, "reason": "No recommendation training data"}
        row = safe_profiles(pd.DataFrame([{**d, "visible_portfolio": ";".join(held), "n_visible": len(held), "has_history": 0, "max_contract_age": 0}]))
        scores = self.recommender.predict_proba(row)[0]
        candidates = self.rank_products(held, scores)
        return {"client_id": client_id, "held": held, "recommendations": candidates[:max(1, min(k, 5))],
                "model": "multinomial logistic baseline; history inputs disabled", "score_type": "relative ranking score, not purchase probability",
                "policy": "use a new insurance line to break score ties; advisor suitability review required", "version": ARTIFACT_VERSION}

    def rank_products(self, held, scores):
        classes = self.recommender.classes_
        held_lines = {self.branches.get(p) for p in held}
        candidates = [{"product": p, "score": round(float(s), 4), "line": self.branches.get(p, "Unknown"),
                       "new_line": self.branches.get(p) not in held_lines} for p, s in zip(classes, scores) if p not in held]
        candidates.sort(key=lambda x: (x["score"], x["new_line"]), reverse=True)
        return candidates

    def classify(self, text):
        learned = self.intent.classify(text)
        rule = None if self.custom_intents else english_intent(text)
        return {**learned, "intent": rule or learned['intent'], "method": "English rule + TF-IDF baseline" if rule else "TF-IDF baseline",
                "learned_prediction": learned['intent'], "uncertain": not rule and learned['confidence'] < .6}

    def search(self, query, k=4, segment=None):
        # Deterministic bilingual expansion complements English questions over French sources.
        words = {"car": "automobile", "motor": "automobile", "vehicle": "automobile", "fire": "incendie", "theft": "vol",
                 "home": "habitation", "property": "incendie multirisque", "health": "sante maladie", "life": "vie deces",
                 "travel": "voyage assistance", "construction": "chantier", "liability": "responsabilite civile", "machine": "bris machines",
                 "business": "professionnelle entreprises", "coverage": "garanties", "retirement": "retraite",
                 "towing": "remorquage", "distance": "kilometrage", "death": "deces",
                 "deduction": "deduction fiscale", "hospital": "hospitalisation",
                 "employer": "employeur", "premium": "prime cotisation", "aircraft": "aeronef",
                 "machinery": "machines materiel", "sites": "sites chantier", "surge": "surtension",
                 "computers": "informatique ordinateurs", "safe": "coffre fort",
                 "interruption": "pertes exploitation", "neighbours": "voisins",
                 "accident": "accident sinistre", "report": "declarer declaration",
                 "deadline": "delai", "documents": "pieces justificatives",
                 "medical": "medical soins", "cap": "plafond limite",
                 "days": "jours", "stolen": "vol vole", "company": "entreprise personne morale",
                 "limited": "limite", "formulas": "formules categories",
                 "refund": "remboursement", "costs": "frais", "falling": "chute",
                 "building": "batiment immeuble"}
        expanded = query + " " + " ".join(v for w, v in words.items() if re.search(r"\b" + w + r"\b", query.lower()))
        phrases = {r"\bkey[ -]person\b": "homme cle",
                   r"\baircraft\b": "navigation aerienne",
                   r"\b(?:tax[ -]deduction|tax[ -]deductible)\b": "deductible fiscalement"}
        expanded += " " + " ".join(translation for pattern, translation in phrases.items()
                                  if re.search(pattern, fold_text(query)))
        # Catalogue identifiers must retrieve their own product family, never borrow
        # an unrelated policy's attractive coverage (e.g. motor benefits for property).
        folded = fold_text(query)
        scope = None
        legacy_scopes = [
            (r'multirisques? professionnelles?', 'professionnels-entreprises__assurance-multirisque-professionnelle'),
            (r'faculte|police abonnement|transport marchandises', 'professionnels-entreprises__assurance-transport-marchandises'),
            (r'sante et prevoyance|frais.sante', 'professionnels-entreprises__assurance-frais-sante'),
            (r'bris de machines', 'professionnels-entreprises__assurance-bris-machine'),
            (r'engins de chantier', 'professionnels-entreprises__assurance-engins-chantier'),
        ]
        for pattern, slug in (() if CUSTOM_DATASET else legacy_scopes):
            if re.search(pattern, folded):
                scope = slug
                break
        lexical = self.bm25.scores(expanded)
        order = np.argsort(-lexical, kind="stable")
        if self.dense is not None:
            dense_scores = self.dense.scores(query)
            if expanded.strip() != query.strip():
                dense_scores = .5 * dense_scores + .5 * self.dense.scores(expanded.strip())
            if not np.any(lexical > 0) and float(np.max(dense_scores)) < .81:
                return {"passages": [], "method": "hybrid with conservative no-evidence threshold",
                        "limitation": "No sufficiently supported passage; ask an advisor"}
            query_chars = self.char_vectorizer.transform([query])
            char_scores = (self.char_passages @ query_chars.T).toarray().ravel()
            fused = np.zeros(len(lexical))
            for scores, weight in ((dense_scores, 2), (lexical, 1), (char_scores, 1)):
                ranks = np.argsort(np.argsort(-scores, kind="stable"), kind="stable")
                fused += weight / (5 + ranks)
            order = np.argsort(-fused, kind="stable")
        out = []
        for i in order:
            row = self.corpus.iloc[int(i)]
            if scope and row['page'] != scope:
                continue
            if segment and row["segment"] not in (segment, "general"):
                continue
            if self.dense is None and lexical[i] <= 0:
                continue
            out.append({**row[["chunk_id", "title", "url", "text"]].to_dict(), "rank": len(out) + 1,
                        "source_type": "public product information", "language": "source language" if CUSTOM_DATASET else "French", "contractual_conditions": False,
                        "product_scope": scope or row['page']})
            if len(out) >= k:
                break
        return {"passages": out, "method": "BM25, character matching and multilingual-e5 rank fusion" if self.dense else "BM25 with English/French expansion",
                "limitation": "General-condition PDFs not available; do not infer exclusions or contractual terms"}

    def contracts(self, client_id=None):
        with connect() as db:
            rows = db.execute("SELECT * FROM contracts" + (" WHERE client_id=?" if client_id else ""), (client_id,) if client_id else ()).fetchall()
        out = []
        today = date.fromisoformat(demo_date())
        for row in rows:
            r = decode(row)
            try:
                r["days_to_expiry"] = (date.fromisoformat(r["data"]["expiry_date"]) - today).days
            except ValueError:
                r["days_to_expiry"] = None
            out.append(r)
        return sorted(out, key=lambda x: x["days_to_expiry"] if x["days_to_expiry"] is not None else 99999)

    def renewal(self, client_id):
        items = self.contracts(client_id)
        out = []
        for c in items:
            d, days = c["data"], c["days_to_expiry"]
            unpaid = str(d["payment_status"]).lower() in ("impaye", "impayé", "non paye", "non payé", "unpaid")
            action = "escalate" if unpaid or (days is not None and days < 0) else "remind" if days is not None and days <= 45 else "schedule"
            basket = frozenset([f"gar::{g}" for g in d["guarantees"]] + [f"sub::{d['sub_branch']}", f"prod::{d['product']}"])
            allowed = self.observed_guarantees.get(f"prod::{d['product']}", set())
            suggestions = guarantee_recommend(basket, self.rules, allowed=allowed)
            if action == 'remind' and suggestions:
                action = 'offer'
            out.append({"contract_id": c["id"], "product": d["product"], "expiry_date": d["expiry_date"], "days_to_expiry": days,
                        "action": action, "scheduled_contact_date": str(date.fromisoformat(d["expiry_date"]) - pd.Timedelta(days=RENEWAL_PREPARATION_LEAD_DAYS))[:10] if days is not None else None,
                        "guarantee_suggestions": suggestions,
                        "payment": {"current_status": d["payment_status"], "operational_priority": "high" if unpaid else "standard",
                                    "prediction": self.payment_prediction(d), "limitation": "Current unpaid status is a collection trigger, not a future-risk prediction"}})
        return {"client_id": client_id, "demo_date": demo_date(), "contracts": out, "mode": "historical simulation"}

    def payment_prediction(self, contract):
        path = MODEL_DIR / "app_payment.joblib"
        if not path.exists():
            return {"available": False, "reason": "No validated pre-outcome payment model artifact"}
        bundle = joblib.load(path)
        features = np.array([[float(contract.get("contract_age") or 0), len(contract.get("guarantees", []))]])
        return {"score": round(float(bundle["model"].predict_proba(features)[0, 1]), 3), "provenance": "small real historical sample",
                "limitation": "Exploratory retrospective score; pre-outcome feature availability unverified; do not use as calibrated future risk"}

    def weekly_activity(self, end=None):
        end = end or datetime.now(timezone.utc)
        start = end - timedelta(days=7)
        bounds = (start.isoformat(), end.isoformat())
        with connect() as db:
            events = db.execute(
                """SELECT kind, COUNT(*) n FROM events
                   WHERE created_at >= ? AND created_at < ? AND kind != 'monitoring_report'
                   GROUP BY kind""", bounds
            ).fetchall()
            decisions = db.execute(
                """SELECT action, COUNT(*) n FROM decisions
                   WHERE created_at >= ? AND created_at < ? GROUP BY action""", bounds
            ).fetchall()
            dispatches = db.execute(
                "SELECT COUNT(*) FROM outbox WHERE dispatched_at >= ? AND dispatched_at < ?", bounds
            ).fetchone()[0]
        return {"period_start_utc": bounds[0], "period_end_utc": bounds[1],
                "events": {row["kind"]: row["n"] for row in events},
                "decisions": {row["action"]: row["n"] for row in decisions},
                "simulated_dispatches": dispatches}

    def renewal_dispatch_metrics(self):
        contracts = self.contracts()
        expiry = {c["id"]: date.fromisoformat(c["data"]["expiry_date"])
                  for c in contracts if c["days_to_expiry"] is not None}
        with connect() as db:
            schedules = db.execute("SELECT contract_id, case_id FROM renewal_schedule").fetchall()
            dispatches = db.execute(
                "SELECT case_id, MIN(demo_date) AS first_dispatch FROM outbox GROUP BY case_id"
            ).fetchall()
        first_dispatch = {row["case_id"]: row["first_dispatch"] for row in dispatches}
        timely = sum(
            (expiry[row["contract_id"]] - date.fromisoformat(first_dispatch[row["case_id"]])).days >= 30
            for row in schedules if row["contract_id"] in expiry and row["case_id"] in first_dispatch
        )
        return {"contracts": len(contracts), "contracts_with_expiry": len(expiry),
                "scheduled_contacts": len(schedules), "timely_approved_dispatches": timely,
                "approved_dispatch_coverage": timely / len(expiry) if expiry else None}

    def metrics(self):
        with connect() as db:
            cases = db.execute("SELECT status,COUNT(*) n FROM cases GROUP BY status").fetchall()
            approvals = db.execute("SELECT action,COUNT(*) n FROM decisions GROUP BY action").fetchall()
            sent = db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
            events = db.execute("SELECT kind,COUNT(*) n FROM events GROUP BY kind").fetchall()
            drafts = db.execute("SELECT COUNT(*) FROM drafts").fetchone()[0]
            times = [json.loads(r[0]).get('elapsed_seconds') for r in db.execute("SELECT data FROM events WHERE kind='draft_generated'")]
        actions = {r["action"]: r["n"] for r in approvals}
        acceptance = draft_acceptance_metrics()
        return {"cases_by_status": {r["status"]: r["n"] for r in cases}, "decisions": actions, "simulated_dispatches": sent,
                "draft_versions": drafts, "renewal_dispatch": self.renewal_dispatch_metrics(), "events": {r["kind"]: r["n"] for r in events},
                "median_draft_seconds": float(np.median([v for v in times if isinstance(v, (int, float))])) if any(isinstance(v, (int, float)) for v in times) else None,
                "approval_without_edit_rate": acceptance["approval_without_edit_rate"],
                "reviewed_generated_drafts": acceptance["reviewed"],
                "provenance": "observed local demo activity; no real conversion or renewal uplift measured"}


@lru_cache(maxsize=1)
def engine():
    return Engine()
