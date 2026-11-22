"""Persistent demo state. Raw analytical data is never modified."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import DB_PATH


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect():
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init():
    with connect() as db:
        db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        INSERT OR IGNORE INTO settings VALUES('demo_date','2025-08-01');
        CREATE TABLE IF NOT EXISTS clients(id TEXT PRIMARY KEY,kind TEXT NOT NULL,label TEXT NOT NULL,data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS contracts(id TEXT PRIMARY KEY,client_id TEXT NOT NULL,data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS cases(id INTEGER PRIMARY KEY,client_id TEXT NOT NULL REFERENCES clients(id),
          subject TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'received',created_at TEXT NOT NULL,error TEXT);
        CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY,case_id INTEGER NOT NULL REFERENCES cases(id),
          role TEXT NOT NULL,body TEXT NOT NULL,created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS drafts(id INTEGER PRIMARY KEY,case_id INTEGER NOT NULL REFERENCES cases(id),
          version INTEGER NOT NULL,body TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'pending',evidence TEXT NOT NULL,
          quality TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(case_id,version));
        CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY,draft_id INTEGER REFERENCES drafts(id),
          action TEXT NOT NULL,reason TEXT NOT NULL,created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS outbox(id INTEGER PRIMARY KEY,draft_id INTEGER NOT NULL UNIQUE REFERENCES drafts(id),
          case_id INTEGER NOT NULL REFERENCES cases(id),body TEXT NOT NULL,dispatched_at TEXT NOT NULL,demo_date TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,kind TEXT NOT NULL,case_id INTEGER,data TEXT NOT NULL,created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS batches(id INTEGER PRIMARY KEY,digest TEXT NOT NULL UNIQUE,status TEXT NOT NULL,
          data TEXT NOT NULL,findings TEXT NOT NULL,trace TEXT NOT NULL,created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS imported_records(id TEXT PRIMARY KEY,batch_id INTEGER NOT NULL REFERENCES batches(id),data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS duplicate_reviews(id INTEGER PRIMARY KEY,left_id TEXT NOT NULL,right_id TEXT NOT NULL,
          score REAL NOT NULL,evidence TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'pending',UNIQUE(left_id,right_id));
        CREATE TABLE IF NOT EXISTS identity_links(source_id TEXT PRIMARY KEY,canonical_id TEXT NOT NULL,review_id INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS human_reviews(id INTEGER PRIMARY KEY,draft_id INTEGER NOT NULL REFERENCES drafts(id),
          reviewer TEXT NOT NULL,data TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(draft_id,reviewer));
        CREATE TABLE IF NOT EXISTS renewal_schedule(contract_id TEXT PRIMARY KEY REFERENCES contracts(id),
          contact_date TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'scheduled',case_id INTEGER REFERENCES cases(id));
        """)


def event(kind, data=None, case_id=None):
    with connect() as db:
        db.execute("INSERT INTO events(kind,case_id,data,created_at) VALUES(?,?,?,?)",
                   (kind, case_id, json.dumps(data or {}, default=str), now()))


def demo_date():
    with connect() as db:
        return db.execute("SELECT value FROM settings WHERE key='demo_date'").fetchone()[0]


def decode(row, fields=("data",)):
    if row is None:
        return None
    out = dict(row)
    for f in fields:
        if f in out:
            out[f] = json.loads(out[f])
    return out
