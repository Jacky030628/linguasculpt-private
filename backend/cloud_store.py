"""Durable per-node state in managed PostgreSQL. Never writes to Vercel disk."""
import os
import threading
from contextlib import contextmanager
import psycopg
from psycopg.rows import dict_row

DDL='''CREATE TABLE IF NOT EXISTS linguasculpt_tasks (
 id TEXT PRIMARY KEY, payload JSONB NOT NULL, state JSONB NOT NULL DEFAULT '{}',
 status TEXT NOT NULL, stage TEXT NOT NULL DEFAULT 'prepare', version INTEGER NOT NULL DEFAULT 0,
 error TEXT, updated TIMESTAMPTZ NOT NULL DEFAULT now())'''

_schema_ready = False
_schema_lock = threading.Lock()


def ensure_schema(url):
    global _schema_ready
    with _schema_lock:
        if _schema_ready:return
        with psycopg.connect(url,connect_timeout=10,options='-c lock_timeout=5000 -c statement_timeout=15000') as conn:
            conn.execute(DDL)
            conn.execute('ALTER TABLE linguasculpt_tasks ADD COLUMN IF NOT EXISTS lease_id TEXT')
            conn.execute('ALTER TABLE linguasculpt_tasks ADD COLUMN IF NOT EXISTS lease_until TIMESTAMPTZ')
        _schema_ready = True


@contextmanager
def database():
    url=os.getenv('DATABASE_URL')
    if not url:raise RuntimeError('DATABASE_URL is not configured')
    ensure_schema(url)
    with psycopg.connect(url,row_factory=dict_row,connect_timeout=10,
                         options='-c lock_timeout=5000 -c statement_timeout=15000') as conn:
        yield conn
