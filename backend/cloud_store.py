"""Durable per-node state in managed PostgreSQL. Never writes to Vercel disk."""
import os
from contextlib import contextmanager
import psycopg
from psycopg.rows import dict_row

DDL='''CREATE TABLE IF NOT EXISTS linguasculpt_tasks (
 id TEXT PRIMARY KEY, payload JSONB NOT NULL, state JSONB NOT NULL DEFAULT '{}',
 status TEXT NOT NULL, stage TEXT NOT NULL DEFAULT 'prepare', version INTEGER NOT NULL DEFAULT 0,
 error TEXT, updated TIMESTAMPTZ NOT NULL DEFAULT now())'''


@contextmanager
def database():
    url=os.getenv('DATABASE_URL')
    if not url:raise RuntimeError('DATABASE_URL is not configured')
    with psycopg.connect(url,row_factory=dict_row,connect_timeout=10) as conn:
        conn.execute(DDL)
        yield conn
