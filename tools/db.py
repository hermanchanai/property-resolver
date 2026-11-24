"""Read-only Postgres access to the source database.

The DSN is never copied into this repo: it is read from SOURCE_ENV_FILE (default: this repo's
gitignored .env) as SUPABASE_DB_URL. Every transaction is forced READ ONLY and
asserted before any statement runs (the pooler silently drops connect-time options).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from dotenv import dotenv_values, load_dotenv

REPO = Path(__file__).resolve().parents[1]
load_dotenv(REPO / ".env", override=True)
DEFAULT_SOURCE_ENV = REPO / ".env"


def _dsn() -> str:
    env_file = Path(os.environ.get("SOURCE_ENV_FILE", DEFAULT_SOURCE_ENV))
    dsn = dotenv_values(env_file).get("SUPABASE_DB_URL")
    if not dsn:
        raise RuntimeError(f"SUPABASE_DB_URL missing from {env_file}")
    return dsn


@contextmanager
def ro_cursor():
    conn = psycopg2.connect(_dsn(), connect_timeout=20)
    try:
        conn.set_session(readonly=True, autocommit=False)
        cur = conn.cursor()
        cur.execute("set transaction read only")
        cur.execute("show transaction_read_only")
        if cur.fetchone()[0] != "on":
            raise RuntimeError("transaction is not read-only; refusing to query")
        yield cur
        conn.rollback()
    finally:
        conn.close()


def query(sql: str, params=None) -> tuple[list[str], list[tuple]]:
    with ro_cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return cols, cur.fetchall()
