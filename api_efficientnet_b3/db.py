"""Tenant product lookup over an SSH-tunneled PostgreSQL connection.

Schema per tenant: wecomm_<tenant_id>. Matching is fuzzy text search against
products.slug (already lowercase/hyphenated) using difflib (stdlib) over an
ILIKE-narrowed candidate set.
"""

import difflib
import re

from psycopg2 import sql
from psycopg2.pool import ThreadedConnectionPool
from sshtunnel import SSHTunnelForwarder

from config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    SSH_HOST,
    SSH_PKEY_PATH,
    SSH_PORT,
    SSH_USER,
    USE_SSH_TUNNEL,
)

_tunnel: SSHTunnelForwarder | None = None
_pool: ThreadedConnectionPool | None = None


def _ensure_tunnel() -> SSHTunnelForwarder:
    global _tunnel
    if _tunnel is None or not _tunnel.is_active:
        _tunnel = SSHTunnelForwarder(
            (SSH_HOST, SSH_PORT),
            ssh_username=SSH_USER,
            ssh_pkey=SSH_PKEY_PATH,
            remote_bind_address=(DB_HOST, DB_PORT),
        )
        _tunnel.start()
    return _tunnel


def _ensure_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        if USE_SSH_TUNNEL:
            tunnel = _ensure_tunnel()
            host, port = "127.0.0.1", tunnel.local_bind_port
        else:
            host, port = DB_HOST, DB_PORT
        _pool = ThreadedConnectionPool(
            1,
            5,
            host=host,
            port=port,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
    return _pool


def _slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")


def fetch_top_products(tenant_id: str, labels: list[str]) -> list[dict]:
    """Fuzzy-match predicted class labels against a tenant's products.slug column.

    Returns every matching product, ranked best-first by match_score (0-100),
    so callers can both return product_ids and resolve the matched name (e.g.
    for tagging a stored image) without a second DB round trip.
    """
    # ponytail: tenant_id is quoted as an identifier (injection-safe) but not
    # checked against a tenant registry, per explicit instruction. Add a
    # registry lookup here if bad tenant_ids start reaching the DB.
    schema = sql.Identifier(f"wecomm_{tenant_id}")
    slugged_labels = [_slugify(label) for label in labels]
    patterns = [f"%{s}%" for s in slugged_labels]
    query = sql.SQL("SELECT uuid, name, slug FROM {}.products WHERE slug ILIKE ANY(%s)").format(schema)

    pool = _ensure_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT count(*) FROM {}.products").format(schema))
            total_count = cur.fetchone()[0]
            cur.execute(query, (patterns,))
            rows = cur.fetchall()
    finally:
        pool.putconn(conn)

    print(f"[db] tenant={tenant_id} total_products={total_count} patterns={patterns} matched={len(rows)}")

    if not rows:
        return []

    def score(slug: str) -> float:
        return max(
            difflib.SequenceMatcher(None, s, slug.lower()).ratio()
            for s in slugged_labels
        )

    scored = [(row, score(row[2])) for row in rows]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [
        {"product_id": str(row[0]), "name": row[1], "match_score": round(sc * 100, 2)}
        for row, sc in scored
    ]
