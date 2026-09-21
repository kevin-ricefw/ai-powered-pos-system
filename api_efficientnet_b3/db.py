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

from azure_blob import resolve_tenant_image_url
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
            connect_timeout=5,
        )
    return _pool


def _slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")


# ponytail: tax_type/discount_type/discount_value/scale_lb_factor aren't
# stored per-product anywhere in this schema (checked products, categories,
# tax_rates, settings) — per explicit instruction, kept as static defaults
# until there's a real source for them.
_STATIC_PRODUCT_FIELDS = {
    "tax_type": "percentage",
    "discount_type": "percentage",
    "discount_value": 0,
    "scale_lb_factor": None,
}

MIN_MATCH_SCORE = 50.0  # percent; preferred match floor, relaxed to backfill below MIN_RESULTS
MIN_RESULTS = 3  # backfill with next-best-scoring matches (even below MIN_MATCH_SCORE) up to this many
MAX_RESULTS = 10  # hard cap on returned products


def _fallback_thumbnail(name: str) -> str:
    initials = (name or "WECOMM").strip()[:2]
    return f"https://ui-avatars.com/api/?name={initials}&color=6B7280&background=F3F4F6"


def _fetch_product_details(
    cur, schema: sql.Identifier, tenant_id: str, product_ids: list[int]
) -> dict[int, dict]:
    cur.execute(
        sql.SQL(
            """
            SELECT
                p.id, p.uuid, p.name, p.sku, p.price, p.scale, p.disable_discount,
                p.has_depositable_products,
                c.id, c.name, c.image,
                u.id, u.name, u.symbol, u.type, u.ratio, u.category_id,
                t.percentage
            FROM {schema}.products p
            LEFT JOIN {schema}.categories c ON c.id = p.category_id
            LEFT JOIN {schema}.unit_of_measurements u ON u.id = p.sales_unit_of_measurement_id
            LEFT JOIN {schema}.tax_rates t ON t.id = p.tax_rate_id
            WHERE p.id = ANY(%s)
            """
        ).format(schema=schema),
        (product_ids,),
    )
    details = {}
    for row in cur.fetchall():
        (
            pid, uuid_, name, sku, price, scale, disable_discount, has_depositable_products,
            cat_id, cat_name, cat_image,
            uom_id, uom_name, uom_symbol, uom_type, uom_ratio, uom_category_id,
            tax_rate,
        ) = row
        details[pid] = {
            "id": pid,
            "uuid": str(uuid_),
            "name": name,
            "sku": sku,
            "price": float(price),
            "thumbnail": _fallback_thumbnail(name),
            "category": (
                {"id": cat_id, "name": cat_name, "image_url": resolve_tenant_image_url(tenant_id, cat_image)}
                if cat_id is not None
                else None
            ),
            "sales_unit_of_measurement": (
                {
                    "id": uom_id,
                    "name": uom_name,
                    "symbol": uom_symbol,
                    "type": uom_type,
                    "ratio": float(uom_ratio) if uom_ratio is not None else None,
                    "category_id": uom_category_id,
                }
                if uom_id is not None
                else None
            ),
            "tax_rate": float(tax_rate) if tax_rate is not None else None,
            "scale": scale,
            "disable_discount": disable_discount,
            "has_depositable_products": has_depositable_products,
            "deposit_products": [],
            **_STATIC_PRODUCT_FIELDS,
        }
    return details


def _fetch_thumbnails(cur, schema: sql.Identifier, tenant_id: str, product_ids: list[int]) -> dict[int, str]:
    cur.execute(
        sql.SQL(
            """
            SELECT DISTINCT ON (product_id) product_id, url
            FROM {}.product_images
            WHERE product_id = ANY(%s)
            ORDER BY product_id, id DESC
            """
        ).format(schema),
        (product_ids,),
    )
    resolved = ((pid, resolve_tenant_image_url(tenant_id, url)) for pid, url in cur.fetchall())
    return {pid: url for pid, url in resolved if url}


def _fetch_deposit_products(cur, schema: sql.Identifier, product_ids: list[int]) -> dict[int, list[dict]]:
    cur.execute(
        sql.SQL(
            """
            SELECT pd.parent_product_id, dp.id, dp.name, dp.price, pd.quantity
            FROM {schema}.product_deposits pd
            JOIN {schema}.products dp ON dp.id = pd.deposit_product_id
            WHERE pd.parent_product_id = ANY(%s)
            """
        ).format(schema=schema),
        (product_ids,),
    )
    deposits: dict[int, list[dict]] = {}
    for parent_id, dep_id, dep_name, dep_price, quantity in cur.fetchall():
        deposits.setdefault(parent_id, []).append(
            {"id": dep_id, "name": dep_name, "price": float(dep_price), "quantity": float(quantity)}
        )
    return deposits


def fetch_top_products(tenant_id: str, labels: list[str]) -> list[dict]:
    """Fuzzy-match predicted class labels (already expanded to their aliases,
    across the top-k predicted classes) against a tenant's products.slug column.

    Returns between MIN_RESULTS and MAX_RESULTS distinct products (fewer only
    if the tenant has fewer matching candidates than MIN_RESULTS total),
    ordered the same as `labels` (i.e. detection order — highest-confidence
    prediction first) among matches scoring at/above MIN_MATCH_SCORE, ties
    broken by match score descending. If fewer than MIN_RESULTS clear
    MIN_MATCH_SCORE, backfilled with the next-best-scoring candidates
    regardless of threshold. Each result is enriched with full product
    details (category, sales unit of measurement, tax rate, deposit
    products, ...) for direct display/checkout use.
    """
    # ponytail: tenant_id is quoted as an identifier (injection-safe) but not
    # checked against a tenant registry, per explicit instruction. Add a
    # registry lookup here if bad tenant_ids start reaching the DB.
    schema = sql.Identifier(f"wecomm_{tenant_id}")
    slugged_labels = [_slugify(label) for label in labels]
    patterns = [f"%{s}%" for s in slugged_labels]
    query = sql.SQL("SELECT id, uuid, name, slug FROM {}.products WHERE slug ILIKE ANY(%s)").format(schema)

    pool = _ensure_pool()
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT count(*) FROM {}.products").format(schema))
            total_count = cur.fetchone()[0]
            cur.execute(query, (patterns,))
            rows = cur.fetchall()

            if not rows:
                print(f"[db] tenant={tenant_id} total_products={total_count} patterns={patterns} matched=0")
                return []

            def match_info(slug: str) -> tuple[float, int]:
                """Best fuzzy score against any alias, plus the rank (index into
                slugged_labels, i.e. detection order) of the earliest alias that
                clears MIN_MATCH_SCORE — used to sort by detection order rather
                than by score."""
                ratios = [
                    difflib.SequenceMatcher(None, s, slug.lower()).ratio()
                    for s in slugged_labels
                ]
                best_score = max(ratios)
                matched_rank = next(
                    (i for i, r in enumerate(ratios) if r * 100 >= MIN_MATCH_SCORE),
                    len(ratios),
                )
                return best_score, matched_rank

            scored = [(row, *match_info(row[3])) for row in rows]
            top = [item for item in scored if item[1] * 100 >= MIN_MATCH_SCORE]
            top.sort(key=lambda item: (item[2], -item[1]))

            if len(top) < MIN_RESULTS:
                backfill = [item for item in scored if item[1] * 100 < MIN_MATCH_SCORE]
                backfill.sort(key=lambda item: -item[1])
                top += backfill[: MIN_RESULTS - len(top)]

            top = top[:MAX_RESULTS]
            top_ids = [row[0] for row, _, _ in top]

            details = _fetch_product_details(cur, schema, tenant_id, top_ids)
            thumbnails = _fetch_thumbnails(cur, schema, tenant_id, top_ids)
            deposits = _fetch_deposit_products(cur, schema, top_ids)
    finally:
        pool.putconn(conn)

    print(f"[db] tenant={tenant_id} total_products={total_count} patterns={patterns} matched={len(rows)}")

    results = []
    for row, sc, _rank in top:
        pid = row[0]
        product = details[pid]
        if pid in thumbnails:
            product["thumbnail"] = thumbnails[pid]
        product["deposit_products"] = deposits.get(pid, [])
        product["match_score"] = round(sc * 100, 2)
        results.append(product)
    return results
