"""Azure Blob Storage upload for images received by /api/detect-product.

Stored under <product_name>/<request_id>.jpg so a human can review pending
detections by product, keyed for lookup by request_id once the frontend
confirms back which product was actually picked.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from config import (
    AZURE_CONTAINER_NAME,
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_TENANT_IMAGES_CONNECTION_STRING,
    AZURE_TENANT_IMAGES_CONTAINER,
)

_blob_service_client = None
_tenant_images_client = None


def get_blob_service_client():
    global _blob_service_client
    if _blob_service_client is None:
        from azure.storage.blob import BlobServiceClient

        _blob_service_client = BlobServiceClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING
        )
    return _blob_service_client


def _safe_folder(name: Optional[str]) -> str:
    if not name or not name.strip():
        return "unmatched"
    return re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower())


def upload_pending_image(
    request_id: str,
    contents: bytes,
    content_type: Optional[str],
    product_name: Optional[str] = None,
) -> Optional[str]:
    if not AZURE_STORAGE_CONNECTION_STRING:
        print(f"[azure_blob] AZURE_STORAGE_CONNECTION_STRING not set — skipping upload for {request_id}")
        return None

    blob_name = f"{_safe_folder(product_name)}/{request_id}.jpg"
    client = get_blob_service_client()
    container = client.get_container_client(AZURE_CONTAINER_NAME)
    container.upload_blob(
        name=blob_name,
        data=contents,
        content_type=content_type or "image/jpeg",
        overwrite=True,
    )
    return blob_name


def get_tenant_images_client():
    global _tenant_images_client
    if _tenant_images_client is None:
        from azure.storage.blob import BlobServiceClient

        _tenant_images_client = BlobServiceClient.from_connection_string(
            AZURE_TENANT_IMAGES_CONNECTION_STRING
        )
    return _tenant_images_client


def resolve_tenant_image_url(tenant_id: str, path: Optional[str], expiry_minutes: int = 60) -> Optional[str]:
    """Turn a relative storage path (as stored in the tenant DB, e.g.
    "images/categories/xyz.png") into a temporary read-only SAS URL. All
    tenants share one container (AZURE_TENANT_IMAGES_CONTAINER); the blob is
    prefixed per tenant as "wecomm<tenant_id>/<path>". Passes an
    already-absolute URL through unchanged. Returns None if there's no path
    or no connection string configured.
    """
    if not path:
        return None
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not AZURE_TENANT_IMAGES_CONNECTION_STRING:
        return None

    from azure.storage.blob import BlobSasPermissions, generate_blob_sas

    client = get_tenant_images_client()
    blob_name = f"wecomm{tenant_id}/{path}"
    sas = generate_blob_sas(
        account_name=client.account_name,
        container_name=AZURE_TENANT_IMAGES_CONTAINER,
        blob_name=blob_name,
        account_key=client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes),
    )
    blob_url = client.get_blob_client(container=AZURE_TENANT_IMAGES_CONTAINER, blob=blob_name).url
    return f"{blob_url}?{sas}"
