"""Azure Blob Storage upload for images received by /api/detect-product.

Stored under <product_name>/<request_id>.jpg so a human can review pending
detections by product, keyed for lookup by request_id once the frontend
confirms back which product was actually picked.
"""

import re
from typing import Optional

from config import AZURE_CONTAINER_NAME, AZURE_STORAGE_CONNECTION_STRING

_blob_service_client = None


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
