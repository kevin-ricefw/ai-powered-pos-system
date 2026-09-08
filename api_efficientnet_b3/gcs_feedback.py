"""GCS image-only feedback storage (no metadata files)."""

import re
import uuid
from typing import Optional

from config import FEEDBACK_BUCKET

PREFIX_CONFIRM = "confirmations"
PREFIX_CORRECT = "corrections"
PREFIX_NEW = "new"

MIME_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}

_gcs_client = None


def get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage

        _gcs_client = storage.Client()
    return _gcs_client


def normalize_produce_name(name: str) -> str:
    if not name or not str(name).strip():
        raise ValueError("Produce name cannot be empty")
    cleaned = str(name).strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("Produce name cannot be empty")
    return cleaned


def extension_from_upload(content_type: Optional[str], filename: Optional[str]) -> str:
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        if ct in MIME_EXT:
            return MIME_EXT[ct]
    if filename and "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()
        if re.match(r"^\.(jpe?g|png|webp|gif|bmp)$", ext):
            return ext if ext != ".jpeg" else ".jpg"
    return ".jpg"


def upload_feedback_image(
    folder_prefix: str,
    produce_name: str,
    data: bytes,
    content_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    if not FEEDBACK_BUCKET:
        raise RuntimeError("FEEDBACK_BUCKET is not configured")

    label = normalize_produce_name(produce_name)
    ext = extension_from_upload(content_type, filename)
    blob_path = f"{folder_prefix}/{label}/{uuid.uuid4().hex}{ext}"

    client = get_gcs_client()
    bucket = client.bucket(FEEDBACK_BUCKET)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(
        data,
        content_type=content_type or "image/jpeg",
    )
    return blob_path


def _folder_names_from_prefix(bucket_name: str, root_prefix: str) -> set[str]:
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    names: set[str] = set()
    iterator = bucket.list_blobs(prefix=root_prefix, delimiter="/")
    for page in iterator.pages:
        for prefix in page.prefixes:
            part = prefix.rstrip("/").split("/")[-1]
            if part:
                names.add(part)
    return names


def list_produce_names(model_classes: list[str]) -> list[str]:
    names: set[str] = set()
    for cls in model_classes:
        try:
            names.add(normalize_produce_name(cls))
        except ValueError:
            continue

    if FEEDBACK_BUCKET:
        try:
            for root in (PREFIX_CONFIRM, PREFIX_CORRECT, PREFIX_NEW):
                names.update(_folder_names_from_prefix(FEEDBACK_BUCKET, f"{root}/"))
        except Exception:
            pass

    return sorted(names)


def filter_produce_names(all_names: list[str], query: str, limit: int = 20) -> list[str]:
    q = normalize_produce_name(query) if query and query.strip() else ""
    if not q:
        return all_names[:limit]
    return [n for n in all_names if n.startswith(q)][:limit]
