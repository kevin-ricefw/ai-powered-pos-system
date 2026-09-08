"""
EfficientNet-B3 vegetable classifier API + GCS feedback collection.

Local:  python app.py  -> http://localhost:8080
"""

import base64
import time
import uuid
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from azure_blob import upload_pending_image
from config import PORT
from db import fetch_top_products
from gcs_feedback import (
    FEEDBACK_BUCKET,
    PREFIX_CONFIRM,
    PREFIX_CORRECT,
    PREFIX_NEW,
    filter_produce_names,
    list_produce_names,
    normalize_produce_name,
    upload_feedback_image,
)
from inference import classify_many, classes as MODEL_CLASSES

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="VeggieLens EfficientNet-B3 API",
    description=(
        "32-class produce classifier with camera UI, Swagger docs, and GCS feedback "
        "(confirmations / corrections / new produce)."
    ),
    version="1.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


class Prediction(BaseModel):
    label: str
    prob: float = Field(..., ge=0, le=1)
    threshold: float = Field(..., ge=0, le=1)


class InferRequest(BaseModel):
    file: Union[str, List[str]] = Field(
        ...,
        description=(
            "Base64-encoded image string, or up to 3 Base64 strings "
            "(screenshots from the weighing scale). "
            "Supports raw Base64 or data URLs (data:image/jpeg;base64,...)."
        ),
        examples=["/9j/4AAQSkZJRg...", ["/9j/4AAQ...", "/9j/4BBR..."]],
    )

    @field_validator("file")
    @classmethod
    def validate_file_count(cls, value: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(value, list):
            if not value:
                raise ValueError("At least one image is required")
            if len(value) > 3:
                raise ValueError("Maximum 3 images allowed")
        return value


class InferResponse(BaseModel):
    success: bool = True
    inference_time_ms: float
    predictions: List[Prediction]


class HealthResponse(BaseModel):
    status: str
    model: str
    feedback_bucket: str


class ProduceListResponse(BaseModel):
    produce: List[str]


class FeedbackSaveResponse(BaseModel):
    status: str = "saved"
    path: str


class DetectProductResponse(BaseModel):
    request_id: str
    product_ids: List[str]


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def camera_ui():
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index)


@app.get("/health", tags=["System"], summary="Health check", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model="efficientnet_b3.onnx",
        feedback_bucket=FEEDBACK_BUCKET or "not configured",
    )


@app.get(
    "/produce-list",
    tags=["Feedback"],
    summary="List produce names for autocomplete",
    response_model=ProduceListResponse,
)
async def produce_list(q: Optional[str] = Query(None, description="Optional filter prefix")):
    all_names = list_produce_names(MODEL_CLASSES)
    if q:
        return ProduceListResponse(produce=filter_produce_names(all_names, q))
    return ProduceListResponse(produce=all_names)


def _decode_base64_image(data: str) -> np.ndarray:
    payload = data.strip()
    if payload.startswith("data:"):
        if "," not in payload:
            raise ValueError("Invalid data URL image")
        payload = payload.split(",", 1)[1]

    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception:
        raise ValueError("Invalid base64 image data")

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image bytes")
    return image


def _images_from_base64_payload(file: Union[str, List[str]]) -> list[np.ndarray]:
    payloads = [file] if isinstance(file, str) else file
    images: list[np.ndarray] = []
    for payload in payloads:
        try:
            images.append(_decode_base64_image(payload))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return images


async def _images_from_multipart(request: Request) -> list[np.ndarray]:
    form = await request.form()
    uploads: list[UploadFile] = []

    file_field = form.get("file")
    if file_field is not None:
        if isinstance(file_field, UploadFile):
            uploads.append(file_field)
        else:
            raise HTTPException(status_code=400, detail="file must be an image upload")

    for key in ("file2", "file3"):
        extra = form.get(key)
        if extra is not None and isinstance(extra, UploadFile):
            uploads.append(extra)

    if not uploads:
        raise HTTPException(status_code=400, detail="Missing file field")
    if len(uploads) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 images allowed")

    images: list[np.ndarray] = []
    for upload in uploads:
        contents = await upload.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        images.append(image)
    return images


@app.post(
    "/api/infer",
    tags=["Inference"],
    summary="Classify image(s)",
    response_model=InferResponse,
    responses={
        400: {"description": "Invalid or unreadable image"},
        415: {"description": "Unsupported Content-Type"},
    },
)
async def infer(request: Request):
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        body = InferRequest.model_validate(await request.json())
        images = _images_from_base64_payload(body.file)
    elif content_type.startswith("multipart/form-data"):
        images = await _images_from_multipart(request)
    else:
        raise HTTPException(
            status_code=415,
            detail="Send JSON with base64 file field, or multipart form-data",
        )

    start_time = time.perf_counter()
    results = classify_many(images)
    inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return InferResponse(
        success=True,
        inference_time_ms=inference_time_ms,
        predictions=results,
    )


@app.post(
    "/api/detect-product",
    tags=["Inference"],
    summary="Classify image and match against a tenant's products",
    response_model=DetectProductResponse,
)
async def detect_product(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    tenant_id: str = Form(...),
):
    contents, content_type, _ = await _read_image(image)
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    request_id = str(uuid.uuid4())
    predictions = classify_many([frame], k=5)
    labels = [p["label"] for p in predictions]
    print(f"[detect-product] request_id={request_id} tenant_id={tenant_id} predicted_labels={labels}")
    matches = await run_in_threadpool(fetch_top_products, tenant_id, labels)
    product_ids = [m["product_id"] for m in matches]
    top_name = matches[0]["name"] if matches else None

    background_tasks.add_task(upload_pending_image, request_id, contents, content_type, top_name)

    return DetectProductResponse(request_id=request_id, product_ids=product_ids)


async def _read_image(image: UploadFile) -> tuple[bytes, Optional[str], Optional[str]]:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file")
    return contents, image.content_type, image.filename


@app.post(
    "/confirm",
    tags=["Feedback"],
    summary="Confirm prediction — save to confirmations/<label>/",
    response_model=FeedbackSaveResponse,
)
async def confirm_feedback(
    image: UploadFile = File(...),
    label: str = Form(...),
):
    try:
        normalized = normalize_produce_name(label)
        contents, content_type, filename = await _read_image(image)
        path = upload_feedback_image(
            PREFIX_CONFIRM, normalized, contents, content_type, filename
        )
        return FeedbackSaveResponse(path=path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.post(
    "/correct",
    tags=["Feedback"],
    summary="Correct prediction — save to corrections/<correct_label>/",
    response_model=FeedbackSaveResponse,
)
async def correct_feedback(
    image: UploadFile = File(...),
    correct_label: str = Form(...),
):
    try:
        normalized = normalize_produce_name(correct_label)
        contents, content_type, filename = await _read_image(image)
        path = upload_feedback_image(
            PREFIX_CORRECT, normalized, contents, content_type, filename
        )
        return FeedbackSaveResponse(path=path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.post(
    "/new-produce",
    tags=["Feedback"],
    summary="New produce — save to new/<produce_name>/",
    response_model=FeedbackSaveResponse,
)
async def new_produce_feedback(
    image: UploadFile = File(...),
    produce_name: str = Form(...),
):
    try:
        normalized = normalize_produce_name(produce_name)
        contents, content_type, filename = await _read_image(image)
        path = upload_feedback_image(
            PREFIX_NEW, normalized, contents, content_type, filename
        )
        return FeedbackSaveResponse(path=path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
