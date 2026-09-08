FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model downloaded from GCS during Cloud Build (see cloudbuild.yaml).
COPY app.py inference.py gcs_feedback.py config.py db.py azure_blob.py class_thresholds.csv ./
COPY models/models_efficientnet_b3.onnx ./
COPY static/ ./static/
COPY VeggieLens_API.postman_collection.json ./

ENV ONNX_PATH=/app/models_efficientnet_b3.onnx
ENV THRESHOLD_PATH=/app/class_thresholds.csv

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
