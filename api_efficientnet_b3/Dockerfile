FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py inference.py gcs_feedback.py config.py db.py azure_blob.py ./
COPY models/efficientnet_b7_48class.onnx models/class_to_idx.json models/class_aliases.json ./models/
COPY static/ ./static/
COPY VeggieLens_API.postman_collection.json ./

ENV ONNX_PATH=/app/models/efficientnet_b7_48class.onnx
ENV CLASS_MAP_PATH=/app/models/class_to_idx.json

EXPOSE 8080 443

# Serve HTTPS on $PORT (443) only if real certs are mounted at /certs; otherwise
# fall back to plain HTTP on 8080 so a missing/unmounted cert can't crash-loop
# the container into an outage. Mount certs at /certs/fullchain.pem +
# /certs/privkey.pem to enable TLS.
CMD ["sh", "-c", "if [ -f /certs/fullchain.pem ] && [ -f /certs/privkey.pem ]; then exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --ssl-certfile /certs/fullchain.pem --ssl-keyfile /certs/privkey.pem; else echo 'No TLS certs at /certs -- falling back to plain HTTP on 8080' >&2; exec uvicorn app:app --host 0.0.0.0 --port 8080; fi"]
