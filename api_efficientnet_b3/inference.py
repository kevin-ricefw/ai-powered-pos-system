"""ONNX EfficientNet-B3 inference (no Flask/camera deps — Cloud Run safe)."""

import csv

import cv2
import numpy as np
import onnxruntime as ort

from config import ONNX_PATH, THRESHOLD_PATH

classes: list[str] = []
thresholds: dict[str, float] = {}
with open(THRESHOLD_PATH) as f:
    for row in csv.DictReader(f):
        classes.append(row["class"])
        thresholds[row["class"]] = float(row["threshold"])

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(
    ONNX_PATH,
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name
IMG_SIZE = 300

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[None]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _topk_from_probs(probs: np.ndarray, k: int = 3) -> list[dict]:
    top_idx = probs.argsort()[::-1][:k]
    return [
        {
            "label": classes[i],
            "prob": float(probs[i]),
            "threshold": thresholds[classes[i]],
        }
        for i in top_idx
    ]


def classify(frame_bgr: np.ndarray, k: int = 3) -> list[dict]:
    inp = preprocess(frame_bgr)
    logits = session.run(None, {input_name: inp})[0][0]
    return _topk_from_probs(softmax(logits), k)


def classify_many(frames_bgr: list[np.ndarray], k: int = 3) -> list[dict]:
    """Run inference on one or more images; average logits when multiple are sent."""
    if not frames_bgr:
        raise ValueError("At least one image is required")
    if len(frames_bgr) == 1:
        return classify(frames_bgr[0], k)

    logits_list = []
    for frame_bgr in frames_bgr:
        inp = preprocess(frame_bgr)
        logits_list.append(session.run(None, {input_name: inp})[0][0])
    avg_logits = np.mean(logits_list, axis=0)
    return _topk_from_probs(softmax(avg_logits), k)
