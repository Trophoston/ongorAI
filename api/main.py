from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from box.realtime import LabelSmoother, RollingMetrics
from ongor.tflite_util import backend_name

# Vision/pose imports — required for this API. If they fail, /predict returns 503
# with the import error so the cause is obvious instead of a bare 503.
HAS_VISION = False
VISION_IMPORT_ERROR: str | None = None
PoseEngine = None
try:
    import cv2
    import numpy as np
    from ongor.pose_engine import PoseEngine

    HAS_VISION = True
except Exception as e:  # noqa: BLE001
    cv2 = None  # type: ignore
    np = None  # type: ignore
    VISION_IMPORT_ERROR = f"{type(e).__name__}: {e}"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class PosePredictor:
    """Lazily builds a single shared PoseEngine and runs predictions on frames."""

    def __init__(self) -> None:
        self._engine: PoseEngine | None = None
        self._lock = threading.Lock()
        self._metrics = RollingMetrics()
        self._smoother = LabelSmoother(
            window_size=int(os.getenv("ONGOR_SMOOTH_WINDOW", "5")),
            switch_ratio=float(os.getenv("ONGOR_SMOOTH_SWITCH_RATIO", "1.2")),
            max_missing=int(os.getenv("ONGOR_SMOOTH_MAX_MISSING", "2")),
        )
        self._stream_id: str | None = None

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None
            self._smoother.reset()

    def _ensure_engine(self) -> PoseEngine:
        if not HAS_VISION:
            raise HTTPException(
                status_code=503,
                detail=f"Vision/pose detection unavailable: {VISION_IMPORT_ERROR}",
            )
        if self._engine is None:
            self._engine = PoseEngine(
                flip=_env_bool("ONGOR_POSE_FLIP"),
                hold_time=float(os.getenv("ONGOR_POSE_HOLD", "0.6")),
                conf_threshold=float(os.getenv("ONGOR_POSE_CONF", "0.85")),
            )
        return self._engine

    def predict_image(
        self, image_bytes: bytes, stream_id: str | None = None
    ) -> dict[str, Any]:
        if not HAS_VISION:
            raise HTTPException(
                status_code=503,
                detail=f"Vision/pose detection unavailable: {VISION_IMPORT_ERROR}",
            )
        decode_started = time.perf_counter()
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        self._metrics.observe(
            "decode_ms", (time.perf_counter() - decode_started) * 1000.0
        )
        if frame is None:
            raise HTTPException(status_code=400, detail="cannot decode image")

        with self._lock:
            engine = self._ensure_engine()
            if stream_id is not None and stream_id != self._stream_id:
                self._stream_id = stream_id
                self._smoother.reset()
                engine.stabilizer.reset()
            inference_started = time.perf_counter()
            _, predictions = engine.process_topk(frame, k=3)
            self._metrics.observe(
                "inference_ms",
                (time.perf_counter() - inference_started) * 1000.0,
            )
            pred = predictions[0] if predictions else None
            raw_label = pred.label if pred is not None else None
            raw_conf = pred.confidence if pred is not None else 0.0
            label, conf = self._smoother.update(raw_label, raw_conf)
            confirmed = engine.stabilizer.update(label, conf)
            self._metrics.mark_ai(time.monotonic())

        return {
            "prediction": asdict(pred) if pred is not None else None,
            "top_predictions": [asdict(item) for item in predictions],
            "smoothed_prediction": (
                {"label": label, "confidence": conf} if label is not None else None
            ),
            "confirmed": confirmed,
            "pose_detected": pred is not None,
        }

    def metrics_snapshot(self) -> dict[str, Any]:
        snapshot = self._metrics.snapshot()
        snapshot.setdefault("decode_ms", 0.0)
        snapshot.setdefault("inference_ms", 0.0)
        return snapshot


app = FastAPI(title="Ong-Or Pose Predict API", version="0.3.0")
predictor = PosePredictor()
_box_metrics_lock = threading.Lock()
_box_metrics: dict[str, Any] = {}


class BoxMetrics(BaseModel):
    capture_fps: float = 0.0
    ai_fps: float = 0.0
    jpeg_ms: float = 0.0
    request_ms: float = 0.0
    dropped_frames: int = 0


@app.post("/metrics/box")
def update_box_metrics(metrics: BoxMetrics) -> dict[str, bool]:
    with _box_metrics_lock:
        _box_metrics.clear()
        values = metrics.model_dump() if hasattr(metrics, "model_dump") else metrics.dict()
        _box_metrics.update(values)
    return {"ok": True}

cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
def on_shutdown() -> None:
    predictor.close()


@app.get("/health")
def health() -> dict[str, Any]:
    metrics = predictor.metrics_snapshot()
    with _box_metrics_lock:
        box_metrics = dict(_box_metrics)
    server_ai_fps = metrics.get("ai_fps", 0.0)
    metrics.update(box_metrics)
    metrics["inference_fps"] = server_ai_fps
    metrics.setdefault("capture_fps", 0.0)
    metrics.setdefault("jpeg_ms", 0.0)
    metrics.setdefault("request_ms", 0.0)
    metrics.setdefault("dropped_frames", 0)
    metrics["tflite_backend"] = backend_name()
    metrics["tflite_threads"] = max(1, int(os.getenv("ONGOR_TFLITE_THREADS", "4")))
    return {
        "ok": True,
        "ts": time.time(),
        "vision": HAS_VISION,
        "error": VISION_IMPORT_ERROR,
        "pose": {
            "flip": _env_bool("ONGOR_POSE_FLIP"),
            "hold": float(os.getenv("ONGOR_POSE_HOLD", "0.6")),
            "conf": float(os.getenv("ONGOR_POSE_CONF", "0.85")),
            "presence": float(os.getenv("ONGOR_POSE_PRESENCE", "0.6")),
            "roi_margin": float(os.getenv("ONGOR_POSE_ROI_MARGIN", "0.85")),
            "search": _env_bool("ONGOR_POSE_SEARCH", True),
            "reacquire_interval": max(
                1, int(os.getenv("ONGOR_POSE_REACQUIRE_INTERVAL", "3"))
            ),
            "smooth_window": int(os.getenv("ONGOR_SMOOTH_WINDOW", "5")),
            "smooth_max_missing": int(
                os.getenv("ONGOR_SMOOTH_MAX_MISSING", "2")
            ),
        },
        "metrics": metrics,
    }


@app.get("/labels")
def labels() -> dict[str, Any]:
    """ท่าที่โมเดลรู้จัก (ตัด idle ออก) + ชื่อแสดงผล — ให้ฝั่งเกมดึงไปสร้างโจทย์"""
    from ongor.labels import EN_NAMES, GAME_POSES, THAI_NAMES

    poses = list(GAME_POSES)
    return {
        "poses": poses,
        "en": {p: EN_NAMES.get(p, p) for p in poses},
        "thai": {p: THAI_NAMES.get(p, p) for p in poses},
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    stream_id: Optional[str] = Form(default=None),
) -> dict[str, Any]:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    return predictor.predict_image(data, stream_id=stream_id)
