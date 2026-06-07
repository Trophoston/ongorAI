from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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


class PosePredictor:
    """Lazily builds a single shared PoseEngine and runs predictions on frames."""

    def __init__(self) -> None:
        self._engine: PoseEngine | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.close()
                self._engine = None

    def _ensure_engine(self) -> PoseEngine:
        if not HAS_VISION:
            raise HTTPException(
                status_code=503,
                detail=f"Vision/pose detection unavailable: {VISION_IMPORT_ERROR}",
            )
        if self._engine is None:
            self._engine = PoseEngine()
        return self._engine

    def predict_image(self, image_bytes: bytes) -> dict[str, Any]:
        if not HAS_VISION:
            raise HTTPException(
                status_code=503,
                detail=f"Vision/pose detection unavailable: {VISION_IMPORT_ERROR}",
            )
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="cannot decode image")

        with self._lock:
            engine = self._ensure_engine()
            _, pred = engine.process(frame)
            label = pred.label if pred is not None else None
            conf = pred.confidence if pred is not None else 0.0
            confirmed = engine.stabilizer.update(label, conf)

        return {
            "prediction": asdict(pred) if pred is not None else None,
            "confirmed": confirmed,
            "pose_detected": pred is not None,
        }


app = FastAPI(title="Ong-Or Pose Predict API", version="0.2.0")
predictor = PosePredictor()

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
    return {
        "ok": True,
        "ts": time.time(),
        "vision": HAS_VISION,
        "error": VISION_IMPORT_ERROR,
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
async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    return predictor.predict_image(data)
