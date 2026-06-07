"""โหลด Teachable Machine classifier head และทำนายท่า"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from .labels import LABELS, THAI_NAMES, Prediction

from .paths import CLASSIFIER_KERAS as MODEL_PATH


class PoseClassifier:
    def __init__(self, model_path: Path | str | None = None) -> None:
        path = Path(model_path) if model_path else MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"ไม่พบ {path} — รัน `python convert_model.py` ก่อน"
            )
        self._model = tf.keras.models.load_model(path, compile=False)

    def predict(self, feature: np.ndarray) -> Prediction:
        """รับ feature (14739,) -> Prediction"""
        x = feature[None, :].astype(np.float32)
        probs = self._model(x, training=False).numpy()[0]
        idx = int(np.argmax(probs))
        label = LABELS[idx]
        return Prediction(
            label=label,
            confidence=float(probs[idx]),
            thai=THAI_NAMES.get(label, label),
        )

    def predict_topk(self, feature: np.ndarray, k: int = 3) -> list[Prediction]:
        x = feature[None, :].astype(np.float32)
        probs = self._model(x, training=False).numpy()[0]
        order = np.argsort(probs)[::-1][:k]
        return [
            Prediction(
                label=LABELS[i],
                confidence=float(probs[i]),
                thai=THAI_NAMES.get(LABELS[i], LABELS[i]),
            )
            for i in order
        ]
