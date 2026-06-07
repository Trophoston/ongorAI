"""
classifier สำหรับ pipeline MediaPipe — รับ keypoints (132,) คืน Prediction
ใช้ classifier_mp.tflite + label_map.json (ผลผลิตจาก train_mediapipe.py)

interface เหมือน PoseClassifier เดิม เพื่อให้เกม/main.py ใช้แทนกันได้
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .labels import THAI_NAMES, Prediction
from .mediapipe_runner import load_label_map
from .tflite_util import load_interpreter as _load_interpreter

from .paths import CLASSIFIER_MP_TFLITE as MODEL_PATH


class MediaPipeClassifier:
    def __init__(self, model_path: Path | str | None = None) -> None:
        path = Path(model_path) if model_path else MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"ไม่พบ {path} — รัน train_mediapipe.py ก่อน"
            )
        self._interp = _load_interpreter(path)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        self._in_dim = int(self._in["shape"][-1])
        self._labels = load_label_map()

    def _infer(self, feature: np.ndarray) -> np.ndarray:
        if feature.shape[-1] != self._in_dim:
            raise ValueError(
                f"feature ขนาด {feature.shape[-1]} ไม่ตรงกับโมเดล {self._in_dim}"
            )
        x = feature.reshape(1, self._in_dim).astype(np.float32)
        self._interp.set_tensor(self._in["index"], x)
        self._interp.invoke()
        return self._interp.get_tensor(self._out["index"])[0]

    def _label(self, idx: int) -> str:
        return self._labels.get(idx, str(idx))

    def predict(self, feature: np.ndarray) -> Prediction:
        probs = self._infer(feature)
        idx = int(np.argmax(probs))
        label = self._label(idx)
        return Prediction(
            label=label,
            confidence=float(probs[idx]),
            thai=THAI_NAMES.get(label, label),
        )

    def predict_topk(self, feature: np.ndarray, k: int = 3) -> list[Prediction]:
        probs = self._infer(feature)
        order = np.argsort(probs)[::-1][:k]
        return [
            Prediction(
                label=self._label(int(i)),
                confidence=float(probs[i]),
                thai=THAI_NAMES.get(self._label(int(i)), self._label(int(i))),
            )
            for i in order
        ]
