"""
classifier เวอร์ชัน TFLite — เบา เหมาะกับ Arduino Uno Q
ใช้ tflite-runtime ถ้ามี (pip install tflite-runtime) มิฉะนั้น fallback ไป tf.lite

API เหมือน PoseClassifier ใน classifier.py ใช้แทนกันได้
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .labels import LABELS, THAI_NAMES, Prediction

from .paths import CLASSIFIER_TFLITE as MODEL_PATH


def _load_interpreter(model_path: Path):
    """พยายามใช้ tflite_runtime ก่อน (เบา) แล้วค่อย fallback ไป tensorflow"""
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        return Interpreter(model_path=str(model_path))
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
        return Interpreter(model_path=str(model_path))
    except ImportError:
        pass
    import tensorflow as tf  # fallback หนักสุด
    return tf.lite.Interpreter(model_path=str(model_path))


class TFLitePoseClassifier:
    def __init__(self, model_path: Path | str | None = None) -> None:
        path = Path(model_path) if model_path else MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"ไม่พบ {path} — รัน `python convert_to_tflite.py` ก่อน"
            )
        self._interp = _load_interpreter(path)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        self._in_dim = int(self._in["shape"][-1])

    def _infer(self, feature: np.ndarray) -> np.ndarray:
        if feature.shape[-1] != self._in_dim:
            raise ValueError(
                f"feature ขนาด {feature.shape[-1]} ไม่ตรงกับโมเดล {self._in_dim}"
            )
        x = feature.reshape(1, self._in_dim).astype(np.float32)
        self._interp.set_tensor(self._in["index"], x)
        self._interp.invoke()
        return self._interp.get_tensor(self._out["index"])[0]

    def predict(self, feature: np.ndarray) -> Prediction:
        probs = self._infer(feature)
        idx = int(np.argmax(probs))
        label = LABELS[idx]
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
                label=LABELS[i],
                confidence=float(probs[i]),
                thai=THAI_NAMES.get(LABELS[i], LABELS[i]),
            )
            for i in order
        ]
