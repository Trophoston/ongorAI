"""
ตัวโหลด TFLite interpreter ที่ใช้ร่วมกันทั้งโปรเจกต์
เลือก backend อัตโนมัติตามที่มีในเครื่อง โดยไล่จาก "เบาสุด" ก่อน:
  1. tflite-runtime      <- แนะนำบนบอร์ด ARM/aarch64 (เล็ก เร็ว มี wheel)
  2. ai-edge-litert      <- ตัวต่อจาก tflite-runtime ของ Google
  3. tensorflow (tf.lite) <- หนักสุด ใช้บนเดสก์ท็อปเวลา dev

ทั้งหมดคืน object ที่มี .allocate_tensors / .get_input_details /
.get_output_details / .set_tensor / .invoke / .get_tensor เหมือนกัน
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_BACKEND_NAME = "uninitialized"


def load_interpreter(path: Path | str) -> Any:
    global _BACKEND_NAME
    p = str(path)
    num_threads = max(1, int(os.getenv("ONGOR_TFLITE_THREADS", "4")))
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        _BACKEND_NAME = "tflite-runtime"
        return Interpreter(model_path=p, num_threads=num_threads)
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore

        _BACKEND_NAME = "ai-edge-litert"
        return Interpreter(model_path=p, num_threads=num_threads)
    except ImportError:
        pass
    import tensorflow as tf  # type: ignore

    _BACKEND_NAME = "tensorflow-lite"
    return tf.lite.Interpreter(model_path=p, num_threads=num_threads)


def backend_name() -> str:
    return _BACKEND_NAME
