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

from pathlib import Path
from typing import Any


def load_interpreter(path: Path | str) -> Any:
    p = str(path)
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=p)
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore

        return Interpreter(model_path=p)
    except ImportError:
        pass
    import tensorflow as tf  # type: ignore

    return tf.lite.Interpreter(model_path=p)
