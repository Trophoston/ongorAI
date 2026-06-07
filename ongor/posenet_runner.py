"""
PoseNet feature extractor — เลียนแบบ pre-processing ของ Teachable Machine Pose
ให้ได้ vector ขนาด 14739 = heatmap(17*17*17) + offset(17*17*34)

ใช้โมเดล PoseNet MobileNetV1 (multiplier 0.75, stride 16, input 257)
ดาวน์โหลดครั้งแรกอัตโนมัติจาก storage.googleapis.com ของ tfjs-models
"""
from __future__ import annotations

import sys
import types
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def _numpy_compat_shim() -> None:
    """
    tfjs_graph_converter รุ่นเก่าใช้ alias ที่ numpy>=1.24 ถอดออกแล้ว
    (np.bool/np.object/np.int/np.float) — คืนให้ชั่วคราวก่อน import
    """
    for name, real in (
        ("bool", np.bool_),
        ("object", object),
        ("int", int),
        ("float", float),
        ("complex", complex),
        ("str", str),
    ):
        if not hasattr(np, name):
            setattr(np, name, real)


def _stub_decision_forests() -> None:
    """
    tfjs_graph_converter -> tensorflowjs ทำการ `import tensorflow_decision_forests`
    แบบ hard import ซึ่งบางเครื่องชนกับ protobuf (gencode/runtime mismatch)
    เราไม่ได้ใช้ TFDF เลย จึงยัดโมดูลปลอมเข้า sys.modules ให้ import ผ่านไป
    (ปลอดภัย: tensorflowjs อ้างชื่อนี้แค่ตอน import ไม่ได้เรียกใช้งานจริงใน path ของเรา)
    """
    if "tensorflow_decision_forests" in sys.modules:
        return
    try:
        import tensorflow_decision_forests  # noqa: F401
        return  # ของจริงใช้ได้อยู่แล้ว ไม่ต้อง stub
    except Exception:  # noqa: BLE001  (รวม protobuf VersionError)
        stub = types.ModuleType("tensorflow_decision_forests")
        keras_stub = types.ModuleType("tensorflow_decision_forests.keras")
        stub.keras = keras_stub  # type: ignore[attr-defined]
        sys.modules["tensorflow_decision_forests"] = stub
        sys.modules["tensorflow_decision_forests.keras"] = keras_stub

INPUT_RES = 257
OUTPUT_RES = 17  # (257-1)/16 + 1
HEATMAP_CH = 17  # 17 keypoints
OFFSET_CH = 34   # 2 * 17 (x,y)
FEATURE_DIM = OUTPUT_RES * OUTPUT_RES * (HEATMAP_CH + OFFSET_CH)  # 14739

# ใช้ posenet ที่ tfjs host ไว้ — graph model สำหรับ MobileNetV1 mult 0.75
POSENET_URL = (
    "https://storage.googleapis.com/tfjs-models/savedmodel/"
    "posenet/mobilenet/float/075/model-stride16.json"
)
from .paths import POSENET_DIR as MODELS_DIR


def _download_posenet() -> Path:
    """ดาวน์โหลดไฟล์ tfjs graph model (model.json + shards) ครั้งแรก"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_json = MODELS_DIR / "model.json"
    if model_json.exists():
        return model_json

    base = POSENET_URL.rsplit("/", 1)[0]
    print(f"[posenet] downloading model.json -> {model_json}")
    urllib.request.urlretrieve(POSENET_URL, model_json)

    import json
    manifest = json.loads(model_json.read_text())
    shard_paths: set[str] = set()
    for group in manifest.get("weightsManifest", []):
        for p in group.get("paths", []):
            shard_paths.add(p)
    for shard in shard_paths:
        url = f"{base}/{shard}"
        dst = MODELS_DIR / shard
        print(f"[posenet] downloading {shard}")
        urllib.request.urlretrieve(url, dst)
    return model_json


class PoseNetExtractor:
    """ตัวรัน PoseNet — รับภาพ BGR, คืน feature vector (14739,)"""

    def __init__(self) -> None:
        model_json = _download_posenet()
        _numpy_compat_shim()
        _stub_decision_forests()
        try:
            import tfjs_graph_converter.api as tfjs_api
        except ImportError as e:
            raise RuntimeError(
                "ต้องติดตั้ง tfjs-graph-converter: pip install tfjs-graph-converter"
            ) from e
        graph = tfjs_api.load_graph_model(str(model_json.parent))
        # wrap frozen graph -> callable TF2 function (รับ numpy/tensor, คืน list ของ output)
        self._func = tfjs_api.graph_to_function_v2(graph)
        # มิเรอร์ภาพแนวนอน — เว็บ Teachable Machine เปิดกล้องแบบ selfie (flip)
        # โมเดลจึงถูกเทรนบนภาพมิเรอร์ ค่าเริ่มต้นจึง True เพื่อให้ตรงกับตอนเทรน
        self.flip: bool = True

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        เลียนแบบ posenet.padAndResizeTo ของ TFJS:
        เติมขอบ(zero-pad) ด้านสั้นให้เป็นสี่เหลี่ยมจัตุรัส -> resize 257 -> RGB -> [-1,1]
        (ไม่ใช่ center-crop เพราะ TM ใช้ pad-and-resize คงสัดส่วนไว้ทั้งภาพ)
        """
        if self.flip:
            frame_bgr = frame_bgr[:, ::-1, :]
        h, w = frame_bgr.shape[:2]
        side = max(h, w)
        padded = np.zeros((side, side, 3), dtype=frame_bgr.dtype)
        y0 = (side - h) // 2
        x0 = (side - w) // 2
        padded[y0:y0 + h, x0:x0 + w] = frame_bgr
        resized = cv2.resize(padded, (INPUT_RES, INPUT_RES))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        normed = (rgb / 127.5) - 1.0
        return normed[None, ...]

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        """คืน feature vector (14739,) พร้อมโยนเข้า classifier"""
        x = self.preprocess(frame_bgr)
        outputs = self._func(tf.constant(x))
        # PoseNet graph model มี 4 outputs: heatmap, offset, displacement_fwd, bwd
        # graph_to_function_v2 คืนเป็น list/tuple ของ tensor — เลือกจาก shape
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        heatmap = None
        offset = None
        for t in outputs:
            arr = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
            if arr.ndim != 4:
                continue
            ch = arr.shape[-1]
            if ch == HEATMAP_CH and heatmap is None:
                heatmap = arr
            elif ch == OFFSET_CH and offset is None:
                offset = arr
        if heatmap is None or offset is None:
            raise RuntimeError("ไม่พบ output heatmap/offset จาก PoseNet")
        # sigmoid ของ heatmap (ตรงกับ base_model.predict ของ tfjs-posenet)
        heatmap = 1.0 / (1.0 + np.exp(-heatmap))
        # *** สำคัญ: ต้อง concat ตามแกน channel ก่อน flatten ***
        # TM ทำ tf.concat([heatmapScores, offsets], axis=channel) แล้วค่อย dataSync
        # => ที่แต่ละจุด (h,w) เรียง [heatmap17, offset34] ติดกัน
        combined = np.concatenate([heatmap, offset], axis=-1)  # (1,17,17,51)
        feature = combined.reshape(-1).astype(np.float32)
        assert feature.shape[0] == FEATURE_DIM, feature.shape
        return feature


if __name__ == "__main__":
    # ทดสอบเร็ว ๆ
    extractor = PoseNetExtractor()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    vec = extractor.extract(dummy)
    print("feature shape:", vec.shape, "min:", vec.min(), "max:", vec.max())
