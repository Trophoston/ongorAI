"""
BlazePose extractor — รันโมเดล MediaPipe Pose ผ่าน TFLite "ตรง ๆ"
โดยไม่ต้องลงแพ็กเกจ `mediapipe` (ซึ่งไม่มี wheel สำหรับ Linux aarch64 เช่นบน
Arduino Uno Q) ต้องการแค่ tflite-runtime + opencv + numpy

หลักการ: ป้อนทั้งเฟรม (letterbox เป็นจัตุรัส 256x256) เข้าโมเดล pose_landmark
สเตจเดียว แล้วคืน 33 จุด (x,y,z,visibility) ในพิกัด "ภาพ-normalized [0,1]" ให้ตรง
กับที่ `mp.solutions.pose` เคยให้ จากนั้น normalize_keypoints() ทำให้เป็น feature
132 มิติเหมือนเดิม -> classifier_mp.tflite ที่เทรนไว้ใช้ได้เลยไม่ต้องเทรนใหม่

*** preprocessing/normalize ต้องตรงกับตอนสร้าง CSV (train_mediapipe.py) เป๊ะ ***
  - ไม่ flip (dataset_to_csv.py อ่านรูปตามจริง) -> default flip=False
  - ความแม่นเทียบ mp.solutions: mean error ของ x,y ~0.009 (ภาพ-normalized)
    และผลทำนายของ classifier ออกมา label เดียวกัน

ข้อจำกัดของวิธีสเตจเดียว: ทำงานดีเมื่อ "คนยืนตรงเต็มตัวอยู่กลางเฟรม" (ซึ่งเป็น
สภาพการใช้งานของเกมนี้อยู่แล้ว) เพราะข้ามสเตจ detection/ROI ของ MediaPipe ไป
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .paths import BLAZEPOSE_LANDMARK_TFLITE
from .paths import LABEL_MAP as LABEL_MAP_PATH
from .tflite_util import load_interpreter

N_LANDMARKS = 33
FEATURE_DIM = N_LANDMARKS * 4  # 132
_INPUT_SIZE = 256              # ขนาด input ของ pose_landmark_full.tflite

# index ของ landmark สำคัญ (MediaPipe Pose)
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24

# เส้นเชื่อมโครงร่าง (subset มาตรฐานของ BlazePose) — ไว้วาด skeleton ด้วย cv2
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),       # แขน + ไหล่
    (11, 23), (12, 24), (23, 24),                           # ลำตัว
    (23, 25), (25, 27), (24, 26), (26, 28),                 # ขา
    (27, 31), (28, 32),                                     # เท้า
    (0, 11), (0, 12),                                       # คอ-ไหล่ (คร่าว ๆ)
)


def load_label_map(path: Path | str | None = None) -> dict[int, str]:
    """โหลด {index: label} จาก label_map.json (ผลผลิตของ train_mediapipe.py)"""
    p = Path(path) if path else LABEL_MAP_PATH
    try:
        data = json.loads(p.read_text())
        return {int(k): str(v) for k, v in data.items()}
    except Exception as e:  # noqa: BLE001
        print(f"[mediapipe_runner] โหลด label_map ไม่ได้: {e}")
        return {}


def normalize_keypoints(kp: np.ndarray) -> np.ndarray:
    """
    normalize ให้ตรงกับ normalize() ใน train_mediapipe.py เป๊ะ:
      - เลื่อน origin ไปที่จุดกึ่งกลางสะโพก
      - หาร scale ด้วยระยะ ไหล่กลาง -> สะโพกกลาง (torso height)
      - x, y normalize / z หาร scale / visibility คงไว้
    *** ถ้าแก้สูตรนี้ ต้องแก้ใน train_mediapipe.py ให้ตรงกันด้วย ***
    """
    pts = kp.reshape(N_LANDMARKS, 4).copy()
    hip_x = (pts[L_HIP, 0] + pts[R_HIP, 0]) / 2
    hip_y = (pts[L_HIP, 1] + pts[R_HIP, 1]) / 2
    sh_x = (pts[L_SHOULDER, 0] + pts[R_SHOULDER, 0]) / 2
    sh_y = (pts[L_SHOULDER, 1] + pts[R_SHOULDER, 1]) / 2
    scale = float(np.sqrt((sh_x - hip_x) ** 2 + (sh_y - hip_y) ** 2)) + 1e-6

    pts[:, 0] = (pts[:, 0] - hip_x) / scale
    pts[:, 1] = (pts[:, 1] - hip_y) / scale
    pts[:, 2] = pts[:, 2] / scale
    return pts.flatten().astype(np.float32)


def _letterbox(img: np.ndarray, size: int = _INPUT_SIZE):
    """ย่อ+เติมขอบให้เป็นจัตุรัส size x size (คงสัดส่วน) คืน (canvas, scale, pad_x, pad_y)"""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=img.dtype)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, s, px, py


@dataclass
class PoseResult:
    keypoints: np.ndarray | None   # (132,) normalized — โยนเข้า classifier ได้เลย
    landmarks: np.ndarray | None   # (33,4) image-normalized [x,y,z,visibility] ไว้วาด skeleton
    frame: np.ndarray              # เฟรมที่ใช้ประมวลผลจริง (หลัง flip ถ้ามี)


class MediaPipeExtractor:
    """
    ดึง keypoints จากเฟรม BGR ด้วยโมเดล BlazePose (TFLite) — ไม่พึ่งแพ็กเกจ mediapipe

    คง interface เดิมไว้ (ชื่อคลาส/เมธอด/พารามิเตอร์) เพื่อให้โค้ดส่วนอื่นใช้แทนกันได้
    """

    def __init__(
        self,
        model_complexity: int = 1,            # คงไว้เพื่อความเข้ากันได้ (ตอนนี้ใช้รุ่น full เสมอ)
        min_detection_confidence: float = 0.6,  # presence ต่ำกว่านี้ = ถือว่าไม่เจอคน
        min_tracking_confidence: float = 0.5,  # คงไว้เพื่อความเข้ากันได้ (ไม่มี tracking ในโหมดนี้)
        static_image_mode: bool = False,       # คงไว้เพื่อความเข้ากันได้
        model_path: Path | str | None = None,
    ) -> None:
        path = Path(model_path) if model_path else BLAZEPOSE_LANDMARK_TFLITE
        if not path.exists():
            raise FileNotFoundError(
                f"ไม่พบโมเดล BlazePose: {path}\n"
                f"ต้องมีไฟล์ pose_landmark_full.tflite (มากับ repo ใน models/blazepose/)"
            )
        self._interp = load_interpreter(path)
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        # หา output ตาม "รูปร่าง" (ลำดับ index อาจต่างกันในแต่ละ backend)
        self._out_landmarks = None  # [1,195] = 39 จุด x 5
        self._out_presence = None   # [1,1]   = ความมั่นใจว่ามีคน (logit)
        for d in self._interp.get_output_details():
            shape = list(d["shape"])
            if shape[-1] == 195:
                self._out_landmarks = d["index"]
            elif shape == [1, 1]:
                self._out_presence = d["index"]
        if self._out_landmarks is None:
            raise RuntimeError("โมเดลไม่มี output landmark (shape ...,195) — ไฟล์โมเดลผิดรุ่น?")

        self.presence_threshold = float(min_detection_confidence)
        # default False เพื่อให้ตรงกับ dataset_to_csv.py (ซึ่งไม่ flip)
        self.flip: bool = False

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def process(self, frame_bgr: np.ndarray) -> PoseResult:
        """ประมวลผลเฟรม -> PoseResult (keypoints + landmarks ในครั้งเดียว)"""
        if self.flip:
            frame_bgr = cv2.flip(frame_bgr, 1)
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp, s, px, py = _letterbox(rgb, _INPUT_SIZE)
        x = (inp.astype(np.float32) / 255.0)[None]

        self._interp.set_tensor(self._in["index"], x)
        self._interp.invoke()

        # เช็คว่ามีคนไหมก่อน (กันภาพเปล่าให้ landmark มั่ว)
        if self._out_presence is not None:
            presence = self._sigmoid(float(self._interp.get_tensor(self._out_presence)[0, 0]))
            if presence <= self.presence_threshold:
                return PoseResult(keypoints=None, landmarks=None, frame=frame_bgr)

        out = self._interp.get_tensor(self._out_landmarks)[0].reshape(39, 5)[:N_LANDMARKS]
        # map พิกัดจาก 256-space กลับไปเป็น "ภาพ-normalized [0,1]" ให้ตรงกับ mp.solutions
        lm = np.empty((N_LANDMARKS, 4), dtype=np.float32)
        lm[:, 0] = (out[:, 0] - px) / s / w           # x
        lm[:, 1] = (out[:, 1] - py) / s / h           # y
        lm[:, 2] = out[:, 2] / s / w                  # z (สเกลเดียวกับ x)
        lm[:, 3] = 1.0 / (1.0 + np.exp(-out[:, 3]))   # visibility (logit -> [0,1])

        kp = normalize_keypoints(lm.flatten())
        return PoseResult(keypoints=kp, landmarks=lm, frame=frame_bgr)

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """ทางลัด: คืนเฉพาะ normalized keypoints (132,) หรือ None"""
        return self.process(frame_bgr).keypoints

    def close(self) -> None:
        self._interp = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def draw_landmarks(
    frame_bgr: np.ndarray,
    landmarks: np.ndarray | None,
    color=(0, 255, 0),
    radius: int = 3,
) -> np.ndarray:
    """วาด skeleton จาก landmarks (33,4) image-normalized ลงบนเฟรม BGR ด้วย cv2"""
    if landmarks is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _, _ in landmarks]
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(frame_bgr, p, radius, (0, 0, 255), -1)
    return frame_bgr
