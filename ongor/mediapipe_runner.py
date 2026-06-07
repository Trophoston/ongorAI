"""
BlazePose extractor — รันโมเดล MediaPipe Pose ผ่าน TFLite "ตรง ๆ"
โดยไม่ต้องลงแพ็กเกจ `mediapipe` (ซึ่งไม่มี wheel สำหรับ Linux aarch64 เช่นบน
Arduino Uno Q) ต้องการแค่ tflite-runtime + opencv + numpy

หลักการ (2 รอบ + ROI tracking — เลียนแบบสเตจ detection ของ MediaPipe):
  รอบ 1: ป้อนทั้งเฟรมเข้าโมเดล pose_landmark เพื่อ "หาตำแหน่งคนคร่าวๆ"
  รอบ 2: crop รอบตัวคนแล้วรันซ้ำ -> landmark แม่นแม้คนตัวเล็ก/ไม่อยู่กลางเฟรม
  จากนั้นจำกรอบไว้ (ROI tracking) เฟรมถัดไปรันแค่รอบเดียวบนกรอบเดิม (เร็วขึ้น)
คืน 33 จุด (x,y,z,visibility) พิกัด "ภาพ-normalized [0,1]" ให้ตรงกับที่
`mp.solutions.pose` เคยให้ แล้ว normalize_keypoints() ทำเป็น feature 132 มิติ
-> classifier_mp.tflite ที่เทรนไว้ใช้ได้เลยไม่ต้องเทรนใหม่

*** preprocessing/normalize ต้องตรงกับตอนสร้าง CSV (train_mediapipe.py) เป๊ะ ***
  - ไม่ flip (dataset_to_csv.py อ่านรูปตามจริง) -> default flip=False
  - ความแม่นเทียบ mp.solutions: mean error x,y ~0.003-0.01 (ภาพ-normalized)
    แม้คนตัวเล็ก ~25% ของเฟรม (เพราะรอบ 2 crop ซูมเข้าหาคนก่อน)
"""
from __future__ import annotations

import json
import os
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
        min_tracking_confidence: float = 0.5,  # คงไว้เพื่อความเข้ากันได้ (ROI tracking ใช้ภายใน)
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

        self.presence_threshold = float(
            os.getenv("ONGOR_POSE_PRESENCE", str(min_detection_confidence))
        )
        self.roi_margin = float(os.getenv("ONGOR_POSE_ROI_MARGIN", "0.85"))
        self.search_when_lost = os.getenv("ONGOR_POSE_SEARCH", "1").lower() not in (
            "0", "false", "no", "off"
        )
        self.reacquire_interval = max(
            1, int(os.getenv("ONGOR_POSE_REACQUIRE_INTERVAL", "3"))
        )
        self._lost_frames = 0
        self._roi: tuple[int, int, int, int] | None = None  # กรอบคนเฟรมก่อน (ROI tracking)
        # default False เพื่อให้ตรงกับ dataset_to_csv.py (ซึ่งไม่ flip)
        self.flip: bool = False

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def _run_region(self, rgb, box, gate=True):
        """
        รันโมเดล landmark บน sub-region (x0,y0,x1,y1 พิกเซล) ของ rgb
        คืน (landmarks(33,4) ในพิกัด "ภาพเต็ม-normalized [0,1]", presence) หรือ (None, presence)

        gate=True  -> ถ้า presence ต่ำกว่าเกณฑ์ ถือว่าไม่เจอคน (คืน None)
        gate=False -> คืน landmark เสมอ (ใช้ตอน "หาตำแหน่งคร่าวๆ" จากทั้งเฟรม
                      ซึ่งคนตัวเล็ก presence จะต่ำ แต่ยังพอบอกตำแหน่งได้)
        """
        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = box
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(w, int(x1)); y1 = min(h, int(y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None, 0.0
        crop = rgb[y0:y1, x0:x1]
        inp, s, px, py = _letterbox(crop, _INPUT_SIZE)
        self._interp.set_tensor(self._in["index"], (inp.astype(np.float32) / 255.0)[None])
        self._interp.invoke()

        presence = 1.0
        if self._out_presence is not None:
            presence = self._sigmoid(float(self._interp.get_tensor(self._out_presence)[0, 0]))
            if gate and presence <= self.presence_threshold:
                return None, presence

        out = self._interp.get_tensor(self._out_landmarks)[0].reshape(39, 5)[:N_LANDMARKS]
        lm = np.empty((N_LANDMARKS, 4), dtype=np.float32)
        # พิกัดใน crop (256-space) -> กลับไปพิกัด "ภาพเต็ม-normalized [0,1]"
        lm[:, 0] = ((out[:, 0] - px) / s + x0) / w        # x
        lm[:, 1] = ((out[:, 1] - py) / s + y0) / h        # y
        lm[:, 2] = out[:, 2] / s / w                      # z (สเกลเดียวกับ x)
        lm[:, 3] = 1.0 / (1.0 + np.exp(-out[:, 3]))       # visibility
        return lm, presence

    def _bbox_from(self, lm, w, h):
        """หากรอบจัตุรัสรอบตัวคนจาก landmarks (เฉพาะจุดที่เห็นชัด) + ขยาย margin"""
        vis = lm[lm[:, 3] > 0.3][:, :2]
        if len(vis) < 4:
            vis = lm[:, :2]
        xs, ys = vis[:, 0] * w, vis[:, 1] * h
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = max(x1 - x0, y1 - y0) * (1.0 + self.roi_margin)
        side = max(side, 64)  # อย่าให้กรอบเล็กเกินไป
        return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)

    def _search_boxes(self, w: int, h: int) -> list[tuple[int, int, int, int]]:
        """
        กล่องค้นหาเมื่อ tracking หลุด: ทั้งเฟรม + crop ซ้าย/กลาง/ขวา หรือ บน/กลาง/ล่าง
        ช่วยเคสคนอยู่ชิดขอบเฟรม โดยไม่ต้องพึ่ง pose_detection.tflite
        """
        boxes: list[tuple[int, int, int, int]] = [(0, 0, w, h)]
        if not self.search_when_lost:
            return boxes

        if w >= h:
            side = h
            xs = (0, max(0, (w - side) // 2), max(0, w - side))
            boxes.extend((x, 0, x + side, h) for x in xs)
        else:
            side = w
            ys = (0, max(0, (h - side) // 2), max(0, h - side))
            boxes.extend((0, y, w, y + side) for y in ys)

        uniq: list[tuple[int, int, int, int]] = []
        seen = set()
        for b in boxes:
            if b not in seen:
                seen.add(b)
                uniq.append(b)
        return uniq

    def _acquire(self, rgb: np.ndarray) -> np.ndarray | None:
        """หา ROI ใหม่จากหลาย candidate crop แล้วเลือก landmark ที่ดีที่สุด"""
        h, w = rgb.shape[:2]
        best_lm = None
        best_score = -1.0

        for box in self._search_boxes(w, h):
            # หาตำแหน่งคร่าวๆ ใน candidate (ไม่ gate — คนไกล/เล็ก presence อาจต่ำ)
            rough, _ = self._run_region(rgb, box, gate=False)
            if rough is None:
                continue
            # crop รอบตัวคนจาก rough แล้วรันซ้ำแบบ gate จริง
            lm, presence = self._run_region(rgb, self._bbox_from(rough, w, h))
            if lm is None:
                continue
            score = presence + 0.15 * float(np.mean(lm[:, 3]))
            if score > best_score:
                best_score = score
                best_lm = lm

        return best_lm

    def process(self, frame_bgr: np.ndarray) -> PoseResult:
        """
        ประมวลผล 1 เฟรม -> PoseResult
        ใช้ ROI tracking: ปกติรันบนกรอบคนเฟรมก่อน (1 ครั้ง) เพื่อความเร็ว
        ถ้าหาไม่เจอ/หลุดกรอบ ค่อย "หาใหม่" จากทั้งเฟรมแล้ว crop ซ้ำ (2 ครั้ง)
        ทำให้แม่นแม้คนตัวเล็ก/ไม่อยู่กลางเฟรม (เทียบเท่าสเตจ detection ของ mediapipe)
        """
        if self.flip:
            frame_bgr = cv2.flip(frame_bgr, 1)
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        lm = None
        # 1) ลองตามกรอบเดิมก่อน (เร็ว: รันครั้งเดียว)
        if self._roi is not None:
            lm, _ = self._run_region(rgb, self._roi)
        # 2) หาไม่เจอ -> หาใหม่จากทั้งเฟรม แล้ว crop รอบตัวคนรันซ้ำให้แม่น
        if lm is None:
            should_acquire = self._lost_frames % self.reacquire_interval == 0
            lm = self._acquire(rgb) if should_acquire else None
            if lm is None:
                self._lost_frames += 1
                self._roi = None
                return PoseResult(keypoints=None, landmarks=None, frame=frame_bgr)

        self._lost_frames = 0
        self._roi = tuple(int(v) for v in self._bbox_from(lm, w, h))  # จำไว้ใช้เฟรมถัดไป
        kp = normalize_keypoints(lm.flatten())
        return PoseResult(keypoints=kp, landmarks=lm, frame=frame_bgr)

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """ทางลัด: คืนเฉพาะ normalized keypoints (132,) หรือ None"""
        return self.process(frame_bgr).keypoints

    def close(self) -> None:
        self._interp = None
        self._roi = None
        self._lost_frames = 0

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
