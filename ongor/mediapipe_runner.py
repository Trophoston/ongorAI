"""
MediaPipe Pose extractor — แทนที่ posenet_runner.py
เบากว่ามาก ไม่มีปัญหา protobuf/tfjs

หลักสำคัญ: preprocessing ตอน inference ต้องตรงกับตอนสร้าง CSV เป๊ะ
  - dataset_to_csv.py อ่านรูป "ตามจริง" ไม่ flip
  - ดังนั้น default flip=False (ถ้า flip ไม่ตรงกัน ท่าซ้าย/ขวาจะสลับ!)

ใช้งาน:
    ext = MediaPipeExtractor()
    res = ext.process(frame_bgr)
    if res.keypoints is not None:
        feature = res.keypoints          # (132,) normalized แล้ว
        landmarks = res.landmarks        # ไว้วาด skeleton
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

N_LANDMARKS = 33
FEATURE_DIM = N_LANDMARKS * 4  # 132

from .paths import LABEL_MAP as LABEL_MAP_PATH

# index ของ landmark สำคัญ (MediaPipe Pose)
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24

_mp_pose = mp.solutions.pose


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


def landmarks_to_raw(results) -> np.ndarray | None:
    """ดึง keypoints ดิบ (132,) ก่อน normalize — เหมือนที่เก็บลง CSV"""
    if not results.pose_landmarks:
        return None
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility]
         for lm in results.pose_landmarks.landmark],
        dtype=np.float32,
    ).flatten()


@dataclass
class PoseResult:
    keypoints: np.ndarray | None   # (132,) normalized — โยนเข้า classifier ได้เลย
    landmarks: object | None       # results.pose_landmarks (ไว้วาด skeleton)
    frame: np.ndarray              # เฟรมที่ใช้ประมวลผลจริง (หลัง flip ถ้ามี)


class MediaPipeExtractor:
    """ดึง keypoints จากเฟรม BGR — ประมวลผล MediaPipe ครั้งเดียวต่อเฟรม"""

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        self._pose = _mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            static_image_mode=static_image_mode,
        )
        # default False เพื่อให้ตรงกับ dataset_to_csv.py (ซึ่งไม่ flip)
        self.flip: bool = False

    def process(self, frame_bgr: np.ndarray) -> PoseResult:
        """ประมวลผลเฟรม -> PoseResult (keypoints + landmarks ในครั้งเดียว)"""
        if self.flip:
            frame_bgr = cv2.flip(frame_bgr, 1)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        raw = landmarks_to_raw(results)
        kp = normalize_keypoints(raw) if raw is not None else None
        return PoseResult(keypoints=kp, landmarks=results.pose_landmarks, frame=frame_bgr)

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """ทางลัด: คืนเฉพาะ normalized keypoints (132,) หรือ None"""
        return self.process(frame_bgr).keypoints

    def close(self) -> None:
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
