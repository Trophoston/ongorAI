"""
PoseEngine — API ง่าย ๆ สำหรับเรียกใช้บนบอร์ด
รวม MediaPipe + classifier_mp + ตัวจับท่าให้เสถียร (stabilizer) ไว้ในที่เดียว

ใช้งานพื้นฐาน (บรรทัดเดียวต่อเฟรม):
    from ongor import PoseEngine
    engine = PoseEngine()
    pred = engine.predict(frame_bgr)        # -> Prediction | None
    if pred: print(pred.label, pred.confidence)

ใช้แบบจับท่า "ค้างไว้ถึงนับ" (เหมาะกับเกม):
    confirmed = engine.read_confirmed(frame_bgr)   # -> label str | None
    if confirmed: print("ยืนยันท่า:", confirmed)
"""
from __future__ import annotations

import time

import numpy as np

from .classifier_mp import MediaPipeClassifier
from .labels import Prediction
from .mediapipe_runner import MediaPipeExtractor, PoseResult


class PoseStabilizer:
    """
    กรองผลทำนายให้ "ยืนยัน" เฉพาะท่าที่ค้างไว้นานพอและมั่นใจพอ
    + บังคับให้กลับมา idle/นิ่งก่อน ถึงจะยืนยันท่าถัดไปได้ (กันนับซ้ำ)

    update() คืน label เมื่อ "เพิ่งยืนยัน" ท่าใหม่ มิฉะนั้นคืน None
    """

    def __init__(
        self,
        hold_time: float = 0.6,        # ต้องค้างท่ากี่วินาทีถึงนับ
        conf_threshold: float = 0.85,  # confidence ขั้นต่ำ
        release_time: float = 0.25,    # ต้องนิ่ง/idle นานเท่าไรถึง re-arm
        idle_label: str = "idle",
    ) -> None:
        self.hold_time = hold_time
        self.conf_threshold = conf_threshold
        self.release_time = release_time
        self.idle_label = idle_label
        self._cand: str | None = None
        self._cand_since: float = 0.0
        self._release_since: float | None = None
        self._armed: bool = True  # เริ่มมาพร้อมยืนยันท่าแรกได้เลย

    def reset(self) -> None:
        self._cand = None
        self._release_since = None
        self._armed = True

    def hold_progress(self, now: float | None = None) -> float:
        """ความคืบหน้าการค้างท่าปัจจุบัน 0..1 (ไว้วาด progress ring)"""
        if self._cand is None or not self._armed:
            return 0.0
        now = now if now is not None else time.time()
        return min(1.0, (now - self._cand_since) / self.hold_time)

    def update(
        self, label: str | None, confidence: float, now: float | None = None
    ) -> str | None:
        now = now if now is not None else time.time()
        strong = label if (label and confidence >= self.conf_threshold) else None

        # อยู่ในภาวะ "ปล่อย" (ไม่เจอท่า/มั่นใจน้อย/เป็น idle)
        if strong is None or strong == self.idle_label:
            if self._release_since is None:
                self._release_since = now
            if not self._armed and (now - self._release_since) >= self.release_time:
                self._armed = True
            self._cand = None
            return None

        # เจอท่าจริง มั่นใจพอ
        self._release_since = None
        if strong != self._cand:
            self._cand = strong
            self._cand_since = now
        if self._armed and (now - self._cand_since) >= self.hold_time:
            self._armed = False
            confirmed = strong
            self._cand = None
            return confirmed
        return None


class PoseEngine:
    """
    รวมทุกอย่างของฝั่งมองเห็นไว้ในคลาสเดียว — เรียกง่ายบนบอร์ด

    predict(frame)         -> Prediction | None        (ผลดิบต่อเฟรม)
    process(frame)         -> (PoseResult, Prediction)  (เอา landmarks ไปวาดด้วย)
    read_confirmed(frame)  -> label str | None          (ยืนยันท่าค้างไว้)
    """

    def __init__(
        self,
        flip: bool = False,
        model_complexity: int = 1,
        hold_time: float = 0.6,
        conf_threshold: float = 0.85,
    ) -> None:
        self.extractor = MediaPipeExtractor(model_complexity=model_complexity)
        self.extractor.flip = flip
        self.classifier = MediaPipeClassifier()
        self.stabilizer = PoseStabilizer(
            hold_time=hold_time, conf_threshold=conf_threshold
        )
        self.last_result: PoseResult | None = None
        self.last_pred: Prediction | None = None

    def process(self, frame_bgr: np.ndarray) -> tuple[PoseResult, Prediction | None]:
        """ประมวลผล 1 เฟรม คืนทั้ง PoseResult (มี landmarks) และ Prediction"""
        res, predictions = self.process_topk(frame_bgr, k=1)
        pred = predictions[0] if predictions else None
        return res, pred

    def process_topk(
        self, frame_bgr: np.ndarray, k: int = 3
    ) -> tuple[PoseResult, list[Prediction]]:
        """ประมวลผลหนึ่งเฟรมและคืนผลเรียงตาม confidence โดย infer classifier ครั้งเดียว"""
        res = self.extractor.process(frame_bgr)
        predictions = (
            self.classifier.predict_topk(res.keypoints, k=k)
            if res.keypoints is not None
            else []
        )
        self.last_result = res
        self.last_pred = predictions[0] if predictions else None
        return res, predictions

    def predict(self, frame_bgr: np.ndarray) -> Prediction | None:
        """ทางลัด: คืนเฉพาะ Prediction (label, confidence, thai)"""
        return self.process(frame_bgr)[1]

    def read_confirmed(
        self, frame_bgr: np.ndarray, now: float | None = None
    ) -> str | None:
        """
        ประมวลผล + ป้อนเข้า stabilizer
        คืน label เมื่อ "เพิ่งยืนยัน" ท่าที่ค้างไว้ มิฉะนั้น None
        """
        _, pred = self.process(frame_bgr)
        label = pred.label if pred else None
        conf = pred.confidence if pred else 0.0
        return self.stabilizer.update(label, conf, now=now)

    def close(self) -> None:
        self.extractor.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
