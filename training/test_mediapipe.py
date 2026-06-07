"""
ทดสอบ pipeline ใหม่: กล้อง -> MediaPipe -> classifier_mp.tflite
รันหลัง dataset_to_csv.py + train_mediapipe.py เสร็จแล้ว

python test_mediapipe.py              # กล้องสด
python test_mediapipe.py --flip       # มิเรอร์ (ถ้าซ้าย/ขวาสลับให้ลองอันนี้)
python test_mediapipe.py --headless   # ไม่เปิดหน้าต่าง พิมพ์ผลทาง terminal
python test_mediapipe.py --selftest   # โหลด+เช็ค shape ไม่ใช้กล้อง
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ongor.mediapipe_runner import MediaPipeExtractor, load_label_map
from ongor.paths import CLASSIFIER_MP_TFLITE as TFLITE_PATH

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def load_interpreter():
    """โหลด TFLite (ใช้ tflite_runtime ถ้ามี ไม่งั้น fallback tensorflow)"""
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        return Interpreter(model_path=str(TFLITE_PATH))
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=str(TFLITE_PATH))


class Classifier:
    """ห่อ TFLite interpreter ให้เรียกง่าย"""

    def __init__(self) -> None:
        if not TFLITE_PATH.exists():
            raise FileNotFoundError(
                f"ไม่พบ {TFLITE_PATH} — รัน train_mediapipe.py ก่อน"
            )
        self.interp = load_interpreter()
        self.interp.allocate_tensors()
        self.in_d = self.interp.get_input_details()[0]
        self.out_d = self.interp.get_output_details()[0]
        self.labels = load_label_map()

    def predict(self, feat: np.ndarray) -> np.ndarray:
        x = feat[None, :].astype(np.float32)
        self.interp.set_tensor(self.in_d["index"], x)
        self.interp.invoke()
        return self.interp.get_tensor(self.out_d["index"])[0]

    def label(self, idx: int) -> str:
        return self.labels.get(idx, str(idx))


def run_selftest() -> int:
    clf = Classifier()
    n_in = int(clf.in_d["shape"][-1])
    n_out = int(clf.out_d["shape"][-1])
    x = np.random.randn(n_in).astype(np.float32)
    probs = clf.predict(x)
    print(f"[selftest] input_dim  = {n_in} (ควรเป็น 132)")
    print(f"[selftest] output_dim = {n_out}")
    print(f"[selftest] sum(prob)  = {probs.sum():.4f} (ควรใกล้ 1.0)")
    print(f"[selftest] labels     = {[clf.label(i) for i in range(n_out)]}")
    ok = n_in == 132 and abs(probs.sum() - 1.0) < 1e-3
    print("[selftest]", "ผ่าน ✅" if ok else "ไม่ผ่าน ❌")
    return 0 if ok else 1


def run_camera(camera: int, headless: bool, flip: bool) -> int:
    clf = Classifier()
    extractor = MediaPipeExtractor()
    extractor.flip = flip

    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"[run] เริ่ม (flip={flip}) — ทำท่าหน้ากล้อง (q/ESC ออก)")
    top_label, top_conf = "...", 0.0
    fps = 0.0
    t_prev = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # ประมวลผลครั้งเดียว — ได้ทั้ง keypoints และ landmarks
            res = extractor.process(frame)
            display = res.frame.copy()   # เฟรมเดียวกับที่ป้อนเข้าโมเดล (skeleton ตรงเสมอ)

            if res.keypoints is not None:
                probs = clf.predict(res.keypoints)
                idx = int(np.argmax(probs))
                top_label = clf.label(idx)
                top_conf = float(probs[idx])

                t_now = time.time()
                fps = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
                t_prev = t_now

                if headless:
                    top3 = sorted(enumerate(probs), key=lambda kv: -kv[1])[:3]
                    parts = [f"{clf.label(i)} {p * 100:.0f}%" for i, p in top3]
                    print(f"[{fps:4.1f}fps]  " + "  |  ".join(parts))

            if not headless:
                if res.landmarks is not None:
                    mp_draw.draw_landmarks(
                        display, res.landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec((0, 255, 0), 2, 2),
                        mp_draw.DrawingSpec((255, 255, 255), 1, 1),
                    )
                color = (0, 255, 0) if top_conf >= 0.8 else \
                        (0, 200, 255) if top_conf >= 0.5 else (0, 0, 255)
                cv2.putText(display, f"{top_label}  {top_conf * 100:.0f}%",
                            (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
                cv2.putText(display, f"{fps:.1f} fps", (12, display.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.imshow("test_mediapipe", display)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--flip", action="store_true",
                    help="มิเรอร์เฟรม (default ไม่มิเรอร์ ให้ตรงกับตอนสร้าง CSV)")
    args = ap.parse_args()
    if args.selftest:
        return run_selftest()
    return run_camera(args.camera, args.headless, args.flip)


if __name__ == "__main__":
    sys.exit(main())
