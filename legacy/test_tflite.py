"""
ทดสอบ classifier.tflite อย่างเดียว — ไม่เกี่ยวกับบอร์ด/เกม
ดูว่า pipeline กล้อง -> PoseNet -> TFLite ทำงานและแยกท่าได้จริงไหม

โหมดการใช้งาน:
  python test_tflite.py                 # ใช้กล้อง โชว์ผลสดบนหน้าจอ
  python test_tflite.py --headless      # ใช้กล้องแต่ไม่เปิดหน้าต่าง พิมพ์ผลทาง terminal
  python test_tflite.py --selftest      # ไม่ใช้กล้อง ตรวจแค่ว่าโหลด tflite + รัน dummy ได้
  python test_tflite.py --image foo.jpg # ทดสอบจากรูปนิ่งรูปเดียว

ปุ่มบนหน้าต่าง:  ESC / q = ออก
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _print_topk(preds, prefix: str = "") -> None:
    parts = [f"{p.thai}({p.label}) {p.confidence * 100:4.1f}%" for p in preds]
    print(prefix + "  |  ".join(parts))


def run_selftest() -> int:
    """ไม่ใช้กล้อง — แค่ยืนยันว่า tflite โหลดได้และ output ถูกต้อง"""
    from ongor.classifier_tflite import TFLitePoseClassifier
    from ongor.labels import LABELS

    print("[selftest] โหลด classifier.tflite ...")
    clf = TFLitePoseClassifier()
    print(f"[selftest] input dim = {clf._in_dim} (ควรเป็น 14739)")

    rng = np.random.default_rng(0)
    feat = rng.standard_normal(14739).astype(np.float32)
    preds = clf.predict_topk(feat, k=len(LABELS))

    n = len(LABELS)
    total = sum(p.confidence for p in preds)
    print(f"[selftest] จำนวนคลาส = {len(preds)} (ควรเป็น {n})")
    print(f"[selftest] sum(prob) = {total:.4f} (ควรใกล้ 1.0)")
    _print_topk(preds[:3], prefix="[selftest] top3: ")

    ok = len(preds) == n and abs(total - 1.0) < 1e-3
    print("[selftest] ผล:", "ผ่าน ✅" if ok else "ไม่ผ่าน ❌")
    return 0 if ok else 1


def run_image(path: str, flip: bool = True) -> int:
    import cv2
    from ongor.classifier_tflite import TFLitePoseClassifier
    from ongor.posenet_runner import PoseNetExtractor

    frame = cv2.imread(path)
    if frame is None:
        print(f"[err] เปิดรูป {path} ไม่ได้")
        return 1

    print("[init] โหลด PoseNet + TFLite ...")
    extractor = PoseNetExtractor()
    extractor.flip = flip
    clf = TFLitePoseClassifier()

    feat = extractor.extract(frame)
    preds = clf.predict_topk(feat, k=3)
    _print_topk(preds, prefix=f"[{path}] ")
    return 0


def run_camera(camera: int, headless: bool, infer_every: int, flip: bool = True) -> int:
    import cv2
    from ongor.classifier_tflite import TFLitePoseClassifier
    from ongor.posenet_runner import PoseNetExtractor

    print(f"[init] โหลด PoseNet + TFLite ... (flip/mirror = {flip})")
    extractor = PoseNetExtractor()
    extractor.flip = flip
    clf = TFLitePoseClassifier()

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"[err] เปิดกล้อง {camera} ไม่ได้")
        return 1

    print("[run] เริ่มแล้ว — ทำท่าหน้ากล้อง  (ESC/q เพื่อออก)")
    frame_idx = 0
    preds = None
    t_last = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame_idx += 1
            if frame_idx % max(1, infer_every) == 0:
                t0 = time.time()
                try:
                    feat = extractor.extract(frame)
                    preds = clf.predict_topk(feat, k=3)
                except Exception as e:  # noqa: BLE001
                    print(f"[infer err] {e}")
                    preds = None
                dt = time.time() - t0
                fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)

            if headless:
                if preds is not None and frame_idx % (infer_every * 5) == 0:
                    _print_topk(preds, prefix=f"[{fps:4.1f} infer/s] ")
                continue

            # วาดผลลงเฟรม
            if preds is not None:
                for i, p in enumerate(preds):
                    color = (0, 255, 0) if i == 0 else (200, 200, 200)
                    bar = int(p.confidence * 200)
                    y = 36 + i * 34
                    cv2.rectangle(frame, (12, y - 18), (12 + bar, y - 2), color, -1)
                    cv2.putText(
                        frame, f"{p.label} {p.confidence * 100:4.1f}%",
                        (220, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2, cv2.LINE_AA,
                    )
            cv2.putText(
                frame, f"{fps:4.1f} infer/s", (12, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow("test_tflite — pose", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--infer-every", type=int, default=2)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--image", default=None)
    ap.add_argument("--no-flip", dest="flip", action="store_false",
                    help="ปิดการมิเรอร์ภาพ (ค่าเริ่มต้นมิเรอร์ ให้ตรงกับเว็บ TM)")
    ap.set_defaults(flip=True)
    args = ap.parse_args()

    if args.selftest:
        return run_selftest()
    if args.image:
        return run_image(args.image, flip=args.flip)
    return run_camera(args.camera, args.headless, args.infer_every, flip=args.flip)


if __name__ == "__main__":
    sys.exit(main())
