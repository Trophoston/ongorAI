"""
ตัวอย่าง: อ่านภาพจากกล้อง USB แล้วทำนายท่าทางแบบเรียลไทม์
ใช้ PoseEngine โดยตรง (ไม่ผ่าน HTTP) — เบาและไวที่สุด เหมาะกับบนบอร์ด

รัน:
    .venv/bin/python examples/predict_webcam.py
    .venv/bin/python examples/predict_webcam.py --camera 0 --show

กด q เพื่อออก (เฉพาะตอนใช้ --show)
"""
from __future__ import annotations

import argparse
import time

import cv2

from ongor.pose_engine import PoseEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="index ของกล้อง (ปกติ 0)")
    ap.add_argument("--show", action="store_true", help="เปิดหน้าต่างแสดงผล (ต้องมีจอ)")
    ap.add_argument("--flip", action="store_true", help="กลับซ้าย-ขวา (โหมดกระจก)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"เปิดกล้อง index {args.camera} ไม่ได้")

    engine = PoseEngine(flip=args.flip)
    print("เริ่มทำงาน — Ctrl+C เพื่อหยุด")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            # pred = ผลทำนายดิบของเฟรมนี้ (label, confidence, thai)
            res, pred = engine.process(frame)

            # confirmed = label เมื่อ "ค้างท่าไว้นานพอ+มั่นใจพอ" เท่านั้น (เหมาะกับสั่งงาน)
            label = pred.label if pred else None
            conf = pred.confidence if pred else 0.0
            confirmed = engine.stabilizer.update(label, conf)

            if confirmed:
                print(f"[ยืนยันท่า] {confirmed}")
            elif pred:
                print(f"  {pred.label:24s} {pred.confidence:5.2f}  {pred.thai}", end="\r")

            if args.show:
                if res.landmarks is not None:
                    from mediapipe.python.solutions import drawing_utils, pose as mp_pose
                    drawing_utils.draw_landmarks(
                        frame, res.landmarks, mp_pose.POSE_CONNECTIONS
                    )
                txt = f"{pred.label} {pred.confidence:.2f}" if pred else "no pose"
                cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)
                cv2.imshow("Ong-Or Pose", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        engine.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
