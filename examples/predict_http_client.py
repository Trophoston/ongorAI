"""
ตัวอย่าง: เรียก API /predict ผ่าน HTTP (จับภาพจากกล้องแล้วส่งไปให้เซิร์ฟเวอร์ทำนาย)
ใช้เมื่อโค้ดของคุณอยู่คนละเครื่อง/คนละภาษากับตัว API

รันเซิร์ฟเวอร์ก่อน:
    .venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

แล้วรันไฟล์นี้:
    .venv/bin/python examples/predict_http_client.py --url http://localhost:8000
"""
from __future__ import annotations

import argparse

import cv2
import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000", help="ที่อยู่ API")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"เปิดกล้อง index {args.camera} ไม่ได้")

    endpoint = args.url.rstrip("/") + "/predict"
    print(f"ส่งภาพไปที่ {endpoint} — Ctrl+C เพื่อหยุด")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            # เข้ารหัสเฟรมเป็น JPEG แล้วส่งเป็น multipart
            _, buf = cv2.imencode(".jpg", frame)
            files = {"image": ("frame.jpg", buf.tobytes(), "image/jpeg")}
            r = requests.post(endpoint, files=files, timeout=5)
            data = r.json()
            pred = data.get("prediction")
            if data.get("confirmed"):
                print(f"[ยืนยันท่า] {data['confirmed']}")
            elif pred:
                print(f"  {pred['label']:24s} {pred['confidence']:5.2f}  {pred['thai']}", end="\r")
            else:
                print("  ไม่พบท่า", end="\r")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()
