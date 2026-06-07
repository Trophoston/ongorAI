"""
เก็บ training data จากกล้อง — บันทึก MediaPipe keypoints ลง CSV

วิธีใช้:
  python collect_data.py

ปุ่มควบคุม:
  1-9 / a-z  เริ่ม/หยุดบันทึกท่าที่ map ไว้
  SPACE       หยุดบันทึก (กลับ idle)
  d           ดูจำนวน sample แต่ละคลาส
  q / ESC     ออก

ผลลัพธ์: data/keypoints.csv  (ต่อท้ายทุกครั้งที่รัน)
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ongor.labels import LABELS, GAME_POSES
from ongor.paths import KEYPOINTS_CSV as OUT_CSV

# ---- config ----
CAMERA = 0
FPS_RECORD = 10      # เก็บ sample สูงสุด N เฟรม/วินาที (ลดความซ้ำซ้อน)
MIN_VISIBILITY = 0.5 # ไม่บันทึกถ้า keypoint หลักไม่ชัด

# map ท่ากับปุ่ม (index ตรงกับ LABELS)
KEY_MAP: dict[int, str] = {ord(str(i + 1)): lbl for i, lbl in enumerate(GAME_POSES) if i < 9}
KEY_MAP[ord("0")] = "idle"

# ---- mediapipe ----
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
N_LANDMARKS = 33
FEATURE_DIM = N_LANDMARKS * 4  # x, y, z, visibility


def extract_keypoints(results) -> np.ndarray | None:
    """แปลง MediaPipe results -> float array (132,) หรือ None ถ้าไม่เจอคน"""
    if not results.pose_landmarks:
        return None
    kp = np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
        dtype=np.float32,
    ).flatten()
    # ตรวจ visibility ของ shoulder/hip (landmark 11,12,23,24) ต้องพอมองเห็น
    vis_check = [results.pose_landmarks.landmark[i].visibility for i in [11, 12, 23, 24]]
    if np.mean(vis_check) < MIN_VISIBILITY:
        return None
    return kp


def draw_status(frame, recording_label: str | None, counts: dict[str, int]) -> None:
    h, w = frame.shape[:2]
    if recording_label:
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 180), -1)
        cv2.putText(frame, f"REC: {recording_label}", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "idle — กดปุ่มเพื่อบันทึก", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

    # แสดงจำนวน sample ด้านล่าง
    y = h - (len(LABELS)) * 22 - 10
    for lbl in LABELS:
        n = counts.get(lbl, 0)
        color = (0, 255, 0) if n >= 50 else (0, 200, 255) if n > 0 else (100, 100, 100)
        cv2.putText(frame, f"{lbl}: {n}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        y += 22

    # key guide
    guide = "  ".join(f"{k}:{v}" for k, v in list(KEY_MAP.items())[:5])
    cv2.putText(frame, guide, (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def load_counts() -> dict[str, int]:
    counts: dict[str, int] = {lbl: 0 for lbl in LABELS}
    if OUT_CSV.exists():
        with open(OUT_CSV, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    lbl = row[0]
                    counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def main() -> None:
    OUT_CSV.parent.mkdir(exist_ok=True)
    write_header = not OUT_CSV.exists()

    counts = load_counts()
    recording_label: str | None = None
    last_record_time = 0.0

    cap = cv2.VideoCapture(CAMERA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("=== เก็บ Training Data ===")
    print("ท่าและปุ่ม:")
    for k, v in KEY_MAP.items():
        print(f"  {chr(k)} -> {v}")
    print("SPACE = หยุดบันทึก | d = ดูสถิติ | q/ESC = ออก")
    print(f"บันทึกลง: {OUT_CSV}")

    with open(OUT_CSV, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            header = ["label"] + [
                f"lm{i}_{ax}" for i in range(N_LANDMARKS) for ax in ("x", "y", "z", "vis")
            ]
            writer.writerow(header)

        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        ) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)  # มิเรอร์ selfie
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                # วาด skeleton
                if results.pose_landmarks:
                    mp_draw.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec((0, 255, 0), 2, 2),
                        mp_draw.DrawingSpec((255, 255, 255), 1, 1),
                    )

                # บันทึก sample
                if recording_label:
                    now = time.time()
                    if now - last_record_time >= 1.0 / FPS_RECORD:
                        kp = extract_keypoints(results)
                        if kp is not None:
                            writer.writerow([recording_label] + kp.tolist())
                            csvfile.flush()
                            counts[recording_label] = counts.get(recording_label, 0) + 1
                            last_record_time = now

                draw_status(frame, recording_label, counts)
                cv2.imshow("collect_data — Pose", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                elif key == ord(" "):
                    recording_label = None
                    print("หยุดบันทึก")
                elif key == ord("d"):
                    print("\n--- สถิติ ---")
                    for lbl, n in counts.items():
                        bar = "█" * (n // 5)
                        print(f"  {lbl:28s} {n:4d} {bar}")
                elif key in KEY_MAP:
                    recording_label = KEY_MAP[key]
                    last_record_time = 0.0
                    print(f"บันทึก: {recording_label}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n--- สรุปสุดท้าย ---")
    for lbl, n in counts.items():
        print(f"  {lbl}: {n} samples")


if __name__ == "__main__":
    main()
