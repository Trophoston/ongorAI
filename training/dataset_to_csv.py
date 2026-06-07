"""
อ่านรูปภาพจากโฟลเดอร์ dataset -> รัน MediaPipe Pose -> บันทึก keypoints ลง CSV

โครงสร้าง dataset ที่รองรับ:
  detaset/
    hub_hand_up_Both-samples/   <- label = "hub_hand_up_Both"
    idle-samples/               <- label = "idle"
    prayHand/                   <- label = "prayHand"  (ไม่มี suffix ก็ได้)
    ...

ผลลัพธ์: python_app/data/keypoints.csv

รัน: python dataset_to_csv.py
     python dataset_to_csv.py --dataset ../../detaset   (ระบุ path เอง)
     python dataset_to_csv.py --skip-existing            (ข้ามคลาสที่มีใน CSV แล้ว)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ให้ import ไลบรารี ongor ได้ ไม่ว่ารันจากที่ไหน
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ongor.paths import DATASET_DIR as DEFAULT_DATASET, KEYPOINTS_CSV as OUT_CSV

# ---- mediapipe constants ----
N_LANDMARKS = 33
FEATURE_DIM = N_LANDMARKS * 4          # 132
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def folder_to_label(folder: Path) -> str:
    """hub_hand_up_Both-samples  ->  hub_hand_up_Both"""
    name = folder.name
    if name.endswith("-samples"):
        name = name[: -len("-samples")]
    return name


def load_image_paths(dataset_dir: Path) -> dict[str, list[Path]]:
    """คืน {label: [img_path, ...]} เรียงตาม label"""
    result: dict[str, list[Path]] = {}
    for folder in sorted(dataset_dir.iterdir()):
        if not folder.is_dir():
            continue
        imgs = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS
        )
        if not imgs:
            continue
        label = folder_to_label(folder)
        result.setdefault(label, []).extend(imgs)
    return result


def extract_keypoints(results) -> np.ndarray | None:
    """MediaPipe results -> flat float32 (132,) หรือ None"""
    if not results.pose_landmarks:
        return None
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility]
         for lm in results.pose_landmarks.landmark],
        dtype=np.float32,
    ).flatten()


def process_dataset(
    dataset_dir: Path,
    out_csv: Path,
    skip_existing: bool = False,
    min_visibility: float = 0.3,
) -> None:
    label_map = load_image_paths(dataset_dir)
    if not label_map:
        print(f"[err] ไม่พบโฟลเดอร์รูปภาพใน {dataset_dir}")
        sys.exit(1)

    # ตรวจว่า label ไหนมีใน CSV อยู่แล้ว (ถ้า skip_existing)
    existing_labels: set[str] = set()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()

    if skip_existing and out_csv.exists():
        with open(out_csv, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    existing_labels.add(row[0])
        print(f"[skip] มี {len(existing_labels)} label ใน CSV แล้ว: {existing_labels}")

    header = ["label"] + [
        f"lm{i}_{ax}"
        for i in range(N_LANDMARKS)
        for ax in ("x", "y", "z", "vis")
    ]

    total_saved = 0
    total_skip = 0
    stats: dict[str, dict[str, int]] = {}

    with open(out_csv, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)

        with mp.solutions.pose.Pose(
            static_image_mode=True,      # รูปนิ่ง ไม่ใช่ video
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_visibility,
        ) as pose:

            for label, paths in label_map.items():
                if skip_existing and label in existing_labels:
                    print(f"[skip] {label} — มีใน CSV แล้ว")
                    continue

                saved = skipped = 0
                n = len(paths)
                print(f"\n[{label}] {n} รูป", end="", flush=True)

                for i, img_path in enumerate(paths):
                    if i % 50 == 0:
                        print(f"\r[{label}] {i}/{n} (บันทึก={saved} ข้าม={skipped})", end="", flush=True)

                    frame = cv2.imread(str(img_path))
                    if frame is None:
                        skipped += 1
                        continue

                    # TM เทรนบนภาพมิเรอร์ (selfie) — ใช้ original ก็ได้ ขอแค่ตอน infer ทำเหมือนกัน
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)
                    kp = extract_keypoints(results)

                    if kp is None:
                        skipped += 1
                        continue

                    writer.writerow([label] + kp.tolist())
                    saved += 1
                    total_saved += 1

                csvfile.flush()
                skipped_total = n - saved
                stats[label] = {"saved": saved, "skipped": skipped_total}
                print(f"\r[{label}] เสร็จ ✓  บันทึก={saved}  ข้าม={skipped_total}/{n}          ")

    # สรุป
    print("\n" + "=" * 50)
    print(f"CSV: {out_csv}")
    print(f"{'label':<28} {'saved':>6}  {'skipped':>7}")
    print("-" * 50)
    for lbl, s in stats.items():
        flag = "  ⚠️  (น้อยเกินไป)" if s["saved"] < 30 else ""
        print(f"{lbl:<28} {s['saved']:>6}  {s['skipped']:>7}{flag}")
    print(f"{'รวมทั้งหมด':<28} {total_saved:>6}")

    if total_saved == 0:
        print("\n[err] ไม่ได้บันทึกเลย — ตรวจสอบว่า MediaPipe เจอคนในรูปหรือเปล่า")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET),
                    help=f"path ของโฟลเดอร์ dataset (default: {DEFAULT_DATASET})")
    ap.add_argument("--out", default=str(OUT_CSV),
                    help="output CSV path")
    ap.add_argument("--skip-existing", action="store_true",
                    help="ข้าม label ที่มีใน CSV อยู่แล้ว")
    ap.add_argument("--min-vis", type=float, default=0.3,
                    help="min_detection_confidence ของ MediaPipe (0-1)")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"[err] ไม่พบโฟลเดอร์ {dataset_dir}")
        sys.exit(1)

    print(f"Dataset : {dataset_dir}")
    print(f"Output  : {args.out}")
    process_dataset(
        dataset_dir=dataset_dir,
        out_csv=Path(args.out),
        skip_existing=args.skip_existing,
        min_visibility=args.min_vis,
    )


if __name__ == "__main__":
    main()
