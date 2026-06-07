"""
ตรวจสอบ pipeline inference เต็มด้วยรูปจริงจาก dataset
สุ่มรูปแต่ละคลาส -> MediaPipe -> normalize -> classifier_mp.tflite
แล้วเทียบ label ที่ทำนายได้กับชื่อโฟลเดอร์

จุดประสงค์: จับบั๊ก flip (ซ้าย/ขวาสลับ), normalize ไม่ตรง, label map ผิด

รัน: python verify_dataset.py
     python verify_dataset.py --flip        # ลองแบบมิเรอร์ (ควรแย่ลงถ้า default ถูก)
     python verify_dataset.py --per-class 100
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ongor.mediapipe_runner import (
    MediaPipeExtractor,
    load_label_map,
)
from ongor.paths import DATASET_DIR as DATASET, CLASSIFIER_MP_TFLITE as TFLITE_PATH

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def folder_to_label(folder: Path) -> str:
    name = folder.name
    return name[:-len("-samples")] if name.endswith("-samples") else name


def load_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        return Interpreter(model_path=str(TFLITE_PATH))
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=str(TFLITE_PATH))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--per-class", type=int, default=80,
                    help="จำนวนรูปที่สุ่มทดสอบต่อคลาส")
    ap.add_argument("--flip", action="store_true",
                    help="มิเรอร์เฟรม (เทียบกับ default)")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    label_map = load_label_map()
    label_to_idx = {v: k for k, v in label_map.items()}
    print(f"labels: {[label_map[i] for i in sorted(label_map)]}")
    print(f"flip = {args.flip}\n")

    interp = load_interpreter()
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    extractor = MediaPipeExtractor(static_image_mode=True)
    extractor.flip = args.flip

    rng = random.Random(42)
    n_labels = len(label_map)
    confusion = np.zeros((n_labels, n_labels), dtype=int)
    no_pose = 0
    total = 0
    per_class: dict[str, list[int]] = {}  # label -> [correct, count]

    for folder in sorted(dataset.iterdir()):
        if not folder.is_dir():
            continue
        label = folder_to_label(folder)
        if label not in label_to_idx:
            continue
        true_idx = label_to_idx[label]

        imgs = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
        rng.shuffle(imgs)
        imgs = imgs[: args.per_class]

        correct = count = 0
        for img_path in imgs:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            res = extractor.process(frame)
            if res.keypoints is None:
                no_pose += 1
                continue
            interp.set_tensor(in_d["index"], res.keypoints[None, :])
            interp.invoke()
            probs = interp.get_tensor(out_d["index"])[0]
            pred_idx = int(np.argmax(probs))

            confusion[true_idx, pred_idx] += 1
            count += 1
            total += 1
            if pred_idx == true_idx:
                correct += 1

        per_class[label] = [correct, count]
        acc = correct / count * 100 if count else 0
        flag = "  ⚠️" if acc < 90 else ""
        print(f"  {label:<24} {correct:>4}/{count:<4} = {acc:5.1f}%{flag}")

    overall = sum(c for c, _ in per_class.values())
    print(f"\nรวม: {overall}/{total} = {overall / total * 100:.1f}%  "
          f"(ตรวจไม่เจอคน {no_pose} รูป)")

    # confusion matrix แบบข้อความ — ดูว่าคลาสไหนสับสนกับไหน
    print("\nConfusion (แถว=จริง, คอลัมน์=ทำนาย):")
    idxs = sorted(label_map)
    short = [label_map[i][:10] for i in idxs]
    print("            " + " ".join(f"{s:>10}" for s in short))
    for i in idxs:
        row = " ".join(f"{confusion[i, j]:>10}" for j in idxs)
        print(f"{label_map[i][:10]:>10}  {row}")

    extractor.close()

    # ชี้เป้าคู่ที่สับสนกันเยอะ (มักเป็นคู่ซ้าย/ขวาถ้า flip ผิด)
    print("\nคู่ที่สับสนกันบ่อย (>5%):")
    found = False
    for i in idxs:
        tot = confusion[i].sum()
        if tot == 0:
            continue
        for j in idxs:
            if i != j and confusion[i, j] / tot > 0.05:
                print(f"  {label_map[i]} -> {label_map[j]}: "
                      f"{confusion[i, j]}/{tot} ({confusion[i, j] / tot * 100:.0f}%)")
                found = True
    if not found:
        print("  ไม่มี — pipeline ตรงดี ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
