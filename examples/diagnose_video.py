"""Save top-3 predictions for an Uno Q clip and optionally print confusion."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ongor.pose_engine import PoseEngine


def load_annotations(path: str | None) -> list[tuple[float, float, str]]:
    if not path:
        return []
    with open(path, newline="") as handle:
        return [
            (float(row["start_sec"]), float(row["end_sec"]), row["label"])
            for row in csv.DictReader(handle)
        ]


def expected_label(
    annotations: list[tuple[float, float, str]], timestamp: float
) -> str | None:
    for start, end, label in annotations:
        if start <= timestamp < end:
            return label
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Uno Q camera clip")
    parser.add_argument("--annotations", help="CSV: start_sec,end_sec,label")
    parser.add_argument("--out", default="logs/unoq_predictions.jsonl")
    parser.add_argument("--every", type=int, default=1, help="analyze every Nth frame")
    parser.add_argument("--flip", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    annotations = load_annotations(args.annotations)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    confusion: Counter[tuple[str, str]] = Counter()

    engine = PoseEngine(flip=args.flip)
    frame_index = 0
    analyzed = 0
    try:
        with output.open("w") as handle:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % max(1, args.every):
                    frame_index += 1
                    continue
                timestamp = frame_index / fps
                _, predictions = engine.process_topk(frame, k=3)
                expected = expected_label(annotations, timestamp)
                predicted = predictions[0].label if predictions else None
                row = {
                    "frame": frame_index,
                    "time_sec": round(timestamp, 3),
                    "expected": expected,
                    "top_predictions": [
                        {
                            "label": item.label,
                            "confidence": round(item.confidence, 6),
                        }
                        for item in predictions
                    ],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if expected and predicted:
                    confusion[(expected, predicted)] += 1
                analyzed += 1
                frame_index += 1
    finally:
        cap.release()
        engine.close()

    print(f"saved {analyzed} analyzed frames to {output}")
    if confusion:
        labels = sorted({label for pair in confusion for label in pair})
        print("confusion matrix (rows=expected, columns=predicted)")
        print("expected\\pred," + ",".join(labels))
        for actual in labels:
            values = [str(confusion[(actual, predicted)]) for predicted in labels]
            print(actual + "," + ",".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
