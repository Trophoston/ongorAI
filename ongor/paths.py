"""
ศูนย์รวมพาธทั้งหมดของโปรเจกต์ (single source of truth)
ทุกสคริปต์/โมดูลอ้างพาธจากที่นี่ จะได้ย้ายไฟล์ไปโฟลเดอร์ไหนก็ไม่พัง

โครงสร้าง:
  OngOrDepa/                <- PROJECT_DIR
  ├── detaset/              <- DATASET_DIR (รูปฝึกแยกโฟลเดอร์ตามท่า)
  ├── my-pose-model/        <- โมเดล Teachable Machine เก่า
  └── python_app/           <- APP_DIR
      ├── ongor/            <- PKG_DIR (ไลบรารีร่วม)
      ├── models/           <- MODELS_DIR (โมเดลที่เทรนแล้ว)
      ├── data/             <- DATA_DIR (keypoints.csv)
      ├── logs/             <- LOGS_DIR (score log)
      ├── training/  game/  legacy/  mcu/
"""
from __future__ import annotations

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # .../python_app/ongor
APP_DIR = PKG_DIR.parent                            # .../python_app
PROJECT_DIR = APP_DIR.parent                        # .../OngOrDepa

# โฟลเดอร์
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"
LOGS_DIR = APP_DIR / "logs"
DATASET_DIR = PROJECT_DIR / "detaset"

# ไฟล์ที่ใช้บ่อย
LABEL_MAP = MODELS_DIR / "label_map.json"
CLASSIFIER_MP_TFLITE = MODELS_DIR / "classifier_mp.tflite"
# โมเดล BlazePose ของ MediaPipe (ดึง landmark) — ใช้ตอนรันบนบอร์ดผ่าน tflite ตรง ๆ
# โดยไม่ต้องลงแพ็กเกจ mediapipe (ซึ่งไม่มี wheel สำหรับ Linux aarch64)
BLAZEPOSE_LANDMARK_TFLITE = MODELS_DIR / "blazepose" / "pose_landmark_full.tflite"
CLASSIFIER_MP_KERAS = MODELS_DIR / "classifier_mp.keras"
KEYPOINTS_CSV = DATA_DIR / "keypoints.csv"
SCORES_LOG = LOGS_DIR / "scores.jsonl"

# legacy (Teachable Machine / PoseNet)
TM_MODEL_DIR = PROJECT_DIR / "my-pose-model"
TM_METADATA = TM_MODEL_DIR / "metadata.json"
TM_MODEL_JSON = TM_MODEL_DIR / "model.json"
POSENET_DIR = MODELS_DIR / "posenet"
CLASSIFIER_TFLITE = MODELS_DIR / "classifier.tflite"
CLASSIFIER_KERAS = MODELS_DIR / "classifier.keras"


def ensure_dirs() -> None:
    """สร้างโฟลเดอร์ output ถ้ายังไม่มี"""
    for d in (MODELS_DIR, DATA_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
