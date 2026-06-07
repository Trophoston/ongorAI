"""
เทรน MLP classifier จาก MediaPipe keypoints -> export .tflite โดยตรง

วิธีใช้:
  python train_mediapipe.py                    # เทรนพื้นฐาน
  python train_mediapipe.py --epochs 100       # กำหนด epoch เอง
  python train_mediapipe.py --quantize int8    # quantize ให้เล็กลง (สำหรับบอร์ด)
  python train_mediapipe.py --show-confusion   # แสดง confusion matrix

ผลลัพธ์:
  models/classifier_mp.tflite   (สำหรับ inference บนบอร์ด)
  models/classifier_mp.keras    (สำรอง)
  models/label_map.json         (label -> index mapping)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ongor.paths import (
    KEYPOINTS_CSV as DATA_CSV,
    MODELS_DIR as OUT_DIR,
    CLASSIFIER_MP_TFLITE as TFLITE_OUT,
    CLASSIFIER_MP_KERAS as KERAS_OUT,
    LABEL_MAP,
)

N_LANDMARKS = 33
FEATURE_DIM = N_LANDMARKS * 4  # 132


def load_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    df = pd.read_csv(csv_path)
    print(f"[data] โหลด {len(df)} samples")
    print(f"[data] คลาส: {sorted(df['label'].unique())}")
    for lbl, grp in df.groupby("label"):
        print(f"       {lbl}: {len(grp)} samples")

    X = df.iloc[:, 1:].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    print(f"[data] features: {X.shape[1]} (ควรเป็น {FEATURE_DIM})")
    return X, y, le


def temporal_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    """
    แบ่ง train/val แบบ "กันท้าย" ต่อคลาส (เฟรมท้าย ๆ ของแต่ละท่าเป็น val)
    เหมาะกับ dataset ที่เป็นเฟรมต่อเนื่องจากวิดีโอ — กันไม่ให้เฟรมข้างกัน
    ที่เกือบเหมือนกันหลุดไปทั้ง train และ val (data leakage) ทำให้ acc เกินจริง
    หมายเหตุ: ต้องเรียกก่อน shuffle โดยที่ X,y ยังเรียงตามไฟล์ในแต่ละคลาส
    """
    tr_idx, va_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]  # เรียงตามลำดับเดิม (= ลำดับไฟล์)
        cut = int(len(idx) * (1 - test_size))
        tr_idx.extend(idx[:cut])
        va_idx.extend(idx[cut:])
    tr_idx = np.array(tr_idx)
    va_idx = np.array(va_idx)
    return X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]


def normalize(X: np.ndarray) -> np.ndarray:
    """
    Normalize relative to hip center + torso height
    ทำให้ model robust ต่อตำแหน่งและระยะห่างจากกล้อง

    keypoints index (mediapipe):
      23 = left hip,  24 = right hip
      11 = left shoulder, 12 = right shoulder
    """
    X = X.reshape(-1, N_LANDMARKS, 4)
    # hip center
    hip_x = (X[:, 23, 0] + X[:, 24, 0]) / 2
    hip_y = (X[:, 23, 1] + X[:, 24, 1]) / 2
    # shoulder center
    sh_x = (X[:, 11, 0] + X[:, 12, 0]) / 2
    sh_y = (X[:, 11, 1] + X[:, 12, 1]) / 2
    # torso height (ระยะไหล่->สะโพก ใช้ normalize scale)
    scale = np.sqrt((sh_x - hip_x) ** 2 + (sh_y - hip_y) ** 2) + 1e-6

    # normalize x, y (z ไม่ต้อง shift, visibility คงไว้)
    X_norm = X.copy()
    X_norm[:, :, 0] = (X[:, :, 0] - hip_x[:, None]) / scale[:, None]
    X_norm[:, :, 1] = (X[:, :, 1] - hip_y[:, None]) / scale[:, None]
    # z ปรับ scale เฉย ๆ
    X_norm[:, :, 2] = X[:, :, 2] / scale[:, None]

    return X_norm.reshape(-1, N_LANDMARKS * 4).astype(np.float32)


def build_model(n_features: int, n_classes: int) -> tf.keras.Model:
    """MLP เล็ก เหมาะกับ 132 features — เร็ว บนบอร์ดรันสบาย"""
    inputs = tf.keras.Input(shape=(n_features,), name="keypoints")
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="probs")(x)
    model = tf.keras.Model(inputs, outputs, name="mp_classifier")
    return model


def to_tflite(model: tf.keras.Model, quantize: str) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("[tflite] quantize: float16")
    elif quantize == "int8":
        # ต้องมี representative dataset สำหรับ INT8
        raise ValueError("int8 ต้องการ representative data — ใช้ float16 แทนได้")
    else:
        converter.optimizations = []
        print("[tflite] quantize: float32 (ไม่ quantize)")
    return converter.convert()


def show_confusion(model, X_val, y_val, le):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.tight_layout()
    out = OUT_DIR / "confusion_matrix.png"
    plt.savefig(out)
    print(f"[eval] confusion matrix -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DATA_CSV))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--quantize", choices=["none", "float16"], default="none")
    ap.add_argument("--show-confusion", action="store_true")
    ap.add_argument("--split", choices=["temporal", "random"], default="temporal",
                    help="temporal=กันท้ายต่อคลาส (honest, default) | random=สุ่ม (อาจ leak)")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # ---- โหลด + normalize ----
    X, y, le = load_data(Path(args.csv))
    X = normalize(X)

    # ---- แบ่ง train/val ----
    if args.split == "temporal":
        X_train, X_val, y_train, y_val = temporal_split(X, y, test_size=0.2)
        print(f"[data] split=temporal (honest) train={len(X_train)}  val={len(X_val)}")
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"[data] split=random (อาจ leak) train={len(X_train)}  val={len(X_val)}")

    # ---- สร้าง + เทรน ----
    model = build_model(X.shape[1], len(le.classes_))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7, verbose=1),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=1,
    )

    # ---- ประเมิน ----
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[eval] val_acc = {val_acc * 100:.1f}%  val_loss = {val_loss:.4f}")

    if args.show_confusion:
        show_confusion(model, X_val, y_val, le)

    # ---- บันทึก label map ----
    label_map = {int(i): str(lbl) for i, lbl in enumerate(le.classes_)}
    LABEL_MAP.write_text(json.dumps(label_map, ensure_ascii=False, indent=2))
    print(f"[save] label map -> {LABEL_MAP}")
    print(f"       labels: {list(label_map.values())}")

    # ---- บันทึก Keras ----
    model.save(KERAS_OUT)
    print(f"[save] keras -> {KERAS_OUT}")

    # ---- แปลง TFLite ----
    tflite_bytes = to_tflite(model, args.quantize)
    TFLITE_OUT.write_bytes(tflite_bytes)
    print(f"[save] tflite -> {TFLITE_OUT}  ({len(tflite_bytes) / 1024:.1f} KB)")

    # ---- verify ----
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    x_test = X_val[:1]
    interp.set_tensor(in_det["index"], x_test)
    interp.invoke()
    y_tfl = interp.get_tensor(out_det["index"])[0]
    y_ker = model(x_test, training=False).numpy()[0]
    diff = float(np.max(np.abs(y_tfl - y_ker)))
    print(f"[verify] max|tflite-keras| = {diff:.2e}  (ควรน้อยมาก)")
    print(f"[verify] input_dim={in_det['shape'][-1]}  output_dim={out_det['shape'][-1]}")
    print("\nเสร็จสมบูรณ์ ✅")
    print(f"ไฟล์ที่ใช้บนบอร์ด: {TFLITE_OUT}")
    print(f"ไฟล์ labels:       {LABEL_MAP}")


if __name__ == "__main__":
    main()
