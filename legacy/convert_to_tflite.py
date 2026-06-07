"""
แปลง Teachable Machine classifier (TFJS Layers) -> .tflite
*** ไม่ใช้ tensorflowjs *** จึงเลี่ยงปัญหา protobuf / tensorflow_decision_forests

วิธีการ: โมเดลเป็น Sequential ง่าย ๆ (Dense -> Dropout -> Dense) และ weights.bin
เก็บ float32 เรียงต่อกันตาม weightsManifest — เราอ่านน้ำหนักเอง สร้าง Keras ใหม่
ใส่น้ำหนัก แล้วแปลงเป็น TFLite

รัน:  python convert_to_tflite.py
ผลลัพธ์:
  models/classifier.tflite   (โมเดลสำหรับ inference บนบอร์ด)
  models/classifier.keras    (เผื่อใช้ debug ฝั่ง desktop)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ongor.paths import (
    TM_MODEL_DIR as SRC_DIR,
    TM_MODEL_JSON as MODEL_JSON,
    MODELS_DIR as OUT_DIR,
    CLASSIFIER_TFLITE as TFLITE_OUT,
    CLASSIFIER_KERAS as KERAS_OUT,
)

WEIGHTS_BIN = SRC_DIR / "weights.bin"

_DTYPE = {
    "float32": np.float32,
    "float16": np.float16,
    "int32": np.int32,
    "uint8": np.uint8,
}


def read_tfjs_weights(model_json: Path, weights_bin: Path) -> dict[str, np.ndarray]:
    """อ่าน weights.bin ตามลำดับใน weightsManifest -> dict ชื่อ -> ndarray"""
    manifest = json.loads(model_json.read_text())
    raw = np.fromfile(weights_bin, dtype=np.uint8)

    weights: dict[str, np.ndarray] = {}
    offset = 0
    for group in manifest["weightsManifest"]:
        for spec in group["weights"]:
            name = spec["name"]
            shape = spec["shape"]
            dtype = _DTYPE[spec["dtype"]]
            count = int(np.prod(shape)) if shape else 1
            nbytes = count * np.dtype(dtype).itemsize
            chunk = raw[offset:offset + nbytes].view(dtype)
            if chunk.size != count:
                raise ValueError(
                    f"weights.bin สั้นเกินไปสำหรับ {name}: "
                    f"ต้องการ {count} ได้ {chunk.size}"
                )
            weights[name] = chunk.reshape(shape).copy()
            offset += nbytes

    if offset != raw.nbytes:
        print(f"[warn] เหลือ byte ไม่ได้ใช้ {raw.nbytes - offset} (ปกติควรเป็น 0)")
    return weights


def build_keras(model_json: Path, weights: dict[str, np.ndarray]) -> tf.keras.Model:
    """สร้าง Sequential ตาม config ใน model.json แล้วยัดน้ำหนักที่อ่านมา"""
    topo = json.loads(model_json.read_text())["modelTopology"]
    layers_cfg = topo["config"]["layers"]

    inputs = None
    model_layers = []
    for lc in layers_cfg:
        cls = lc["class_name"]
        cfg = lc["config"]
        if cls == "Dense":
            if inputs is None:
                # batch_input_shape: [null, 14739] -> input_dim = 14739
                bis = cfg["batch_input_shape"]
                inputs = tf.keras.Input(shape=(bis[-1],), name="features")
            layer = tf.keras.layers.Dense(
                units=cfg["units"],
                activation=cfg["activation"],
                use_bias=cfg["use_bias"],
                name=cfg["name"],
            )
            model_layers.append(layer)
        elif cls == "Dropout":
            # Dropout ไม่มีผลตอน inference — ใส่ไว้ให้โครงตรง
            model_layers.append(tf.keras.layers.Dropout(cfg["rate"], name=cfg["name"]))
        else:
            raise NotImplementedError(f"ยังไม่รองรับเลเยอร์ {cls}")

    x = inputs
    for layer in model_layers:
        x = layer(x)
    model = tf.keras.Model(inputs, x, name="tm_classifier")

    # ใส่น้ำหนัก: Dense -> [kernel] หรือ [kernel, bias]
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Dense):
            continue
        kernel = weights[f"{layer.name}/kernel"]
        if layer.use_bias:
            bias = weights[f"{layer.name}/bias"]
            layer.set_weights([kernel, bias])
        else:
            layer.set_weights([kernel])

    return model


def to_tflite(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # float32 ปกติ — แม่นยำสุด, ขนาดราว 5.9MB
    converter.optimizations = []
    return converter.convert()


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print("[1/4] อ่าน weights.bin ...")
    weights = read_tfjs_weights(MODEL_JSON, WEIGHTS_BIN)
    for name, arr in weights.items():
        print(f"      {name:24s} {arr.shape} {arr.dtype}")

    print("[2/4] สร้าง Keras model ...")
    model = build_keras(MODEL_JSON, weights)
    model.summary()
    model.save(KERAS_OUT)
    print(f"      saved -> {KERAS_OUT}")

    print("[3/4] แปลงเป็น TFLite ...")
    tflite_bytes = to_tflite(model)
    TFLITE_OUT.write_bytes(tflite_bytes)
    print(f"      saved -> {TFLITE_OUT}  ({len(tflite_bytes) / 1e6:.2f} MB)")

    print("[4/4] ตรวจสอบ TFLite ด้วย dummy input ...")
    verify(tflite_bytes, model)
    print("เสร็จสมบูรณ์ ✅")


def verify(tflite_bytes: bytes, keras_model: tf.keras.Model) -> None:
    """รัน TFLite vs Keras ด้วย input สุ่ม เทียบผลต้องใกล้กัน"""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, inp["shape"][-1])).astype(np.float32)

    interp.set_tensor(inp["index"], x)
    interp.invoke()
    y_tflite = interp.get_tensor(out["index"])[0]
    y_keras = keras_model(x, training=False).numpy()[0]

    max_diff = float(np.max(np.abs(y_tflite - y_keras)))
    n_out = int(keras_model.output_shape[-1])
    print(f"      output dim = {y_tflite.shape[0]} (ควรเป็น {n_out})")
    print(f"      sum(prob)  = {y_tflite.sum():.4f} (ควรใกล้ 1.0)")
    print(f"      max|tflite - keras| = {max_diff:.2e} (ควรน้อยมาก)")
    if max_diff > 1e-4:
        print("[warn] ผลต่างมากผิดปกติ — ตรวจสอบการแปลงอีกที")


if __name__ == "__main__":
    main()
