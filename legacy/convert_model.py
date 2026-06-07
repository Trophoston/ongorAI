"""
แปลงโมเดล Teachable Machine (TFJS Layers) -> Keras (.keras) ครั้งเดียว
รัน:  python convert_model.py
ผลลัพธ์: python_app/models/classifier.keras
"""
import sys
from pathlib import Path

import tensorflowjs as tfjs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ongor.paths import TM_MODEL_JSON as SRC, MODELS_DIR as DST_DIR, CLASSIFIER_KERAS as DST


def main() -> None:
    DST_DIR.mkdir(exist_ok=True)
    model = tfjs.converters.load_keras_model(str(SRC))
    model.summary()
    model.save(DST)
    print(f"saved -> {DST}")


if __name__ == "__main__":
    main()
