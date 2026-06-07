# ติดตั้ง & ใช้งาน Ong-Or Pose API บน Arduino Uno Q (2GB)

คู่มือนี้สอนตั้งแต่ติดตั้งจน "เขียนโค้ดเรียกใช้ท่าทาง" ได้จริง

> **สำคัญ:** API นี้รันโมเดล BlazePose ผ่าน **TFLite ตรง ๆ ไม่ใช้แพ็กเกจ `mediapipe`**
> เพราะ mediapipe ไม่มี wheel สำหรับ Linux aarch64 (เช่น Uno Q) เลยสักเวอร์ชัน
> โมเดล `pose_landmark_full.tflite` มาพร้อม repo แล้วใน `models/blazepose/`

---

## 0. เข้าใจสถาปัตยกรรม Uno Q ก่อน

Arduino Uno Q มี **สองสมอง**:

| ส่วน | ชิป | หน้าที่ | ภาษา |
|------|-----|---------|------|
| **Linux (MPU)** | Qualcomm Dragonwing QRB2210 (Cortex-A53 4 คอร์, aarch64) | รัน Python, กล้อง, AI, API | Python / Debian Linux |
| **MCU** | STM32U585 | ควบคุมขา I/O, มอเตอร์, เซนเซอร์แบบเรียลไทม์ | Arduino sketch (C++) |

**โมเดลตรวจจับท่าทางรันบนฝั่ง Linux ทั้งหมด** (Python + TFLite)
ส่วน Arduino sketch ฝั่ง MCU เอาไว้รับ "ผลท่าที่ทำนายได้" ไปสั่งงานฮาร์ดแวร์

> 📷 Uno Q **ไม่มีกล้องในตัว** — ต้องต่อ **USB webcam** เข้าพอร์ต USB
> 🧍 วิธีนี้ทำงานดีเมื่อ **คนยืนเต็มตัวอยู่กลางเฟรม** (จัดกล้องให้เห็นทั้งตัว)

---

## 1. เตรียมบอร์ด

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git libgl1 libglib2.0-0 libsm6 libxext6 v4l-utils

v4l2-ctl --list-devices     # เช็คว่าเห็นกล้อง USB (ควรเห็น /dev/video0)
python3 --version           # ดูว่าบอร์ดเป็น Python เวอร์ชันอะไร
```

---

## 2. ดึงโค้ดลงบอร์ด

```bash
cd /opt        # หรือ ~ ก็ได้
sudo git clone https://github.com/Trophoston/ongorAI.git
sudo chown -R $USER:$USER ongorAI     # ให้ user ปัจจุบันเป็นเจ้าของ (เลี่ยงปัญหาสิทธิ์)
cd ongorAI
```

---

## 3. ติดตั้ง — เลือกตามเวอร์ชัน Python ของบอร์ด

มี dependency แค่ไม่กี่ตัว และ **ทุกตัวมี wheel aarch64 สำเร็จรูป** (ไม่ต้องคอมไพล์):
`numpy`, `opencv-python`, `fastapi`, `uvicorn`, `python-multipart` + ตัวรัน tflite

ตัวรัน tflite ต่างกันตามเวอร์ชัน Python:

### ทาง A — บอร์ดเป็น Python ≤ 3.11 → ใช้ `tflite-runtime` (เบาสุด แนะนำ)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy opencv-python fastapi "uvicorn[standard]" python-multipart tflite-runtime
```

### ทาง B — บอร์ดเป็น Python 3.12 / 3.13 → ใช้ `ai-edge-litert`
(`tflite-runtime` ยังไม่มี wheel aarch64 สำหรับ 3.12+ แต่ `ai-edge-litert` มี)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy opencv-python fastapi "uvicorn[standard]" python-multipart ai-edge-litert
```

### ทาง C — อยากได้ Python 3.11 แต่บอร์ดมีแค่ 3.13 → ใช้ `uv` ดึง 3.11 มา
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install numpy opencv-python fastapi "uvicorn[standard]" python-multipart tflite-runtime
```

> 💡 จะใช้ `requirements.txt` แทนก็ได้ (มันเลือก tflite-runtime/ai-edge-litert ให้
> อัตโนมัติตามเวอร์ชัน Python): `pip install -r api/requirements.txt`

---

## 4. รัน API แล้วทดสอบ

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

อีกเทอร์มินัล:
```bash
curl http://localhost:8000/health                       # ต้องได้ "vision":true
curl -F "image=@/path/to/คน.jpg" http://localhost:8000/predict
```

ผลที่ควรได้:
```json
{"prediction":{"label":"hub_hand_up_Both","confidence":0.91,"thai":"ยกมือสองข้าง"},
 "confirmed":null,"pose_detected":true}
```

ทดสอบกล้องสด:
```bash
python examples/predict_webcam.py --camera 0
```

---

## 5. ตั้งให้รันเองตอนเปิดบอร์ด (ทางเลือก)

```bash
cp deploy/ongor-fastapi.env.example /etc/default/ongor-fastapi
sudo nano /etc/default/ongor-fastapi          # แก้พอร์ต/CORS
sudo bash install_systemd_service.sh
sudo systemctl enable --now ongor-fastapi
systemctl status ongor-fastapi
```

---

## 6. วิธีเขียนโค้ดเรียกใช้งาน (3 แบบ)

### แบบ A — เรียกตรงในไพธอน (เร็วสุด ไม่ผ่านเน็ต) ⭐ แนะนำบนบอร์ด
```python
import cv2
from ongor.pose_engine import PoseEngine

engine = PoseEngine()                 # โหลดโมเดลครั้งเดียว
frame = cv2.imread("คน.jpg")          # หรือเฟรมจากกล้อง
pred = engine.predict(frame)          # -> Prediction | None
if pred:
    print(pred.label, pred.confidence, pred.thai)
```
แบบ "ค้างท่าไว้ถึงนับ" (กันสั่งซ้ำ เหมาะกับเล่นเกม/สั่งงาน):
```python
confirmed = engine.read_confirmed(frame)   # คืน label เฉพาะตอนเพิ่งยืนยัน
if confirmed:
    print("ยืนยันท่า:", confirmed)
```
ตัวอย่างเต็ม: [`examples/predict_webcam.py`](examples/predict_webcam.py)

### แบบ B — เรียกผ่าน HTTP (โค้ดอยู่คนละเครื่อง/คนละภาษา)
```bash
pip install requests
python examples/predict_http_client.py --url http://localhost:8000
```
จาก JavaScript:
```js
const fd = new FormData();
fd.append("image", jpegBlob, "frame.jpg");
const r = await fetch("http://<ip-บอร์ด>:8000/predict", { method: "POST", body: fd });
const data = await r.json();   // { prediction:{label,confidence,thai}, confirmed, pose_detected }
```

### แบบ C — ส่งผลไปให้ Arduino sketch (ฝั่ง MCU) สั่งฮาร์ดแวร์
Python (Linux) ทำนายท่า → ส่ง label ผ่าน serial → sketch รับไปสั่ง LED/มอเตอร์

ฝั่ง Python:
```python
import cv2, serial
from ongor.pose_engine import PoseEngine
mcu = serial.Serial("/dev/ttyAMA0", 115200)   # ยืนยันพอร์ตจริง: ls /dev/ttyACM* /dev/ttyAMA*
engine = PoseEngine()
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: continue
    confirmed = engine.read_confirmed(frame)
    if confirmed:
        mcu.write((confirmed + "\n").encode())
```
ฝั่ง Arduino sketch (MCU):
```cpp
void loop() {
  if (Serial.available()) {
    String pose = Serial.readStringUntil('\n');
    if (pose == "hub_hand_up_Both") digitalWrite(LED_BUILTIN, HIGH);
    // ...สั่งงานตามท่า...
  }
}
```

---

## 7. รายการ API (มีแค่ "ทำนายท่า")

| Method | Path | ใช้ทำอะไร |
|--------|------|-----------|
| `GET`  | `/health` | เช็คว่าระบบ + โมเดลพร้อมไหม (`vision: true/false`, `error`) |
| `POST` | `/predict` | อัปโหลดรูป (multipart field ชื่อ `image`) → ผลทำนายท่า |

**ผลลัพธ์ `/predict`:**
```jsonc
{
  "prediction": { "label": "hub_hand_up_Both", "confidence": 0.91, "thai": "ยกมือสองข้าง" },
  "confirmed": null,        // เป็น label เมื่อค้างท่า >0.6s และมั่นใจ >0.85 เท่านั้น
  "pose_detected": true     // false = ในรูปไม่เจอคน/ท่า
}
```

ท่าทั้งหมด (label): `Panomue`(พนมมือ), `thb/thl/thr_touch_head*`(แตะหัว), `hub/hul/hur_hand_up*`(ยกมือ),
`tpb/tpl/tpr_t_post*`(ทีโพส), `idle`(อยู่นิ่ง)

---

## 8. แก้ปัญหาที่เจอบ่อย

| อาการ | สาเหตุ / วิธีแก้ |
|-------|------------------|
| `/predict` ตอบ **503** | โมเดล/ไลบรารีโหลดไม่ได้ → ดู `/health` ช่อง `error` |
| ลง `tflite-runtime` ไม่ได้ (ไม่มี wheel) | บอร์ดเป็น Python 3.12+ → ใช้ `ai-edge-litert` (ทาง B) แทน |
| `numpy`/`opencv` โหลดเป็น `.tar.gz` แล้ว build นาน | Python ใหม่เกินจน ไม่มี wheel → ลด Python ด้วย `uv` (ทาง C) |
| เปิดกล้องไม่ได้ | `v4l2-ctl --list-devices`, ลอง `--camera 1`, `sudo usermod -aG video $USER` |
| ทำนายเพี้ยน | จัดให้เห็น **คนเต็มตัว ยืนตรง** (ระบบ crop ซูมหาคนให้เอง คนตัวเล็กกลางเฟรมก็ได้) |
| `confirmed` เป็น null ตลอด | ปกติ — ต้องค้างท่านิ่ง >0.6 วิ และมั่นใจ >0.85 |
| `pose_detected` เป็น false ทั้งที่มีคน | คนเล็ก/ไกลเกินไป → ขยับเข้าใกล้ให้เต็มเฟรม หรือลด threshold ใน `PoseEngine` |

---

## หมายเหตุทางเทคนิค
- โมเดลที่ใช้คือ `pose_landmark_full.tflite` ของ MediaPipe (BlazePose) รันผ่าน tflite
  โดยป้อนทั้งเฟรม (letterbox 256×256) แบบสเตจเดียว แล้วแปลงเป็น feature 132 มิติ
  ป้อน `classifier_mp.tflite` ที่เทรนไว้ — **ไม่ต้องเทรนใหม่**
- ทดสอบเทียบกับ `mp.solutions.pose` แล้ว: ตำแหน่ง landmark คลาดเคลื่อนเฉลี่ย ~0.009
  (หน่วยภาพ-normalized) และ classifier ทำนาย label เดียวกัน
- ตัว extractor อยู่ที่ [`ongor/mediapipe_runner.py`](ongor/mediapipe_runner.py)
