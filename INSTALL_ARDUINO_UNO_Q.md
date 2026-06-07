# ติดตั้ง & ใช้งาน Ong-Or Pose API บน Arduino Uno Q (2GB)

คู่มือนี้สอนตั้งแต่ติดตั้งจน "เขียนโค้ดเรียกใช้ท่าทาง" ได้จริง

---

## 0. เข้าใจสถาปัตยกรรม Uno Q ก่อน (สำคัญ)

Arduino Uno Q มี **สองสมอง**:

| ส่วน | ชิป | หน้าที่ | ภาษา |
|------|-----|---------|------|
| **Linux (MPU)** | Qualcomm Dragonwing QRB2210 (Cortex-A53 4 คอร์, aarch64) | รัน Python, กล้อง, AI, API | Python / Debian Linux |
| **MCU** | STM32U585 | ควบคุมขา I/O, มอเตอร์, เซนเซอร์แบบเรียลไทม์ | Arduino sketch (C++) |

**โมเดลตรวจจับท่าทางทั้งหมดรันบนฝั่ง Linux** (Python + MediaPipe + tflite)
ส่วน Arduino sketch ฝั่ง MCU เอาไว้รับ "ผลท่าที่ทำนายได้" ไปสั่งงานฮาร์ดแวร์ต่อ

> 📷 Uno Q **ไม่มีกล้องในตัว** — ต้องต่อ **USB webcam** เข้าพอร์ต USB ของบอร์ด

---

## 1. เตรียมบอร์ด

เปิดเทอร์มินัลบน Uno Q (ผ่าน Arduino App Lab หรือ SSH) แล้ว:

```bash
# อัปเดตระบบ
sudo apt update && sudo apt upgrade -y

# ของที่ระบบต้องมี: python venv, git, และไลบรารีของ OpenCV/กล้อง
sudo apt install -y python3-venv python3-pip git \
    libgl1 libglib2.0-0 libsm6 libxext6 v4l-utils

# เช็คว่าเห็นกล้อง USB ไหม (ควรเห็น /dev/video0)
v4l2-ctl --list-devices
```

ตรวจเวอร์ชัน Python (ต้อง 3.9–3.12):
```bash
python3 --version
```

---

## 2. ดึงโค้ดลงบอร์ด

```bash
cd ~
git clone https://github.com/Trophoston/ongorAI.git
cd ongorAI
```

---

## 3. ติดตั้ง (เวอร์ชันบอร์ด — เบา ไม่ใช้ tensorflow)

บนบอร์ด ARM ที่แรม 2GB **อย่าลง tensorflow ตัวเต็ม** (ใหญ่ ~600MB+ และ build นาน)
โค้ด `classifier_mp.py` ถูกออกแบบให้ใช้ **`tflite-runtime`** ก่อนอยู่แล้ว (เล็กมาก เร็วพอ)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# ไลบรารีหลัก (เวอร์ชันถูกล็อกไว้เพราะเคยทำให้เกิด error 503 มาก่อน)
pip install "numpy>=1.24,<2.0" "opencv-python>=4.8" \
            "mediapipe==0.10.21" "protobuf>=4.25.3,<5" \
            "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9"

# ตัวรันโมเดล .tflite แบบเบา (เลือก 1 ใน 2)
pip install tflite-runtime        # แนะนำบนบอร์ด ARM
# ถ้า tflite-runtime ลงไม่ได้ ค่อยใช้ตัวสำรอง:
# pip install ai-edge-litert
```

> **ทำไมต้องล็อกเวอร์ชันพวกนี้?** (สาเหตุของ 503 เดิม)
> - `mediapipe` รุ่น 0.10.30 ขึ้นไป **ตัด API `mp.solutions.pose`** ที่โค้ดนี้ใช้ → ต้องใช้ **0.10.21**
> - `mediapipe 0.10.21` ต้องการ `protobuf 4.25.x` ถ้า lib อื่นดัน protobuf เป็น 6 จะพัง (`MessageFactory ... GetPrototype`)
> - `numpy` ต้อง `<2.0` ให้เข้ากับ wheel ของ mediapipe/opencv

### ถ้า `mediapipe==0.10.21` ลงบน aarch64 ไม่ได้
PyPI บางทีไม่มี wheel aarch64 ของ mediapipe ลองตามลำดับนี้:
```bash
# 1) ลองรุ่นใกล้เคียงที่ยังมี solutions API (ห้ามเกิน 0.10.21)
pip install "mediapipe==0.10.18" || pip install "mediapipe==0.10.14" || pip install "mediapipe==0.10.9"

# 2) ถ้ายังไม่ได้ ใช้ของ Raspberry Pi / ARM ที่ชุมชนทำไว้
pip install mediapipe-rpi4   # หรือค้น wheel aarch64 จาก piwheels.org
```

---

## 4. รัน API แล้วทดสอบ

```bash
# (อยู่ใน .venv แล้ว)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

เปิดอีกเทอร์มินัลทดสอบ:
```bash
# 4.1 เช็คสุขภาพ — ต้องได้ "vision":true
curl http://localhost:8000/health

# 4.2 ทดสอบทำนายด้วยรูปนิ่ง
curl -F "image=@/path/to/คน.jpg" http://localhost:8000/predict
```

ผลที่ควรได้:
```json
{"prediction":{"label":"hul_hand_up_L","confidence":0.83,"thai":"ยกมือซ้าย"},
 "confirmed":null,"pose_detected":true}
```

> ถ้า `"vision":false` แปลว่าไลบรารีตรวจจับยังโหลดไม่ได้ — ดูข้อความใน `"error"`
> ของ `/health` มันจะบอกตรง ๆ ว่าติดอะไร (เช่น import mediapipe พัง)

### ทดสอบกล้องสด
```bash
python examples/predict_webcam.py --camera 0
```

---

## 5. ตั้งให้รันเองตอนเปิดบอร์ด (ทางเลือก)

ในโฟลเดอร์ `deploy/` มีไฟล์ systemd ให้แล้ว:
```bash
# แก้ค่าในไฟล์ env ก่อน (พอร์ต, CORS ฯลฯ)
cp deploy/ongor-fastapi.env.example /etc/default/ongor-fastapi
sudo nano /etc/default/ongor-fastapi

# ติดตั้ง service (สคริปต์อยู่ใน repo)
sudo bash install_systemd_service.sh
sudo systemctl enable --now ongor-fastapi
systemctl status ongor-fastapi
```

---

## 6. วิธีเขียนโค้ดเรียกใช้งาน (3 แบบ)

### แบบ A — เรียกตรงในไพ ธอน (เร็วสุด ไม่ผ่านเน็ต) ⭐ แนะนำบนบอร์ด
```python
import cv2
from ongor.pose_engine import PoseEngine

engine = PoseEngine()                 # โหลดโมเดลครั้งเดียว
frame = cv2.imread("คน.jpg")          # หรือเฟรมจากกล้อง
pred = engine.predict(frame)          # -> Prediction | None
if pred:
    print(pred.label, pred.confidence, pred.thai)
```
ถ้าอยากได้แบบ "ค้างท่าไว้ถึงนับ" (กันสั่งซ้ำ เหมาะกับเล่นเกม/สั่งงาน):
```python
confirmed = engine.read_confirmed(frame)   # คืน label เฉพาะตอนเพิ่งยืนยัน
if confirmed:
    print("ยืนยันท่า:", confirmed)
```
ดูตัวอย่างเต็มที่ [`examples/predict_webcam.py`](examples/predict_webcam.py)

### แบบ B — เรียกผ่าน HTTP (โค้ดอยู่คนละเครื่อง/คนละภาษา)
```bash
pip install requests
python examples/predict_http_client.py --url http://localhost:8000
```
หรือจาก JavaScript / แอปอื่น:
```js
const fd = new FormData();
fd.append("image", jpegBlob, "frame.jpg");
const r = await fetch("http://<ip-บอร์ด>:8000/predict", { method: "POST", body: fd });
const data = await r.json();   // { prediction:{label,confidence,thai}, confirmed, pose_detected }
```

### แบบ C — ส่งผลไปให้ Arduino sketch (ฝั่ง MCU) สั่งฮาร์ดแวร์
โครงสร้างทั่วไป: Python (Linux) ทำนายท่า → ส่ง label ผ่าน serial → sketch รับไปสั่ง LED/มอเตอร์

ฝั่ง Python:
```python
import serial
from ongor.pose_engine import PoseEngine
# พอร์ตเชื่อม MCU ภายในบอร์ด Uno Q (ตรวจจริงด้วย: ls /dev/ttyAMA* /dev/ttyACM*)
mcu = serial.Serial("/dev/ttyAMA0", 115200)
engine = PoseEngine()
# ...ในลูปกล้อง...
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
> พอร์ต serial ที่เชื่อม Linux↔MCU ของ Uno Q ให้ยืนยันชื่อจริงในเอกสารบอร์ด/`ls /dev/tty*`

---

## 7. รายการ API ทั้งหมด (ตอนนี้มีแค่ "ทำนายท่า")

| Method | Path | ใช้ทำอะไร |
|--------|------|-----------|
| `GET`  | `/health` | เช็คว่าระบบ + ไลบรารีตรวจจับพร้อมไหม (`vision: true/false`) |
| `POST` | `/predict` | อัปโหลดรูป (multipart field ชื่อ `image`) → คืนผลทำนายท่า |

**รูปแบบผลลัพธ์ของ `/predict`:**
```jsonc
{
  "prediction": { "label": "hub_hand_up_Both", "confidence": 0.91, "thai": "ยกมือสองข้าง" },
  "confirmed": null,        // จะเป็น label ก็ต่อเมื่อค้างท่า >0.6s และมั่นใจ >0.85
  "pose_detected": true     // false = ในรูปไม่เจอคน/ท่า
}
```

---

## 8. แก้ปัญหาที่เจอบ่อย

| อาการ | สาเหตุ / วิธีแก้ |
|-------|------------------|
| `/predict` ตอบ **503** | ไลบรารีตรวจจับโหลดไม่ได้ → ดู `/health` ช่อง `error`, มักเพราะ mediapipe เวอร์ชันผิด → ใช้ `0.10.21` |
| `module 'mediapipe' has no attribute 'solutions'` | mediapipe ใหม่เกินไป → `pip install "mediapipe==0.10.21"` |
| `MessageFactory ... GetPrototype` | protobuf ชนกัน → `pip install "protobuf>=4.25.3,<5"` |
| เปิดกล้องไม่ได้ | เช็ค `v4l2-ctl --list-devices`, ลองเปลี่ยน `--camera 1`, ดูสิทธิ์ `sudo usermod -aG video $USER` |
| `confirmed` เป็น null ตลอด | ปกติ — ต้อง "ค้างท่า" ให้นิ่ง >0.6 วิ และมั่นใจ >0.85 ถึงจะยืนยัน |
| ทำนายช้า | ลด `model_complexity=0` ตอนสร้าง `PoseEngine(model_complexity=0)` |

---

## หมายเหตุการทดสอบ
API นี้ถูกทดสอบผ่านบน **macOS arm64 / Python 3.9** ด้วย `tensorflow 2.16.2` (แทน tflite-runtime
ที่ลงยากบน Mac) — ผลทำนายและทุก endpoint ทำงานถูกต้อง บนบอร์ด ARM ให้ใช้ `tflite-runtime`
ตามข้อ 3 ซึ่งเบากว่าและเป็นเส้นทางที่โค้ดเลือกใช้เป็นอันดับแรกอยู่แล้ว
