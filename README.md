# Ong-Or — Python side (Arduino Uno Q)

ส่วน Linux ของ Arduino Uno Q — กล้อง USB + **MediaPipe Pose** ตรวจจับท่าทาง
แล้วเดินเกม "อ่องออ" สื่อสารกับฝั่ง MCU (`.ino`) ผ่าน serial

## Pipeline (เวอร์ชันปัจจุบัน)

```
กล้อง → MediaPipe Pose (33 keypoints × 4 = 132) → normalize → MLP classifier → TFLite
```

โมเดลเล็ก ~100 KB, เทรนเองได้, ไม่ต้องพึ่ง Teachable Machine

## ⚠️ Python environment

ใช้ **base conda** (`/opt/anaconda3/bin/python`) ที่มี `mediapipe` + `tensorflow` ครบ
`.venv` ในโฟลเดอร์นี้ **ไม่มี mediapipe** — ถ้าจะใช้ venv ต้อง `pip install mediapipe scikit-learn pandas` ก่อน

```bash
# ตรวจว่า python ตัวไหนมี mediapipe
/opt/anaconda3/bin/python -c "import mediapipe, tensorflow; print('ok')"
```

## โครงสร้าง

```
python_app/
├── dataset_to_csv.py       อ่านรูปจาก ../detaset/<label>-samples/ → data/keypoints.csv
├── train_mediapipe.py      เทรน MLP → models/classifier_mp.tflite + label_map.json
├── verify_dataset.py       รันรูปจริงผ่าน pipeline เช็ค accuracy + จับบั๊ก flip
├── test_mediapipe.py       ทดสอบจำแนกท่า real-time จากกล้อง
├── test_game.py            เทสเกม: --logic (ไม่ใช้กล้อง) / เล่นจริง
├── main.py                 เกม Ong-Or เต็ม + ฟังก์ชัน play() ใช้ซ้ำได้
├── collect_data.py         (ทางเลือก) เก็บ data จากกล้องสดแทนการใช้รูป
└── ongor/
    ├── __init__.py         export API: PoseEngine, SequenceGame, ...
    ├── labels.py           โหลด label จาก label_map.json + ชื่อไทย
    ├── mediapipe_runner.py extractor (process ครั้งเดียว ได้ keypoints+landmarks)
    ├── classifier_mp.py    TFLite classifier → Prediction
    ├── pose_engine.py      ★ API ง่ายสำหรับบอร์ด (MediaPipe+tflite+stabilizer)
    ├── sequence_game.py    ★ เกมลำดับ Simon-Says
    ├── events.py           ★ ระบบส่งค่าออก (print/serial/fan-out)
    └── arduino_link.py     serial bridge ไป MCU
```

## เรียกใช้บนบอร์ดง่าย ๆ

```python
from ongor import PoseEngine

engine = PoseEngine()                    # โหลด MediaPipe + tflite ครบ
pred = engine.predict(frame_bgr)         # -> Prediction(label, confidence, thai) | None

# หรือแบบ "ค้างท่าถึงนับ" (เหมาะกับเกม)
confirmed = engine.read_confirmed(frame_bgr)   # -> label str | None
```

## เกมอ่องออ (เกมความจำท่าทาง)

กติกา:
1. โชว์ **ท่าใหม่ 1 ท่า** ให้จำ — `memorize_time` = **5 วินาที**
2. ครบเวลา → ทำ **ท่าเดิมทั้งหมดตามลำดับ** แล้วจบด้วยท่าใหม่
3. ทำถูกแต่ละท่า → **รีเซ็ตเวลาเป็น 10 วินาที** (`step_timeout`)
4. ทำครบ → เพิ่มท่าใหม่ กลับไปข้อ 1 (ลำดับยาวขึ้น คะแนนสะสม)
5. **ทำผิด หรือเกิน 10 วิ → GAME OVER + บันทึกคะแนน**

คะแนนถูกบันทึกลง `logs/scores.jsonl` ทุกเกม (มี timestamp + duration):
```json
{"ts": 1780794773.1, "time": "2026-06-07T08:12:53.105", "result": "game_over", "score": 1, "round": 2, "duration_sec": 42.3}
```

```bash
$PY test_game.py --logic          # เทสตรรกะเกม (ไม่ใช้กล้อง, deterministic)
$PY main.py                       # เล่นจริง
$PY main.py --list-cameras        # สแกนกล้องที่ใช้ได้
$PY main.py --camera 1            # เลือกกล้องตอนเริ่ม
$PY main.py --serial /dev/ttyACM0 # ส่ง event ไป MCU ด้วย
$PY main.py --no-serial           # ไม่ต่อ serial
```

ปุ่มในเกม: `SPACE`=เริ่ม/เล่นใหม่ · `Q`=หยุด · `0-9`=สลับกล้องตอนเล่น · `ESC`=ออก
> HUD เป็นภาษาอังกฤษ เพราะ `cv2.putText` แสดงภาษาไทยไม่ได้ (รองรับแค่ ASCII)
> ถ้า `--list-cameras` ขึ้น "not authorized" บน macOS → ไปเปิดสิทธิ์กล้องให้ Terminal/Python

ปรับความยากที่ `SeqConfig` ใน `ongor/sequence_game.py` (start_len, max_len,
lives, step_timeout, hold_time ของ stabilizer ฯลฯ)

### ส่งค่าออก (event)

เกม emit `GameEvent` → ต่อ sink ได้หลายแบบพร้อมกัน (`MultiSink`):
`ROUND` · `SHOW` · `GO` · `OK` · `MISS` · `CLEAR` · `NEW` · `LIFE` · `SCORE` ·
`OVER` · `WIN` (ดูรายละเอียดใน `ongor/events.py` → `ArduinoSink`)

## ขั้นตอนใช้งาน (ตั้งแต่ต้น)

```bash
PY=/opt/anaconda3/bin/python

# 1. แปลงรูป dataset → keypoints CSV  (โฟลเดอร์ ../detaset/)
$PY dataset_to_csv.py

# 2. เทรน + export TFLite  (temporal split = honest accuracy)
$PY train_mediapipe.py --epochs 120 --split temporal --show-confusion

# 3. ตรวจ pipeline ด้วยรูปจริง  (ควรได้ ~99-100%, ไม่มีสับสนซ้าย/ขวา)
$PY verify_dataset.py

# 4. ทดสอบกล้องสด
$PY test_mediapipe.py

# 5. เล่นเกมเต็ม
$PY main.py --camera 0 --serial /dev/ttyACM0
```

ปุ่มในเกม: `SPACE`=เริ่ม  `Q`=หยุด  `ESC`=ออก

## ⚠️ เรื่อง flip (มิเรอร์) — สำคัญมาก

preprocessing ตอน inference **ต้องตรงกับตอนสร้าง CSV เป๊ะ** ไม่งั้นท่าซ้าย/ขวาสลับ
- `dataset_to_csv.py` อ่านรูป **ไม่ flip**
- ดังนั้น inference ทุกตัว default **`flip=False`**

พิสูจน์แล้ว: ถ้าเปิด `--flip` ผิด ท่า L/R สลับ 100% (`verify_dataset.py --flip` → 50%)
ถ้ากล้องจริงให้ภาพมิเรอร์มาเอง ค่อยเติม `--flip`

## โมเดลปัจจุบัน — 8 ท่า

| label | ภาษาไทย |
|---|---|
| prayHand | พนมมือ |
| thb_touch_heaad_both | แตะหัวสองมือ |
| thl_touch_head_l | แตะหัวมือซ้าย |
| thr_touch_head_R | แตะหัวมือขวา |
| hub_hand_up_Both | ยกมือสองข้าง |
| hul_hand_up_L | ยกมือซ้าย |
| hur_hand_up_R | ยกมือขวา |
| idle | อยู่นิ่ง (ไม่ใช้เป็นโจทย์) |

accuracy: **99.2%** (temporal split, honest) / 100% บนรูป train

## โปรโตคอลคุยกับฝั่ง .ino (115200 bps, บรรทัดละคำสั่ง)

Python → MCU: `POSE:<name>` · `SCORE:<int>` · `STATE:<idle|round|done>`
MCU → Python: `BTN:START` · `BTN:STOP`

## เพิ่ม/แก้ท่า

1. เพิ่มโฟลเดอร์รูปใน `../detaset/<ชื่อท่า>-samples/`
2. `$PY dataset_to_csv.py` (ลบ `data/keypoints.csv` เก่าก่อนถ้าจะเริ่มใหม่)
3. `$PY train_mediapipe.py` — label_map.json อัปเดตอัตโนมัติ
4. เพิ่มชื่อไทยใน `ongor/labels.py` → `THAI_NAMES`

---

### ไฟล์ pipeline เก่า (PoseNet/Teachable Machine)

`posenet_runner.py`, `classifier.py`, `classifier_tflite.py`, `convert_*.py`
ยังเก็บไว้แต่ไม่ใช้แล้ว — pipeline หลักคือ MediaPipe

---

## FastAPI Host (สำหรับ Front-end และอุปกรณ์อื่นเรียกใช้งาน)

แนวคิดที่ทำได้จริง:
- ให้ Linux บนฝั่ง Arduino Uno Q เป็น **AI/Game backend**
- เปิด API ด้วย FastAPI เพื่อให้ Web front-end, mobile app, หรือ dashboard เรียกได้
- ถ้าต้องคุย MCU ผ่าน serial ให้ backend ตัวเดียวเป็นตัวกลาง (ไม่ให้ front-end จับ serial โดยตรง)

โค้ด API อยู่ที่ `api/main.py`

### ติดตั้ง

```bash
PY=/opt/anaconda3/bin/python

$PY -m pip install -r requirements.txt
```

### รันเซิร์ฟเวอร์

```bash
PY=/opt/anaconda3/bin/python

# โหมดไม่ต่อ MCU
$PY -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# โหมดต่อ MCU ผ่าน serial
ONGOR_SERIAL_PORT=/dev/ttyACM0 $PY -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

เปิด docs ได้ที่:
- `http://<linux-ip>:8000/docs`

### API ที่สำคัญ

- `GET /health` ตรวจว่า service ยังทำงาน
- `GET /labels` ดูรายชื่อท่าที่โมเดลรองรับ
- `GET /game/state` ดูสถานะเกมปัจจุบัน
- `POST /game/start` เริ่มเกม
- `POST /game/stop` หยุดเกม
- `POST /game/confirm` ส่งผลยืนยันท่าจาก client (body: `{"label":"prayHand"}`)
- `POST /game/tick` ให้ server ประมวลผล timer ของเกม
- `POST /vision/predict` อัปโหลดรูป 1 เฟรมแล้วให้ server ทำนาย
- `WS /ws/events` รับ event เกมแบบ real-time

### ตัวอย่างเรียก API

```bash
# เริ่มเกม
curl -X POST http://127.0.0.1:8000/game/start

# ยืนยันว่าผู้เล่นทำท่า prayHand สำเร็จ
curl -X POST http://127.0.0.1:8000/game/confirm \
    -H "Content-Type: application/json" \
    -d '{"label":"prayHand"}'
```

---

## Front-end Integration TODO (Checklist)

ใช้ checklist นี้เป็นแผนงานฝั่งหน้าเว็บ:

1. สร้างไฟล์ config ของ API base URL
2. ทำ API client สำหรับ `health`, `labels`, `game/state`, `game/start`, `game/stop`, `game/confirm`
3. ทำ WebSocket client ต่อ `ws/events` (reconnect อัตโนมัติ)
4. หน้า Lobby: ปุ่ม Start/Stop + สถานะ online/offline
5. หน้า Game HUD: แสดง phase, score, round, time_left
6. แสดง event feed ล่าสุด (เช่น `show_pose`, `input_correct`, `game_over`)
7. ถ้าใช้กล้องบน browser: แคปภาพเป็น JPEG ส่ง `POST /vision/predict` ตามช่วงเวลา
8. ถ้าได้ `confirmed` จาก `/vision/predict` ให้เรียก `/game/confirm` ทันที
9. ทำ debounce/rate limit ฝั่ง client (เช่น 5-10 fps สำหรับส่งภาพ)
10. ทำ error UI สำหรับ timeout, disconnect, bad request
11. แยก environment (`dev`, `staging`, `prod`) สำหรับ API URL
12. ใส่ CORS origin ที่อนุญาตผ่าน env `CORS_ORIGINS`

ตัวอย่าง `CORS_ORIGINS`:

```bash
CORS_ORIGINS=http://localhost:5173,http://192.168.1.20:3000
```

---

## Arduino Uno Q + Linux Deployment Notes

สำหรับรันจริงบนอุปกรณ์:

1. ใช้ Linux box ที่มี Python + USB camera + serial ไป MCU
2. ผูก device path ให้คงที่ (เช่น udev rule ของ `/dev/ttyACM0`)
3. ทดสอบ serial ก่อนเปิด API (`screen /dev/ttyACM0 115200` หรือ script test)
4. รัน FastAPI เป็น systemd service เพื่อ auto restart
5. เปิด firewall เฉพาะ port ที่ต้องใช้ (เช่น 8000 ใน LAN)
6. ถ้าต้องเข้าจากภายนอก ให้มี reverse proxy + TLS (nginx/caddy)

ตัวอย่าง `systemd` service (ย่อ):

```ini
[Unit]
Description=Ong-Or FastAPI
After=network.target

[Service]
WorkingDirectory=/path/to/python_app
Environment=ONGOR_SERIAL_PORT=/dev/ttyACM0
Environment=CORS_ORIGINS=http://192.168.1.50:5173
ExecStart=/opt/anaconda3/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```
