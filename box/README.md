# Ong-Or Box — แอป Arduino App Lab (Uno Q)

กล่องเกม "ความจำท่าทาง" ที่รวม 2 สมองของ Uno Q เข้าด้วยกัน:

```
[USB Webcam] ──► box/main.py (Python, ฝั่ง Linux)
                   │  • อ่านกล้อง
                   │  • เรียก AI ผ่าน HTTP  ──►  Ong-Or Pose API (/predict, /labels)
                   │  • รัน LOGIC เกมทั้งหมด
                   │
                   └─ Bridge ──►  ongor_box.ino (C, ฝั่ง MCU STM32)
                                    • LCD 16x2, Knob, Pixels x8, Buzzer, Distance
                                    • แค่ "แสดงผล" ตามที่ Python สั่ง
```

> **logic เกมอยู่ฝั่ง Python ทั้งหมด** — ฝั่ง C เป็นแค่ I/O ให้ผู้เล่น

## ไฟล์
| ไฟล์ | ฝั่ง | หน้าที่ |
|------|------|---------|
| [`main.py`](main.py) | Linux (Python) | กล้อง + เรียก AI + logic เกม + คุม MCU |
| [`ongor_box.ino`](ongor_box.ino) | MCU (C) | เมนู/คาลิเบรต/จอ/ไฟ/เสียง |

## กติกาเกม (โหมด 1: Play Game)
1. โชว์ **ท่าใหม่ 1 ท่า** ให้จำ **5 วินาที**
2. ผู้เล่นต้องทำ **ท่าเดิมทั้งหมดตามลำดับจากต้น** แล้วจบด้วยท่าใหม่
3. ทำครบลำดับถูกต้อง → **+คะแนน** แล้วเพิ่มท่าใหม่ 1 ท่า (ลำดับยาวขึ้นเรื่อย ๆ)
4. ทำผิด หรือ หมดเวลา (10 วิ/ท่า) → **GAME OVER**

โหมดอื่น: **2 = Test AI** (โชว์ท่าที่ AI เห็นสด ๆ), **3 = Cam Check** (เช็คกล้อง/FPS)

## โปรโตคอล Bridge (Python ↔ MCU)
- `set_mode(m)` — MCU บอก Python ว่าเลือกโหมดไหน (`1/2/3`, `0`=กลับเมนู)
- `poll()` — MCU ขอข้อความแสดงผล คืนสตริง `l1|l2|pix|buz`
  - `l1`,`l2` = 2 บรรทัดบน LCD (≤16 ตัวอักษร, ASCII)
  - `pix` = จำนวนไฟติด 0–8
  - `buz` = เสียง: `1`=tick/โชว์ `2`=ถูก `3`=ชนะ `4`=ผิด/แพ้ (one-shot)

## ติดตั้ง / รัน
1. รัน **Ong-Or Pose API** ให้ได้ก่อน (ดู [`../INSTALL_ARDUINO_UNO_Q.md`](../INSTALL_ARDUINO_UNO_Q.md))
2. นำ `main.py` + `ongor_box.ino` เข้าโปรเจกต์ Arduino App Lab แล้ว Run
3. ค่าปรับได้ผ่าน env (ตัวอย่าง): `ONGOR_API`, `ONGOR_CAM`, `ONGOR_FPS`,
   `ONGOR_MEMORIZE` (วินาทีโชว์ท่า), `ONGOR_STEP_TIMEOUT` (วินาทีต่อท่า), `ONGOR_MAX_LEN`

> ปกติ `main.py` หา API เองผ่าน docker gateway/localhost ถ้าไม่เจอให้กำหนด
> `ONGOR_API=http://<ip-host>:8000` ตรง ๆ
