"""
สื่อสารกับฝั่ง MCU (.ino) ของ Arduino Uno Q ผ่าน serial
ใช้โปรโตคอลข้อความง่าย ๆ บรรทัดละ 1 คำสั่ง:

  จาก Python -> MCU:
    POSE:<pose_name>        ส่งโจทย์ท่า (ให้ MCU โชว์ไฟ/จอ)
    SCORE:<int>             แจ้งคะแนนรวม
    STATE:<idle|round|done> สถานะเกม

  จาก MCU -> Python:
    BTN:START               กดปุ่มเริ่มเกม
    BTN:STOP                กดปุ่มหยุด
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

try:
    import serial  # type: ignore
except ImportError:
    serial = None  # type: ignore


@dataclass
class ArduinoEvent:
    name: str       # เช่น "BTN"
    value: str      # เช่น "START"


class ArduinoLink:
    """wrapper รอบ pyserial — รัน reader thread, ส่ง/รับเป็นบรรทัด"""

    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 115200) -> None:
        self.port = port
        self.baud = baud
        self._ser: Optional["serial.Serial"] = None
        self._events: "queue.Queue[ArduinoEvent]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def open(self) -> bool:
        if serial is None:
            print("[arduino] pyserial ไม่ติดตั้ง — รันโหมด stub")
            return False
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(1.5)  # รอ MCU reset
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
            print(f"[arduino] เชื่อมต่อ {self.port} @ {self.baud}")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[arduino] เปิด {self.port} ไม่สำเร็จ: {e} — โหมด stub")
            self._ser = None
            return False

    def close(self) -> None:
        self._stop.set()
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:  # noqa: BLE001
                pass
            self._ser = None

    def send(self, line: str) -> None:
        if self._ser is None:
            print(f"[arduino:stub] -> {line}")
            return
        try:
            self._ser.write((line + "\n").encode("utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[arduino] write fail: {e}")

    def poll_event(self) -> Optional[ArduinoEvent]:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    # ---- helpers สำหรับเกม ----
    def send_pose(self, pose: str) -> None:
        self.send(f"POSE:{pose}")

    def send_score(self, score: int) -> None:
        self.send(f"SCORE:{score}")

    def send_state(self, state: str) -> None:
        self.send(f"STATE:{state}")

    def _reader(self) -> None:
        assert self._ser is not None
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(64)
            except Exception:  # noqa: BLE001
                break
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("utf-8", errors="ignore").strip()
                if not text or ":" not in text:
                    continue
                name, _, value = text.partition(":")
                self._events.put(ArduinoEvent(name=name, value=value))
