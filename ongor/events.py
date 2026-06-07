"""
ระบบ event/output — ให้เกม "ส่งค่าออก" ไปที่ไหนก็ได้

แนวคิด: เกม emit GameEvent → sink รับไปทำอะไรต่อ
  - print_sink         พิมพ์ลง terminal (ดีบัก)
  - ArduinoSink        แปลง event เป็นบรรทัดส่งไป MCU ผ่าน serial
  - MultiSink          กระจายไปหลาย sink พร้อมกัน
  - callback ธรรมดา    ฟังก์ชัน f(event) อะไรก็ได้

ถ้าไม่อยากส่งค่าออก ก็ไม่ต้องใส่ sink เกมก็เล่นได้ปกติ
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# ประเภท event ที่เกม emit (ใช้ค่าคงที่กันพิมพ์ผิด)
ROUND_START = "round_start"     # เริ่มรอบใหม่   data: round, sequence
SHOW_POSE = "show_pose"         # โชว์ท่าในลำดับ data: index, pose, total
INPUT_START = "input_start"     # ถึงตาผู้เล่นทำ data: sequence
INPUT_CORRECT = "input_correct" # ทำถูก 1 ท่า   data: index, pose, total
INPUT_WRONG = "input_wrong"     # ทำผิด         data: expected, got, index
ROUND_CLEAR = "round_clear"     # ผ่านรอบ       data: round, score
NEW_POSE = "new_pose"           # ท่าใหม่ที่เพิ่ม data: pose
LIFE_LOST = "life_lost"         # เสียชีวิต      data: lives
SCORE = "score"                 # คะแนนเปลี่ยน   data: score
GAME_OVER = "game_over"         # จบเกม(แพ้)    data: score, round
WIN = "win"                     # ชนะครบทุกรอบ  data: score


@dataclass
class GameEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)  # epoch wall-clock ตอน emit

    @property
    def clock(self) -> str:
        """เวลาแบบ HH:MM:SS.mmm"""
        return datetime.fromtimestamp(self.ts).strftime("%H:%M:%S.") + \
            f"{int((self.ts % 1) * 1000):03d}"

    @property
    def iso(self) -> str:
        return datetime.fromtimestamp(self.ts).isoformat(timespec="milliseconds")

    def __str__(self) -> str:
        body = " ".join(f"{k}={v}" for k, v in self.data.items())
        return f"<{self.type}> {body}".rstrip()


# sink = ฟังก์ชันรับ GameEvent
EventSink = Callable[[GameEvent], None]


def print_sink(event: GameEvent) -> None:
    """พิมพ์ event ลง terminal พร้อม timestamp"""
    print(f"[{event.clock}] {event}")


def score_log_sink(path: str | Path | None = None) -> EventSink:
    """
    sink สำหรับบันทึกคะแนนลงไฟล์ (JSONL) ตอนจบเกม
    บันทึกเมื่อ event เป็น game_over หรือ win — 1 บรรทัด/เกม
    default = python_app/logs/scores.jsonl (อ้างจาก ongor.paths ไม่ขึ้นกับ CWD)

    แต่ละบรรทัด: {ts, time, result, score, round, duration_sec}
    """
    if path is None:
        from .paths import SCORES_LOG
        p = SCORES_LOG
    else:
        p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def sink(e: GameEvent) -> None:
        if e.type not in (GAME_OVER, WIN):
            return
        rec = {
            "ts": round(e.ts, 3),
            "time": e.iso,
            "result": e.type,
            "score": e.data.get("score"),
            "round": e.data.get("round"),
            "duration_sec": e.data.get("duration"),
        }
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[score] บันทึก -> {p}: {rec}")

    return sink


class MultiSink:
    """กระจาย event ไปหลาย sink"""

    def __init__(self, *sinks: EventSink) -> None:
        self._sinks = [s for s in sinks if s is not None]

    def add(self, sink: EventSink) -> None:
        self._sinks.append(sink)

    def __call__(self, event: GameEvent) -> None:
        for s in self._sinks:
            try:
                s(event)
            except Exception as e:  # noqa: BLE001
                print(f"[event] sink error: {e}")


class ArduinoSink:
    """
    แปลง GameEvent → บรรทัดข้อความส่งไป MCU ผ่าน ArduinoLink
    โปรโตคอล (บรรทัดละคำสั่ง, ต่อท้ายด้วย \\n):

      ROUND:<n>:<len>     เริ่มรอบ n, ลำดับยาว len
      SHOW:<pose>:<i>     โชว์ท่าที่ i ในลำดับ
      GO                  ถึงตาผู้เล่น
      OK:<pose>:<i>       ทำถูกท่าที่ i
      MISS:<exp>:<got>    ทำผิด
      CLEAR:<round>       ผ่านรอบ
      NEW:<pose>          ท่าใหม่
      LIFE:<n>            ชีวิตที่เหลือ
      SCORE:<n>           คะแนน
      OVER:<n>            จบเกม คะแนน n
      WIN:<n>             ชนะ คะแนน n
    """

    def __init__(self, link: Any) -> None:
        self._link = link

    def _send(self, line: str) -> None:
        try:
            self._link.send(line)
        except Exception as e:  # noqa: BLE001
            print(f"[ArduinoSink] ส่งไม่สำเร็จ: {e}")

    def __call__(self, e: GameEvent) -> None:
        d = e.data
        if e.type == ROUND_START:
            self._send(f"ROUND:{d.get('round')}:{len(d.get('sequence', []))}")
        elif e.type == SHOW_POSE:
            self._send(f"SHOW:{d.get('pose')}:{d.get('index')}")
        elif e.type == INPUT_START:
            self._send("GO")
        elif e.type == INPUT_CORRECT:
            self._send(f"OK:{d.get('pose')}:{d.get('index')}")
        elif e.type == INPUT_WRONG:
            self._send(f"MISS:{d.get('expected')}:{d.get('got')}")
        elif e.type == ROUND_CLEAR:
            self._send(f"CLEAR:{d.get('round')}")
        elif e.type == NEW_POSE:
            self._send(f"NEW:{d.get('pose')}")
        elif e.type == LIFE_LOST:
            self._send(f"LIFE:{d.get('lives')}")
        elif e.type == SCORE:
            self._send(f"SCORE:{d.get('score')}")
        elif e.type == GAME_OVER:
            self._send(f"OVER:{d.get('score')}")
        elif e.type == WIN:
            self._send(f"WIN:{d.get('score')}")
