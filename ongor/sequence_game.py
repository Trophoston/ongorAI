"""
เกมอ่องออ (Ong-Or) — เกมความจำท่าทาง

กติกา:
  1. โชว์ "ท่าใหม่" 1 ท่า ให้จำ  (memorize_time = 5 วินาที)
  2. ครบเวลา → ผู้เล่นต้องทำ "ท่าเดิมทั้งหมดตามลำดับ" แล้วจบด้วยท่าใหม่
  3. ทำถูกแต่ละท่า → รีเซ็ตเวลาทำเป็น step_timeout (10 วินาที)
  4. ทำครบทั้งลำดับ → เพิ่มท่าใหม่ 1 ท่า กลับไปข้อ 1 (ลำดับยาวขึ้น)
  5. เกินเวลา (10 วิ) หรือทำผิด → GAME OVER + เก็บคะแนน

ออกแบบให้เทสได้: ทุก timing รับ now เข้ามาได้ (ฉีดเวลาเองตอนเทสได้)
ส่งค่าออก: ผ่าน on_event(GameEvent) — ต่อ serial/print/score-log ได้
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum

from . import events as ev
from .events import GameEvent
from .labels import GAME_POSES, en_of


class Phase(Enum):
    IDLE = "idle"            # ยังไม่เริ่ม
    SHOW = "show"            # โชว์ท่าใหม่ให้จำ
    INPUT = "input"          # ตาผู้เล่นทำตามลำดับทั้งหมด
    ROUND_CLEAR = "clear"    # ผ่านรอบ พักสั้น ๆ ก่อนเพิ่มท่าใหม่
    GAME_OVER = "over"       # แพ้ (ทำผิด/หมดเวลา)
    WIN = "win"              # ชนะ (ครบ max_len)


@dataclass
class SeqConfig:
    start_len: int = 1          # จำนวนท่าเริ่มต้น
    max_len: int = 50           # ลำดับยาวถึงเท่านี้ = ชนะ (อ่องออเล่นยาว)
    memorize_time: float = 5.0  # โชว์ท่าใหม่ให้จำกี่วินาที
    step_timeout: float = 10.0  # เวลาทำต่อ 1 ท่า (รีเซ็ตเมื่อทำถูก)
    clear_pause: float = 0.8    # พักหลังผ่านรอบ ก่อนโชว์ท่าใหม่
    wrong_is_fail: bool = True  # ทำผิด = แพ้ทันที (โหมดความจำ)


@dataclass
class SeqState:
    phase: Phase = Phase.IDLE
    phase_since: float = 0.0
    sequence: list[str] = field(default_factory=list)
    input_idx: int = 0          # ทำถึงท่าที่เท่าไรในลำดับ
    score: int = 0
    round_no: int = 0
    step_deadline: float = 0.0  # เส้นตายของท่าปัจจุบัน (เฟส INPUT)


class SequenceGame:
    """state machine ของเกมอ่องออ — ขับด้วย on_confirm() + tick()"""

    def __init__(
        self,
        config: SeqConfig | None = None,
        on_event=None,
        rng_seed: int | None = None,
    ) -> None:
        self.cfg = config or SeqConfig()
        self.on_event = on_event  # callable(GameEvent) | None
        self.state = SeqState()
        self._rng = random.Random(rng_seed)
        self._wall_start = 0.0    # เวลานาฬิกาจริงตอนเริ่มเกม (ไว้คิด duration)

    # ---------- API หลัก ----------

    def start(self, now: float | None = None) -> None:
        now = self._now(now)
        self._wall_start = time.time()
        s = self.state
        s.score = 0
        s.round_no = 0
        s.sequence = []
        for _ in range(self.cfg.start_len):
            self._append_pose()
        self._begin_round(now, first=True)

    def stop(self, now: float | None = None) -> None:
        self._enter(Phase.IDLE, self._now(now))

    def on_confirm(self, label: str, now: float | None = None) -> None:
        """เรียกเมื่อ stabilizer ยืนยันท่า 1 ท่า (มีผลเฉพาะเฟส INPUT)"""
        now = self._now(now)
        s = self.state
        if s.phase is not Phase.INPUT:
            return

        expected = s.sequence[s.input_idx]
        if label == expected:
            s.input_idx += 1
            s.step_deadline = now + self.cfg.step_timeout   # รีเซ็ตเวลา 10 วิ
            self._emit(ev.INPUT_CORRECT, index=s.input_idx - 1, pose=label,
                       total=len(s.sequence))
            if s.input_idx >= len(s.sequence):
                self._round_cleared(now)
        else:
            self._emit(ev.INPUT_WRONG, expected=expected, got=label,
                       index=s.input_idx)
            if self.cfg.wrong_is_fail:
                self._game_over(now)

    def tick(self, now: float | None = None) -> None:
        """เรียกทุกเฟรม — จัดการเวลา (โชว์ท่า / หมดเวลา / พักรอบ)"""
        now = self._now(now)
        s = self.state

        if s.phase is Phase.SHOW:
            if now - s.phase_since >= self.cfg.memorize_time:
                self._enter(Phase.INPUT, now)
                s.input_idx = 0
                s.step_deadline = now + self.cfg.step_timeout
                self._emit(ev.INPUT_START, sequence=list(s.sequence))

        elif s.phase is Phase.INPUT:
            if now >= s.step_deadline:
                self._emit(ev.INPUT_WRONG, expected=s.sequence[s.input_idx],
                           got="timeout", index=s.input_idx)
                self._game_over(now)

        elif s.phase is Phase.ROUND_CLEAR:
            if now - s.phase_since >= self.cfg.clear_pause:
                self._append_pose()
                self._begin_round(now)

    # ---------- ภายใน ----------

    def _begin_round(self, now: float, first: bool = False) -> None:
        """เข้าสู่เฟสโชว์ "ท่าใหม่" (ท่าล่าสุดในลำดับ) ให้จำ"""
        s = self.state
        s.round_no = len(s.sequence)
        new_pose = s.sequence[-1]
        if not first:
            self._emit(ev.NEW_POSE, pose=new_pose)
        self._emit(ev.ROUND_START, round=s.round_no, sequence=list(s.sequence))
        self._enter(Phase.SHOW, now)
        # โชว์เฉพาะท่าใหม่ (ท่าก่อนหน้าผู้เล่นจำได้แล้ว)
        self._emit(ev.SHOW_POSE, index=len(s.sequence) - 1, pose=new_pose,
                   total=len(s.sequence))

    def _round_cleared(self, now: float) -> None:
        s = self.state
        s.score += len(s.sequence)
        self._emit(ev.SCORE, score=s.score)
        self._emit(ev.ROUND_CLEAR, round=s.round_no, score=s.score)
        if len(s.sequence) >= self.cfg.max_len:
            self._enter(Phase.WIN, now)
            self._emit(ev.WIN, score=s.score, round=s.round_no,
                       duration=self._duration())
        else:
            self._enter(Phase.ROUND_CLEAR, now)

    def _game_over(self, now: float) -> None:
        self._enter(Phase.GAME_OVER, now)
        self._emit(ev.GAME_OVER, score=self.state.score,
                   round=self.state.round_no, duration=self._duration())

    def _append_pose(self) -> None:
        prev = self.state.sequence[-1] if self.state.sequence else None
        choices = [p for p in GAME_POSES if p != prev]  # กันซ้ำท่าติดกัน
        self.state.sequence.append(self._rng.choice(choices))

    def _enter(self, phase: Phase, now: float) -> None:
        self.state.phase = phase
        self.state.phase_since = now

    def _emit(self, etype: str, **data) -> None:
        if self.on_event is not None:
            self.on_event(GameEvent(etype, data))

    def _duration(self) -> float:
        return round(time.time() - self._wall_start, 2) if self._wall_start else 0.0

    @staticmethod
    def _now(now: float | None) -> float:
        return now if now is not None else time.time()

    # ---------- ข้อมูลสำหรับแสดงผล ----------

    def expected_pose(self) -> str | None:
        s = self.state
        if s.phase is Phase.INPUT and s.input_idx < len(s.sequence):
            return s.sequence[s.input_idx]
        return None

    def showing_pose(self) -> str | None:
        s = self.state
        if s.phase is Phase.SHOW and s.sequence:
            return s.sequence[-1]
        return None

    def time_left(self, now: float | None = None) -> float:
        """เวลาที่เหลือของเฟสปัจจุบัน (SHOW=นับถอยจาก memorize, INPUT=ถึง deadline)"""
        now = self._now(now)
        s = self.state
        if s.phase is Phase.SHOW:
            return max(0.0, self.cfg.memorize_time - (now - s.phase_since))
        if s.phase is Phase.INPUT:
            return max(0.0, s.step_deadline - now)
        return 0.0

    def hud_lines(self, now: float | None = None) -> list[str]:
        """ข้อความ HUD ภาษาอังกฤษ (cv2.putText รองรับแค่ ASCII)"""
        s = self.state
        head = f"Round {s.round_no}  Len {len(s.sequence)}  Score {s.score}"
        if s.phase is Phase.IDLE:
            return [head, "Press SPACE to start"]
        if s.phase is Phase.SHOW:
            pose = self.showing_pose() or ""
            return [head, f"MEMORIZE: {en_of(pose)}",
                    f"starts in {self.time_left(now):.0f}s"]
        if s.phase is Phase.INPUT:
            pose = self.expected_pose() or ""
            return [head,
                    f"Do {s.input_idx + 1}/{len(s.sequence)}: {en_of(pose)}",
                    f"time {self.time_left(now):.1f}s"]
        if s.phase is Phase.ROUND_CLEAR:
            return [head, "CORRECT! next pose..."]
        if s.phase is Phase.WIN:
            return [head, f"YOU WIN!  Score {s.score}"]
        if s.phase is Phase.GAME_OVER:
            return [head, f"GAME OVER  Score {s.score}", "SPACE to retry"]
        return [head]
