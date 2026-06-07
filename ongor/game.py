"""
เกม อ่องออ (Ong-Or) — เบื้องต้น
รูปแบบ Simon-Says: ระบบสุ่มท่าโจทย์ ผู้เล่นทำท่าตามภายในเวลาที่กำหนด
ถ้า classifier ตรวจจับท่าถูกต้องด้วย confidence ผ่านเกณฑ์ + ค้างไว้พอ -> ได้คะแนน
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum

from .labels import GAME_POSES, THAI_NAMES, Prediction


class Phase(Enum):
    IDLE = "idle"
    PROMPT = "prompt"     # โชว์โจทย์
    PLAY = "play"         # ให้ผู้เล่นทำท่า
    RESULT = "result"     # แสดงผลรอบนี้
    DONE = "done"


@dataclass
class Round:
    target: str
    started_at: float
    hold_start: float | None = None
    success: bool = False


@dataclass
class GameConfig:
    rounds_total: int = 5
    play_seconds: float = 6.0          # เวลาในแต่ละรอบ
    prompt_seconds: float = 2.0        # เวลาแสดงโจทย์
    result_seconds: float = 1.5
    conf_threshold: float = 0.85       # confidence ขั้นต่ำ
    hold_seconds: float = 0.8          # ต้องค้างท่าต่อเนื่องเท่าไหร่


@dataclass
class GameState:
    config: GameConfig = field(default_factory=GameConfig)
    phase: Phase = Phase.IDLE
    score: int = 0
    round_idx: int = 0
    current: Round | None = None
    phase_started_at: float = 0.0


class OngOrGame:
    """state machine ของเกม"""

    def __init__(self, config: GameConfig | None = None) -> None:
        self.state = GameState(config=config or GameConfig())
        self._rng = random.Random()

    # ----- API ที่ main loop เรียกใช้ -----

    def start(self) -> None:
        s = self.state
        s.score = 0
        s.round_idx = 0
        s.current = None
        self._transition(Phase.PROMPT)
        self._new_round()

    def stop(self) -> None:
        self._transition(Phase.IDLE)

    def update(self, pred: Prediction | None) -> None:
        """เรียกทุก frame พร้อมผลทำนาย (None ได้ถ้ายังไม่มี)"""
        s = self.state
        now = time.time()
        elapsed = now - s.phase_started_at

        if s.phase is Phase.PROMPT:
            if elapsed >= s.config.prompt_seconds:
                self._transition(Phase.PLAY)
            return

        if s.phase is Phase.PLAY and s.current is not None:
            cur = s.current
            ok = (
                pred is not None
                and pred.label == cur.target
                and pred.confidence >= s.config.conf_threshold
            )
            if ok:
                if cur.hold_start is None:
                    cur.hold_start = now
                elif now - cur.hold_start >= s.config.hold_seconds:
                    cur.success = True
                    s.score += 1
                    self._transition(Phase.RESULT)
                    return
            else:
                cur.hold_start = None

            if elapsed >= s.config.play_seconds:
                self._transition(Phase.RESULT)
            return

        if s.phase is Phase.RESULT:
            if elapsed >= s.config.result_seconds:
                s.round_idx += 1
                if s.round_idx >= s.config.rounds_total:
                    self._transition(Phase.DONE)
                else:
                    self._new_round()
                    self._transition(Phase.PROMPT)
            return

    # ----- ภายใน -----

    def _transition(self, phase: Phase) -> None:
        self.state.phase = phase
        self.state.phase_started_at = time.time()

    def _new_round(self) -> None:
        # หลีกเลี่ยงสุ่มซ้ำท่าเดิมติดกัน
        prev = self.state.current.target if self.state.current else None
        choices = [p for p in GAME_POSES if p != prev]
        target = self._rng.choice(choices)
        self.state.current = Round(target=target, started_at=time.time())

    # ----- ข้อมูลสำหรับแสดงผล -----

    def hud_lines(self) -> list[str]:
        s = self.state
        lines = [f"Score: {s.score}   Round: {s.round_idx + 1}/{s.config.rounds_total}"]
        if s.phase is Phase.PROMPT and s.current:
            lines.append(f"ทำท่า: {THAI_NAMES.get(s.current.target, s.current.target)}")
        elif s.phase is Phase.PLAY and s.current:
            remain = max(0.0, s.config.play_seconds - (time.time() - s.phase_started_at))
            lines.append(
                f"ทำท่า: {THAI_NAMES.get(s.current.target, s.current.target)} "
                f"({remain:0.1f}s)"
            )
        elif s.phase is Phase.RESULT and s.current:
            lines.append("ผ่าน!" if s.current.success else "พลาด")
        elif s.phase is Phase.DONE:
            lines.append(f"จบเกม — คะแนน {s.score}/{s.config.rounds_total}")
        elif s.phase is Phase.IDLE:
            lines.append("กด SPACE เพื่อเริ่ม")
        return lines
