"""
ทดสอบเกมลำดับ (Simon-Says)

โหมด logic (ไม่ใช้กล้อง — เทสตรรกะเกมล้วน ๆ รันได้ทุก python):
    python test_game.py --logic

โหมดเล่นจริงด้วยกล้อง (ต้องมี mediapipe + โมเดล):
    /opt/anaconda3/bin/python test_game.py
    /opt/anaconda3/bin/python test_game.py --flip --serial /dev/ttyACM0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ให้ import ทั้ง ongor (จาก python_app) และ main (จากโฟลเดอร์ game) ได้
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # python_app
sys.path.insert(0, str(Path(__file__).resolve().parent))         # game/


# ======================================================================
# โหมด LOGIC — จำลองการเล่นโดยฉีดเวลาเอง ไม่ต้องมีกล้อง/mediapipe
# ======================================================================
def run_logic() -> int:
    from ongor.sequence_game import SequenceGame, SeqConfig, Phase
    from ongor.events import GameEvent
    from ongor.labels import GAME_POSES

    log: list[GameEvent] = []
    cfg = SeqConfig(start_len=1, max_len=3, memorize_time=0.5,
                    step_timeout=5.0, clear_pause=0.3)
    game = SequenceGame(cfg, on_event=log.append, rng_seed=7)

    t = 0.0

    def advance(dt: float):
        nonlocal t
        steps = max(1, int(dt / 0.05))
        for _ in range(steps):
            t += 0.05
            game.tick(now=t)

    print("=== TEST 1: เล่นเพอร์เฟกต์จนชนะ (ทำท่าเดิมทั้งหมด+ท่าใหม่) ===")
    game.start(now=t)
    assert game.state.phase is Phase.SHOW, game.state.phase

    safety = 0
    while game.state.phase is not Phase.WIN:
        safety += 1
        assert safety < 100, "วนไม่จบ — น่าจะมีบั๊ก"
        advance(0.6)  # รอจบเฟสโชว์ (memorize 0.5s) เข้าสู่ INPUT
        if game.state.phase is Phase.INPUT:
            # ต้องทำท่าเดิมทั้งหมดตามลำดับ แล้วจบด้วยท่าใหม่
            for pose in list(game.state.sequence):
                game.on_confirm(pose, now=t)
            advance(cfg.clear_pause + 0.2)

    win = [e for e in log if e.type == "win"]
    assert win, "ควรมี event win"
    assert win[0].data.get("duration") is not None, "win ควรมี duration"
    print(f"  ✅ ชนะ คะแนน={win[0].data['score']}  duration={win[0].data['duration']}s")

    print("\n=== TEST 2: ทำผิด -> game over ทันที + มี score/timestamp ===")
    log.clear()
    game2 = SequenceGame(SeqConfig(start_len=1, memorize_time=0.3, step_timeout=5.0),
                         on_event=log.append, rng_seed=3)
    t = 0.0
    game2.start(now=t)
    while game2.state.phase is not Phase.INPUT:
        t += 0.05
        game2.tick(now=t)
    wrong = next(p for p in GAME_POSES if p != game2.state.sequence[0])
    game2.on_confirm(wrong, now=t)
    assert game2.state.phase is Phase.GAME_OVER, game2.state.phase
    over = [e for e in log if e.type == "game_over"]
    assert over and over[0].data.get("score") is not None
    assert over[0].ts > 0, "event ต้องมี timestamp"
    print(f"  ✅ ทำผิด -> game_over score={over[0].data['score']} เวลา={over[0].clock}")

    print("\n=== TEST 3: รีเซ็ตเวลา 10วิเมื่อทำถูก + หมดเวลา = game over ===")
    log.clear()
    game3 = SequenceGame(SeqConfig(start_len=2, memorize_time=0.2, step_timeout=2.0),
                         on_event=log.append, rng_seed=1)
    t = 0.0
    game3.start(now=t)
    while game3.state.phase is not Phase.INPUT:
        t += 0.05
        game3.tick(now=t)
    # ทำถูกท่าแรก -> เวลาควรรีเซ็ตเป็น 2.0s
    first = game3.state.sequence[0]
    game3.on_confirm(first, now=t)
    deadline_after = game3.state.step_deadline
    assert abs(deadline_after - (t + 2.0)) < 1e-6, "ทำถูกต้องรีเซ็ต deadline"
    # แล้วปล่อยให้ท่าที่ 2 หมดเวลา
    for _ in range(int(2.5 / 0.05)):
        t += 0.05
        game3.tick(now=t)
    assert game3.state.phase is Phase.GAME_OVER
    miss = [e for e in log if e.type == "input_wrong" and e.data.get("got") == "timeout"]
    assert miss, "ควรมี input_wrong จาก timeout"
    print("  ✅ ทำถูกรีเซ็ตเวลา / หมดเวลาแล้ว game over ถูกต้อง")

    print("\nผ่านทั้งหมด ✅")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logic", action="store_true", help="เทสตรรกะเกม ไม่ใช้กล้อง")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--serial", default=None, help="เช่น /dev/ttyACM0 (ถ้าจะส่งไป MCU)")
    args = ap.parse_args()
    if args.logic:
        return run_logic()
    # โหมดเล่นจริง — ใช้ลูปเดียวกับ main.py (ไม่เขียนซ้ำ)
    from main import play
    return play(camera=args.camera, flip=args.flip, serial=args.serial)


if __name__ == "__main__":
    sys.exit(main())
