"""
Entry point — เกมอ่องออ (Simon-Says) บนฝั่ง Linux ของ Arduino Uno Q

  กล้อง → PoseEngine (MediaPipe + tflite + stabilizer) → SequenceGame → output

รัน:
  python main.py                          # เล่น (ถ้าไม่มี MCU จะ stub serial ให้)
  python main.py --serial /dev/ttyACM0    # ส่ง event ไป MCU
  python main.py --flip                   # ถ้ากล้องให้ภาพมิเรอร์
  python main.py --no-serial              # ไม่ต่อ serial เลย

ปุ่ม: SPACE=เริ่ม/เล่นใหม่  Q=หยุด  ESC=ออก
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ongor.events import MultiSink, print_sink, score_log_sink
from ongor.labels import en_of
from ongor.pose_engine import PoseEngine
from ongor.sequence_game import SeqConfig, SequenceGame

_mp_draw = mp.solutions.drawing_utils
_mp_pose = mp.solutions.pose


def list_cameras(max_idx: int = 6) -> list[int]:
    """สแกนหา index กล้องที่เปิดได้จริง (0..max_idx-1)"""
    found = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found


def open_camera(idx: int):
    """เปิดกล้อง + ตั้งความละเอียด คืน cap (อาจ isOpened()=False ถ้าเปิดไม่ได้)"""
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def _draw(display, game, engine, label, conf, now) -> None:
    if engine.last_result and engine.last_result.landmarks is not None:
        _mp_draw.draw_landmarks(
            display, engine.last_result.landmarks, _mp_pose.POSE_CONNECTIONS,
            _mp_draw.DrawingSpec((0, 255, 0), 2, 2),
            _mp_draw.DrawingSpec((255, 255, 255), 1, 1),
        )
    for i, line in enumerate(game.hud_lines(now)):
        cv2.putText(display, line, (12, 32 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    if label:
        bar = int(engine.stabilizer.hold_progress(now) * 200)
        cv2.rectangle(display, (12, display.shape[0] - 40),
                      (12 + bar, display.shape[0] - 24), (0, 255, 0), -1)
        cv2.putText(display, f"{en_of(label)} {conf * 100:.0f}%",
                    (12, display.shape[0] - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def play(
    camera: int = 0,
    flip: bool = False,
    serial: str | None = None,
    config: SeqConfig | None = None,
    show: bool = True,
) -> int:
    """
    ลูปเล่นเกมหลัก — ใช้ซ้ำได้ทั้งจาก main และ test_game
    serial: path เช่น "/dev/ttyACM0" เพื่อส่ง event ไป MCU (None = ไม่ส่ง)
    """
    sink = MultiSink(print_sink, score_log_sink())  # log คะแนน + timestamp ลงไฟล์
    link = None
    if serial:
        from ongor.arduino_link import ArduinoLink
        from ongor.events import ArduinoSink
        link = ArduinoLink(port=serial)
        link.open()
        sink.add(ArduinoSink(link))

    engine = PoseEngine(flip=flip)
    game = SequenceGame(config or SeqConfig(), on_event=sink)

    cam_idx = camera
    cap = open_camera(cam_idx)
    if not cap.isOpened():
        print(f"[err] open camera {cam_idx} failed")
        return 1

    print(f"[run] camera={cam_idx} | SPACE=start/retry  Q=stop  "
          f"0-9=switch camera  ESC=quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            now = time.time()

            res, pred = engine.process(frame)
            label = pred.label if pred else None
            conf = pred.confidence if pred else 0.0

            confirmed = engine.stabilizer.update(label, conf, now=now)
            if confirmed:
                game.on_confirm(confirmed, now=now)
            game.tick(now=now)

            # รับปุ่มจาก MCU (ถ้ามี)
            if link:
                evt = link.poll_event()
                if evt and evt.name == "BTN" and evt.value == "START":
                    engine.stabilizer.reset()
                    game.start(now=now)

            if show:
                _draw(res.frame, game, engine, label, conf, now)
                cv2.putText(res.frame, f"cam {cam_idx}",
                            (res.frame.shape[1] - 110, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.imshow("Ong-Or — Sequence Game", res.frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord(" "):
                    engine.stabilizer.reset()
                    game.start(now=now)
                elif key == ord("q"):
                    game.stop(now=now)
                elif ord("0") <= key <= ord("9"):
                    # สลับกล้องตอนรัน
                    new_idx = key - ord("0")
                    if new_idx != cam_idx:
                        new_cap = open_camera(new_idx)
                        if new_cap.isOpened() and new_cap.read()[0]:
                            cap.release()
                            cap = new_cap
                            cam_idx = new_idx
                            print(f"[run] switched to camera {cam_idx}")
                        else:
                            new_cap.release()
                            print(f"[run] camera {new_idx} not available")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.close()
        if link:
            link.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--list-cameras", action="store_true",
                    help="สแกนกล้องที่เปิดได้แล้วออก")
    ap.add_argument("--serial", default="/dev/ttyACM0",
                    help="serial port ไป MCU (ตั้ง --no-serial เพื่อไม่ต่อ)")
    ap.add_argument("--no-serial", action="store_true")
    ap.add_argument("--flip", action="store_true",
                    help="มิเรอร์เฟรม (default ไม่มิเรอร์ ให้ตรงกับตอนเทรน)")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    if args.list_cameras:
        cams = list_cameras()
        print(f"กล้องที่ใช้ได้: {cams}" if cams else "ไม่พบกล้อง")
        return 0

    serial = None if args.no_serial else args.serial
    return play(camera=args.camera, flip=args.flip, serial=serial,
                show=not args.no_display)


if __name__ == "__main__":
    sys.exit(main())
