from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ongor.events import ArduinoSink, GameEvent, MultiSink, print_sink, score_log_sink
from ongor.labels import LABELS
from ongor.sequence_game import SeqConfig, SequenceGame

# Optional vision/pose imports (may not be available on all boards)
HAS_VISION = False
PoseEngine = None
try:
    import cv2
    import numpy as np
    from ongor.pose_engine import PoseEngine
    HAS_VISION = True
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore


class ConfirmBody(BaseModel):
    label: str


class FastApiGameBridge:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._game = SequenceGame(SeqConfig(), on_event=self._on_game_event)
        self._events: list[dict[str, Any]] = []
        self._event_id = 0
        self._sinks = MultiSink(print_sink, score_log_sink())
        self._link = None
        self._engine: PoseEngine | None = None
        self._engine_lock = threading.Lock()
        self._tick_hz = max(1.0, float(os.getenv("GAME_TICK_HZ", "20")))
        self._stop_event = threading.Event()
        self._loop_thread: threading.Thread | None = None

        serial_port = os.getenv("ONGOR_SERIAL_PORT", "").strip()
        if serial_port:
            try:
                from ongor.arduino_link import ArduinoLink

                self._link = ArduinoLink(port=serial_port)
                ok = self._link.open()
                if ok:
                    self._sinks.add(ArduinoSink(self._link))
                    print(f"[api] serial connected: {serial_port}")
                else:
                    print(f"[api] serial stub mode: {serial_port}")
            except Exception as e:  # noqa: BLE001
                print(f"[api] serial init failed: {e}")

    def close(self) -> None:
        self.stop_loop()
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        if self._link is not None:
            self._link.close()
            self._link = None

    def start_loop(self) -> None:
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self._stop_event.clear()
        self._loop_thread = threading.Thread(target=self._loop_worker, daemon=True)
        self._loop_thread.start()
        print(f"[api] game loop started @ {self._tick_hz:.1f} Hz")

    def stop_loop(self) -> None:
        self._stop_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=1.5)
            self._loop_thread = None

    def _loop_worker(self) -> None:
        period = 1.0 / self._tick_hz
        while not self._stop_event.is_set():
            t0 = time.time()
            with self._lock:
                self._poll_link_locked()
                self._game.tick(now=t0)
            dt = time.time() - t0
            sleep_for = period - dt
            if sleep_for > 0:
                time.sleep(sleep_for)

    def labels(self) -> list[str]:
        return list(LABELS)

    def state(self) -> dict[str, Any]:
        with self._lock:
            self._poll_link_locked()
            s = self._game.state
            return {
                "phase": s.phase.value,
                "score": s.score,
                "round": s.round_no,
                "sequence_length": len(s.sequence),
                "next_expected": self._game.expected_pose(),
                "showing": self._game.showing_pose(),
                "time_left": round(self._game.time_left(), 2),
            }

    def start(self) -> dict[str, Any]:
        with self._lock:
            self._poll_link_locked()
            self._game.start()
            return self.state()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._poll_link_locked()
            self._game.stop()
            return self.state()

    def confirm(self, label: str) -> dict[str, Any]:
        if label not in LABELS:
            raise HTTPException(status_code=400, detail=f"unknown label: {label}")
        with self._lock:
            self._poll_link_locked()
            self._game.on_confirm(label)
            self._game.tick()
            return self.state()

    def tick(self) -> dict[str, Any]:
        with self._lock:
            self._poll_link_locked()
            self._game.tick()
            return self.state()

    async def stream_events(self, ws: WebSocket) -> None:
        await ws.accept()
        cursor = 0
        try:
            while True:
                await asyncio.sleep(0.2)
                payloads: list[dict[str, Any]] = []
                with self._lock:
                    if self._event_id > cursor and self._events:
                        payloads = [e for e in self._events if e["id"] > cursor]
                for payload in payloads:
                    await ws.send_text(json.dumps(payload, ensure_ascii=False))
                    cursor = payload["id"]
        except WebSocketDisconnect:
            return

    def _on_game_event(self, event: GameEvent) -> None:
        self._sinks(event)
        with self._lock:
            self._event_id += 1
            payload = {
                "id": self._event_id,
                "type": event.type,
                "data": event.data,
                "ts": event.ts,
                "iso": event.iso,
            }
            self._events.append(payload)
            if len(self._events) > 200:
                self._events = self._events[-200:]

    def _poll_link_locked(self) -> None:
        if self._link is None:
            return
        while True:
            evt = self._link.poll_event()
            if evt is None:
                return
            if evt.name == "BTN" and evt.value == "START":
                self._game.start()
            elif evt.name == "BTN" and evt.value == "STOP":
                self._game.stop()

    def _ensure_engine(self) -> PoseEngine:
        if not HAS_VISION:
            raise HTTPException(
                status_code=503,
                detail="Vision/pose detection unavailable (mediapipe not installed)",
            )
        with self._engine_lock:
            if self._engine is None:
                self._engine = PoseEngine()
            return self._engine

    def predict_image(self, image_bytes: bytes) -> dict[str, Any]:
        if not HAS_VISION:
            raise HTTPException(
                status_code=503,
                detail="Vision/pose detection unavailable (mediapipe not installed)",
            )
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="cannot decode image")

        engine = self._ensure_engine()
        with self._engine_lock:
            _, pred = engine.process(frame)
            label = pred.label if pred is not None else None
            conf = pred.confidence if pred is not None else 0.0
            confirmed = engine.stabilizer.update(label, conf)

        return {
            "prediction": asdict(pred) if pred is not None else None,
            "confirmed": confirmed,
        }


app = FastAPI(title="Ong-Or FastAPI Bridge", version="0.1.0")
bridge = FastApiGameBridge()

cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
def on_shutdown() -> None:
    bridge.close()


@app.on_event("startup")
def on_startup() -> None:
    bridge.start_loop()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "ts": time.time(), "vision": HAS_VISION}


@app.get("/labels")
def labels() -> dict[str, Any]:
    return {"labels": bridge.labels()}


@app.get("/game/state")
def game_state() -> dict[str, Any]:
    return bridge.state()


@app.post("/game/start")
def game_start() -> dict[str, Any]:
    return bridge.start()


@app.post("/game/stop")
def game_stop() -> dict[str, Any]:
    return bridge.stop()


@app.post("/game/confirm")
def game_confirm(body: ConfirmBody) -> dict[str, Any]:
    return bridge.confirm(body.label)


@app.post("/game/tick")
def game_tick() -> dict[str, Any]:
    return bridge.tick()


@app.websocket("/ws/events")
async def ws_events(ws: WebSocket) -> None:
    await bridge.stream_events(ws)


@app.post("/vision/predict")
async def vision_predict(image: UploadFile = File(...)) -> dict[str, Any]:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    return bridge.predict_image(data)
