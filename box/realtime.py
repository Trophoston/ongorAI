from __future__ import annotations

import queue
import threading
from collections import deque
from statistics import fmean
from typing import Any


class LatestFrameBuffer:
    """Single-slot frame queue that always keeps the newest frame."""

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[int, Any]] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self.dropped = 0

    def put(self, sequence: int, frame: Any) -> None:
        with self._lock:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
            self._queue.put_nowait((sequence, frame))

    def get(self, timeout: float | None = None) -> tuple[int, Any]:
        return self._queue.get(timeout=timeout)

    def clear(self) -> None:
        with self._lock:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    return


class LatestResultStore:
    """Thread-safe latest result that refuses out-of-order publications."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = -1
        self._result: dict[str, Any] | None = None

    def publish(self, sequence: int, result: dict[str, Any]) -> bool:
        with self._lock:
            if sequence <= self._sequence:
                return False
            self._sequence = sequence
            self._result = result
            return True

    def latest(self) -> tuple[int, dict[str, Any] | None]:
        with self._lock:
            return self._sequence, self._result

    def clear(self) -> None:
        with self._lock:
            self._sequence = -1
            self._result = None


class LabelSmoother:
    """Confidence-weighted label voting with resistance to label switching."""

    def __init__(
        self,
        window_size: int = 5,
        switch_ratio: float = 1.2,
        max_missing: int = 2,
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.switch_ratio = max(1.0, float(switch_ratio))
        self.max_missing = max(1, int(max_missing))
        self._samples: deque[tuple[str | None, float]] = deque(
            maxlen=self.window_size
        )
        self._label: str | None = None
        self._missing = 0

    def reset(self) -> None:
        self._samples.clear()
        self._label = None
        self._missing = 0

    def update(
        self, label: str | None, confidence: float
    ) -> tuple[str | None, float]:
        self._samples.append((label, max(0.0, float(confidence))))
        self._missing = self._missing + 1 if label is None else 0
        if self._missing >= self.max_missing:
            self._samples.clear()
            self._label = None
            return None, 0.0
        scores: dict[str, float] = {}
        counts: dict[str, int] = {}
        for sample_label, sample_confidence in self._samples:
            if sample_label is None:
                continue
            scores[sample_label] = scores.get(sample_label, 0.0) + sample_confidence
            counts[sample_label] = counts.get(sample_label, 0) + 1

        if not scores:
            self._label = None
            return None, 0.0

        winner = max(scores, key=scores.get)
        if self._label is not None and winner != self._label:
            current_score = scores.get(self._label, 0.0)
            if scores[winner] < current_score * self.switch_ratio:
                winner = self._label

        self._label = winner
        confidence = scores[winner] / counts[winner]
        return winner, min(1.0, confidence)


class RollingMetrics:
    """Small thread-safe rolling metrics store for FPS and stage timings."""

    def __init__(self, window_size: int = 60) -> None:
        self._window_size = max(2, int(window_size))
        self._lock = threading.Lock()
        self._capture_times: deque[float] = deque(maxlen=self._window_size)
        self._ai_times: deque[float] = deque(maxlen=self._window_size)
        self._timings: dict[str, deque[float]] = {}
        self._values: dict[str, float | int | str | None] = {}

    def mark_capture(self, now: float) -> None:
        with self._lock:
            self._capture_times.append(float(now))

    def mark_ai(self, now: float) -> None:
        with self._lock:
            self._ai_times.append(float(now))

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            values = self._timings.setdefault(
                name, deque(maxlen=self._window_size)
            )
            values.append(float(value))

    def set_value(self, name: str, value: float | int | str | None) -> None:
        with self._lock:
            self._values[name] = value

    @staticmethod
    def _rate(times: deque[float]) -> float:
        if len(times) < 2:
            return 0.0
        elapsed = times[-1] - times[0]
        return (len(times) - 1) / elapsed if elapsed > 0 else 0.0

    def snapshot(self) -> dict[str, float | int | str | None]:
        with self._lock:
            result: dict[str, float | int | str | None] = {
                "capture_fps": round(self._rate(self._capture_times), 2),
                "ai_fps": round(self._rate(self._ai_times), 2),
            }
            result.update(
                {
                    name: round(fmean(values), 2) if values else 0.0
                    for name, values in self._timings.items()
                }
            )
            result.update(self._values)
            return result
