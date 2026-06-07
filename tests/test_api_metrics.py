from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from api import main as api_main
from ongor.labels import Prediction


class _FakeStabilizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, float]] = []
        self.reset_count = 0

    def update(self, label: str | None, confidence: float) -> str | None:
        self.calls.append((label, confidence))
        return label

    def reset(self) -> None:
        self.reset_count += 1


class _FakeEngine:
    def __init__(self) -> None:
        self.stabilizer = _FakeStabilizer()

    def process_topk(self, frame, k: int = 3):
        predictions = [
            Prediction("left", 0.9, "left"),
            Prediction("right", 0.08, "right"),
            Prediction("idle", 0.02, "idle"),
        ]
        return object(), predictions


class _FakeCv2:
    IMREAD_COLOR = 1

    @staticmethod
    def imdecode(arr, mode):
        return np.zeros((10, 10, 3), dtype=np.uint8)


class PosePredictorMetricsTest(unittest.TestCase):
    def test_predict_returns_top_predictions_and_records_timings(self) -> None:
        predictor = api_main.PosePredictor()
        predictor._engine = _FakeEngine()

        with patch.object(api_main, "HAS_VISION", True), patch.object(
            api_main, "cv2", _FakeCv2
        ):
            result = predictor.predict_image(b"jpeg")

        self.assertEqual(result["prediction"]["label"], "left")
        self.assertEqual(len(result["top_predictions"]), 3)
        self.assertEqual(result["confirmed"], "left")
        metrics = predictor.metrics_snapshot()
        self.assertIn("decode_ms", metrics)
        self.assertIn("inference_ms", metrics)

    def test_health_merges_latest_box_metrics(self) -> None:
        api_main.update_box_metrics(
            api_main.BoxMetrics(
                capture_fps=24.1,
                ai_fps=8.2,
                jpeg_ms=4.3,
                request_ms=130.0,
                dropped_frames=7,
            )
        )

        result = api_main.health()

        self.assertEqual(result["metrics"]["capture_fps"], 24.1)
        self.assertEqual(result["metrics"]["dropped_frames"], 7)
        self.assertIn("inference_ms", result["metrics"])
        self.assertIn("tflite_backend", result["metrics"])

    def test_new_stream_resets_temporal_state(self) -> None:
        predictor = api_main.PosePredictor()
        engine = _FakeEngine()
        predictor._engine = engine

        with patch.object(api_main, "HAS_VISION", True), patch.object(
            api_main, "cv2", _FakeCv2
        ):
            predictor.predict_image(b"jpeg", stream_id="1")
            predictor.predict_image(b"jpeg", stream_id="1")
            predictor.predict_image(b"jpeg", stream_id="2")

        self.assertEqual(engine.stabilizer.reset_count, 2)


if __name__ == "__main__":
    unittest.main()
