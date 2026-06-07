from __future__ import annotations

import unittest

from box.realtime import (
    LabelSmoother,
    LatestFrameBuffer,
    LatestResultStore,
    RollingMetrics,
)


class LatestFrameBufferTest(unittest.TestCase):
    def test_replaces_pending_frame_and_counts_drop(self) -> None:
        frames = LatestFrameBuffer()

        frames.put(1, "old")
        frames.put(2, "new")

        self.assertEqual(frames.get(timeout=0.01), (2, "new"))
        self.assertEqual(frames.dropped, 1)


class LatestResultStoreTest(unittest.TestCase):
    def test_rejects_result_older_than_published_result(self) -> None:
        results = LatestResultStore()

        self.assertTrue(results.publish(4, {"prediction": {"label": "idle"}}))
        self.assertFalse(results.publish(3, {"prediction": {"label": "prayHand"}}))
        self.assertEqual(results.latest()[0], 4)


class LabelSmootherTest(unittest.TestCase):
    def test_requires_repeated_weighted_evidence_before_switching(self) -> None:
        smoother = LabelSmoother(window_size=3, switch_ratio=1.25)

        self.assertEqual(smoother.update("left", 0.9), ("left", 0.9))
        self.assertEqual(smoother.update("right", 0.6)[0], "left")
        self.assertEqual(smoother.update("right", 0.95)[0], "right")

    def test_none_samples_clear_stale_label(self) -> None:
        smoother = LabelSmoother(window_size=5, switch_ratio=1.1)
        smoother.update("left", 0.9)

        smoother.update(None, 0.0)
        label, confidence = smoother.update(None, 0.0)

        self.assertIsNone(label)
        self.assertEqual(confidence, 0.0)


class RollingMetricsTest(unittest.TestCase):
    def test_snapshot_reports_rates_timings_and_drops(self) -> None:
        metrics = RollingMetrics(window_size=4)
        metrics.mark_capture(now=10.0)
        metrics.mark_capture(now=10.04)
        metrics.mark_ai(now=20.0)
        metrics.mark_ai(now=20.2)
        metrics.observe("jpeg_ms", 4.0)
        metrics.observe("request_ms", 100.0)
        metrics.set_value("dropped_frames", 3)

        snapshot = metrics.snapshot()

        self.assertAlmostEqual(snapshot["capture_fps"], 25.0)
        self.assertAlmostEqual(snapshot["ai_fps"], 5.0)
        self.assertEqual(snapshot["jpeg_ms"], 4.0)
        self.assertEqual(snapshot["request_ms"], 100.0)
        self.assertEqual(snapshot["dropped_frames"], 3)


if __name__ == "__main__":
    unittest.main()
