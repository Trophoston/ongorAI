from __future__ import annotations

import unittest

import numpy as np

from ongor.mediapipe_runner import MediaPipeExtractor


class _LostExtractor(MediaPipeExtractor):
    def __init__(self) -> None:
        self.flip = False
        self._roi = None
        self._lost_frames = 0
        self.reacquire_interval = 3
        self.acquire_calls = 0

    def _acquire(self, rgb):
        self.acquire_calls += 1
        return None


class ReacquireCooldownTest(unittest.TestCase):
    def test_heavy_reacquire_runs_periodically_while_pose_is_lost(self) -> None:
        extractor = _LostExtractor()
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        for _ in range(7):
            extractor.process(frame)

        self.assertEqual(extractor.acquire_calls, 3)


if __name__ == "__main__":
    unittest.main()
