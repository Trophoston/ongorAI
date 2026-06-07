from __future__ import annotations

import unittest

from ongor.pose_engine import PoseStabilizer


class PoseStabilizerTest(unittest.TestCase):
    def test_requires_hold_then_release_before_next_confirmation(self) -> None:
        stabilizer = PoseStabilizer(
            hold_time=0.5,
            conf_threshold=0.8,
            release_time=0.2,
        )

        self.assertIsNone(stabilizer.update("prayHand", 0.9, now=0.0))
        self.assertEqual(stabilizer.update("prayHand", 0.9, now=0.5), "prayHand")
        self.assertIsNone(stabilizer.update("prayHand", 0.9, now=1.0))
        self.assertIsNone(stabilizer.update("idle", 0.95, now=1.1))
        self.assertIsNone(stabilizer.update("idle", 0.95, now=1.31))
        self.assertIsNone(stabilizer.update("prayHand", 0.9, now=1.4))
        self.assertEqual(stabilizer.update("prayHand", 0.9, now=1.9), "prayHand")


if __name__ == "__main__":
    unittest.main()
