"""
Ong-Or game package.

เรียกใช้ง่าย ๆ บนบอร์ด:
    from ongor import PoseEngine, SequenceGame, SeqConfig
    from ongor.events import print_sink, ArduinoSink, MultiSink
"""
from .labels import LABELS, GAME_POSES, THAI_NAMES, Prediction

__all__ = [
    "LABELS", "GAME_POSES", "THAI_NAMES", "Prediction",
    "PoseEngine", "PoseStabilizer",
    "SequenceGame", "SeqConfig", "Phase",
    "MediaPipeExtractor", "MediaPipeClassifier",
]


def __getattr__(name: str):
    # lazy import เพื่อไม่ลาก mediapipe/tensorflow มาตอน import labels เฉย ๆ
    if name in ("PoseEngine", "PoseStabilizer"):
        from . import pose_engine
        return getattr(pose_engine, name)
    if name in ("SequenceGame", "SeqConfig", "Phase"):
        from . import sequence_game
        return getattr(sequence_game, name)
    if name == "MediaPipeExtractor":
        from .mediapipe_runner import MediaPipeExtractor
        return MediaPipeExtractor
    if name == "MediaPipeClassifier":
        from .classifier_mp import MediaPipeClassifier
        return MediaPipeClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
