"""
ป้ายกำกับท่าทาง — อ่านจาก my-pose-model/metadata.json อัตโนมัติ
เปลี่ยนโมเดล (เพิ่ม/ลดท่า) แล้วไม่ต้องแก้โค้ด ขอแค่ลำดับตรงกับ output ของ classifier
"""
from __future__ import annotations

import json
from dataclasses import dataclass

# แหล่ง label: ใช้ label_map.json ของโมเดล MediaPipe ก่อน (แหล่งความจริงปัจจุบัน)
# ถ้าไม่มี ค่อย fallback ไป metadata.json ของ Teachable Machine (โมเดลเก่า)
from .paths import LABEL_MAP as _LABEL_MAP, TM_METADATA as _METADATA

# ชื่อภาษาไทยสำหรับแสดงผล (รู้จักท่าไหนก็ใส่ไว้ ท่าที่ไม่รู้จักจะ fallback เป็น label เดิม)
THAI_NAMES = {
    "Panomue": "พนมมือ",
    "prayHand": "พนมมือ",
    "thb_touch_heaad_both": "แตะหัวสองมือ",
    "thl_touch_head_l": "แตะหัวมือซ้าย",
    "thr_touch_head_R": "แตะหัวมือขวา",
    "hub_hand_up_Both": "ยกมือสองข้าง",
    "hul_hand_up_L": "ยกมือซ้าย",
    "hur_hand_up_R": "ยกมือขวา",
    "tpb_t_post_both": "ทีโพสสองข้าง",
    "tpl_t_post_L": "ทีโพสซ้าย",
    "tpr_t_post_R": "ทีโพสขวา",
    "idle": "อยู่นิ่ง",
}

# ชื่ออังกฤษสำหรับแสดงบนหน้าต่าง OpenCV (Hershey font รองรับแค่ ASCII)
EN_NAMES = {
    "Panomue": "Pray",
    "prayHand": "Pray",
    "thb_touch_heaad_both": "Touch Head (Both)",
    "thl_touch_head_l": "Touch Head (L)",
    "thr_touch_head_R": "Touch Head (R)",
    "hub_hand_up_Both": "Hands Up (Both)",
    "hul_hand_up_L": "Hand Up (L)",
    "hur_hand_up_R": "Hand Up (R)",
    "tpb_t_post_both": "T-Pose (Both)",
    "tpl_t_post_L": "T-Pose (L)",
    "tpr_t_post_R": "T-Pose (R)",
    "idle": "Idle",
}


def en_of(label: str) -> str:
    return EN_NAMES.get(label, label)


def _load_labels() -> tuple[str, ...]:
    # 1) label_map.json ของโมเดล MediaPipe (เรียงตาม index)
    try:
        data = json.loads(_LABEL_MAP.read_text())
        if data:
            return tuple(data[str(i)] for i in range(len(data)))
    except Exception:  # noqa: BLE001
        pass
    # 2) metadata.json ของ Teachable Machine (โมเดลเก่า)
    try:
        meta = json.loads(_METADATA.read_text())
        labels = meta.get("labels")
        if labels:
            return tuple(labels)
    except Exception as e:  # noqa: BLE001
        print(f"[labels] อ่าน label ไม่ได้: {e} — ใช้ค่า fallback")
    # fallback เผื่อไม่มีไฟล์ metadata (เช่นรันบนบอร์ดที่ไม่มีโมเดลต้นฉบับ)
    return (
        "Panomue",
        "thb_touch_heaad_both",
        "thl_touch_head_l",
        "thr_touch_head_R",
        "hub_hand_up_Both",
        "hul_hand_up_L",
        "hur_hand_up_R",
        "idle",
    )


LABELS: tuple[str, ...] = _load_labels()

# ท่าที่ใช้ในเกม (ตัด idle ออก ไม่ใช้เป็นโจทย์)
GAME_POSES: tuple[str, ...] = tuple(p for p in LABELS if p.lower() != "idle")


def thai_of(label: str) -> str:
    return THAI_NAMES.get(label, label)


@dataclass
class Prediction:
    label: str
    confidence: float
    thai: str
