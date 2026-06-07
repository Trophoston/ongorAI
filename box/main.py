# main.py — Ong-Or Box (Arduino App Lab / ฝั่ง Linux)
# หน้าที่: เป็น "สมอง" ของกล่อง — อ่านกล้อง, เรียก AI ผ่าน HTTP API (/predict),
#          รัน logic เกมความจำท่าทาง, แล้วส่งผลให้ MCU (ภาษา C) ไปแสดงจอ/ไฟ/เสียง
#
# สถาปัตยกรรม:
#   [USB cam] -> main.py(นี่) --HTTP /predict--> [Ong-Or Pose API]
#                     |  (logic เกมอยู่ที่นี่)
#                     +--Bridge poll/set_mode--> [MCU .ino: LCD/Knob/Pixels/Buzzer]
#
# กติกาเกม (โหมด Play):
#   1. โชว์ "ท่าใหม่" 1 ท่า ให้จำ 5 วินาที
#   2. ผู้เล่นต้องทำ "ท่าเดิมทั้งหมดตามลำดับจากต้น" แล้วจบด้วยท่าใหม่
#   3. ทำถูกครบลำดับ -> +คะแนน, เพิ่มท่าใหม่ 1 ท่า, กลับข้อ 1 (ลำดับยาวขึ้น)
#   4. ทำผิด หรือ หมดเวลา (10 วิ/ท่า) -> GAME OVER

import os, glob, time, socket, struct, threading, random, queue
from arduino.app_utils import App, Bridge

from realtime import LatestFrameBuffer, LatestResultStore, RollingMetrics

try: import requests
except Exception: requests = None
try: import cv2
except Exception: cv2 = None

# ====== CONFIG ======
API_PORT     = int(os.environ.get("ONGOR_API_PORT", "8000"))
API_BASE     = os.environ.get("ONGOR_API", "")          # ว่าง = ให้หาเอง
CAMERA_INDEX = int(os.environ.get("ONGOR_CAM", "0"))
CAM_W        = int(os.environ.get("ONGOR_CAM_W", "640"))
CAM_H        = int(os.environ.get("ONGOR_CAM_H", "480"))
CAM_FPS      = int(os.environ.get("ONGOR_CAM_FPS", "30"))
DISPLAY_FPS  = int(os.environ.get("ONGOR_DISPLAY_FPS", "24"))
AI_FPS       = int(os.environ.get("ONGOR_AI_FPS", os.environ.get("ONGOR_FPS", "12")))
AI_WIDTH     = int(os.environ.get("ONGOR_AI_WIDTH", "640"))
JPEG_QUALITY = int(os.environ.get("ONGOR_JPEG", "78"))
REQ_TIMEOUT  = float(os.environ.get("ONGOR_TIMEOUT", "4.0"))
CAM_FAIL_MAX = 5

# พารามิเตอร์เกม (ปรับได้ผ่าน env)
MEMORIZE_T   = float(os.environ.get("ONGOR_MEMORIZE", "5.0"))   # โชว์ท่าใหม่กี่วินาที
STEP_TIMEOUT = float(os.environ.get("ONGOR_STEP_TIMEOUT", "10.0"))  # เวลาต่อ 1 ท่า
CLEAR_PAUSE  = float(os.environ.get("ONGOR_CLEAR_PAUSE", "1.2"))    # พักหลังผ่านรอบ
MAX_LEN      = int(os.environ.get("ONGOR_MAX_LEN", "50"))           # ยาวถึงเท่านี้ = ชนะ
INPUT_GRACE  = 0.5  # หลังเริ่มเฟส INPUT รอสักครู่ กัน confirm ค้างจากตอนโชว์

# ชื่อท่าแบบสั้น (<=16 ตัว) สำหรับ LCD 16x2 (ไทยแสดงไม่ได้ ใช้อังกฤษ)
SHORT_NAME = {
    "hub_hand_up_Both":     "Both Hands Up",
    "hul_hand_up_L":        "Left Hand Up",
    "hur_hand_up_R":        "Right Hand Up",
    "prayHand":             "Pray Hands",
    "Panomue":              "Pray Hands",
    "thb_touch_heaad_both": "Touch Head x2",
    "thl_touch_head_l":     "Head Left",
    "thr_touch_head_R":     "Head Right",
    "tpb_t_post_both":      "T-Pose Both",
    "tpl_t_post_L":         "T-Pose Left",
    "tpr_t_post_R":         "T-Pose Right",
    "idle":                 "Idle",
}
DEFAULT_POSES = ["hub_hand_up_Both", "hul_hand_up_L", "hur_hand_up_R", "prayHand",
                 "thb_touch_heaad_both", "thl_touch_head_l", "thr_touch_head_R"]

# ====== STATE ======
_lock = threading.Lock()
_mode = 0
_mode_epoch = 0
_display = {"l1": "Ong-Or Ready", "l2": "Select mode", "pix": 0, "buz": 0}
_running = True
_cam = None
_cam_lock = threading.RLock()
_cam_fail = 0
_api_ok = True
_camchk = {"n": 0, "t": 0.0, "fps": 0.0}
_api_en = {}              # ชื่ออังกฤษจาก /labels (fallback ของ SHORT_NAME)
_frames = LatestFrameBuffer()
_results = LatestResultStore()
_metrics = RollingMetrics()
_frame_sequence = 0
_last_result_sequence = -1
_last_metrics_report = 0.0
_capture_shape = (0, 0)
_session = requests.Session() if requests is not None else None

def set_display(l1="", l2="", pix=0, buz=None):
    with _lock:
        _display["l1"] = str(l1)[:16]; _display["l2"] = str(l2)[:16]
        _display["pix"] = int(pix)
        if buz is not None: _display["buz"] = int(buz)

def pose_name(lbl):
    if not lbl: return "-"
    return (SHORT_NAME.get(lbl) or _api_en.get(lbl) or lbl)[:16]

# ====== Bridge (คุยกับ MCU) ======
def bridge_set_mode(mode):
    global _mode, _mode_epoch
    try: m = int(mode)
    except Exception: m = 0
    with _lock:
        if m != _mode:
            _mode_epoch += 1
        _mode = m
    _frames.clear()
    _results.clear()
    print(f"[Bridge] set_mode -> {m}"); return "OK"
def bridge_poll():
    with _lock:
        s = f'{_display["l1"]}|{_display["l2"]}|{_display["pix"]}|{_display["buz"]}'
        _display["buz"] = 0
    return s
Bridge.provide("set_mode", bridge_set_mode)
Bridge.provide("poll", bridge_poll)

# ====== หา API host เอง (แก้ปัญหา container localhost) ======
def _default_gateway():
    try:
        with open("/proc/net/route") as f:
            for line in f.readlines()[1:]:
                fld = line.strip().split()
                if fld[1] == "00000000":
                    return socket.inet_ntoa(struct.pack("<L", int(fld[2], 16)))
    except Exception:
        return None

def _candidate_bases():
    out = []
    if API_BASE: out.append(API_BASE)
    gw = _default_gateway()
    if gw: out.append(f"http://{gw}:{API_PORT}")
    out += [f"http://host.docker.internal:{API_PORT}",
            f"http://172.17.0.1:{API_PORT}",
            f"http://127.0.0.1:{API_PORT}"]
    seen, uniq = set(), []
    for b in out:
        if b not in seen: seen.add(b); uniq.append(b)
    return uniq

def resolve_api_base():
    global API_BASE
    if requests is None: return None
    for b in _candidate_bases():
        try:
            if requests.get(f"{b}/health", timeout=2.0).ok:
                API_BASE = b; print(f"[API] using {b}"); return b
        except Exception:
            continue
    print("[API] no reachable backend"); return None

# ====== Camera ======
def _list_devices():
    devs = sorted(glob.glob("/dev/video*"))
    pref = f"/dev/video{CAMERA_INDEX}"
    if pref in devs: devs.remove(pref); devs.insert(0, pref)
    return devs

def _open_try(src, backend=None):
    try:
        cam = cv2.VideoCapture(src, backend) if backend is not None else cv2.VideoCapture(src)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cam.set(cv2.CAP_PROP_FPS, CAM_FPS)
        if cam.isOpened() and cam.read()[0]: return cam
        cam.release()
    except Exception: pass
    return None

def _pipes(dev):
    base = f"v4l2src device={dev}"
    return [
        ("nv12",  f"{base} ! video/x-raw,format=NV12,width={CAM_W},height={CAM_H},framerate={CAM_FPS}/1 ! videoconvert ! appsink drop=true max-buffers=2"),
        ("mjpeg", f"{base} ! image/jpeg,width={CAM_W},height={CAM_H},framerate={CAM_FPS}/1 ! jpegdec ! videoconvert ! appsink drop=true max-buffers=2"),
        ("raw",   f"{base} ! video/x-raw,width={CAM_W},height={CAM_H},framerate={CAM_FPS}/1 ! videoconvert ! appsink drop=true max-buffers=2"),
        ("plain", f"{base} ! videoconvert ! appsink drop=true max-buffers=2"),
    ]

def _open_camera():
    if cv2 is None: return None
    forced = os.environ.get("ONGOR_CAM_PIPELINE")
    if forced:
        cam = _open_try(forced, cv2.CAP_GSTREAMER)
        if cam: print("[Camera] forced OK"); return cam
    for dev in _list_devices():
        for desc, p in _pipes(dev):
            cam = _open_try(p, cv2.CAP_GSTREAMER)
            if cam: print(f"[Camera] {dev} gst-{desc} OK"); return cam
        cam = _open_try(dev)
        if cam: print(f"[Camera] {dev} auto OK"); return cam
    for idx in range(6):
        cam = _open_try(idx)
        if cam: print(f"[Camera] index {idx} OK"); return cam
    print("[Camera] no working capture device found"); return None

def get_camera():
    global _cam
    with _cam_lock:
        if _cam is None:
            _cam = _open_camera()
        return _cam
def release_camera():
    global _cam, _cam_fail
    with _cam_lock:
        if _cam is not None:
            try: _cam.release()
            except Exception: pass
        _cam = None; _cam_fail = 0
def read_frame():
    global _cam, _cam_fail
    with _cam_lock:
        if _cam is None:
            _cam = _open_camera()
        if _cam is None:
            return None
        try: ok, frame = _cam.read()
        except Exception: ok, frame = False, None
        if not ok or frame is None:
            _cam_fail += 1
            if _cam_fail >= CAM_FAIL_MAX:
                print("[Camera] reopen")
                release_camera()
            return None
        _cam_fail = 0
        return frame
def frame_to_jpeg(frame):
    try:
        h, w = frame.shape[:2]
        if AI_WIDTH > 0 and w > AI_WIDTH:
            new_h = max(1, int(round(h * AI_WIDTH / w)))
            frame = cv2.resize(frame, (AI_WIDTH, new_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None
    except Exception: return None

# ====== API calls ======
def predict(jpeg, stream_id):
    r = _session.post(f"{API_BASE}/predict",
                      files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                      data={"stream_id": str(stream_id)},
                      timeout=REQ_TIMEOUT)
    r.raise_for_status(); return r.json()

def fetch_poses():
    """ดึงท่าที่โมเดลรู้จักจาก /labels (ถ้าล้มเหลวใช้ DEFAULT_POSES)"""
    global _api_en
    try:
        r = requests.get(f"{API_BASE}/labels", timeout=REQ_TIMEOUT); r.raise_for_status()
        d = r.json()
        _api_en = d.get("en", {}) or {}
        poses = [p for p in d.get("poses", []) if p and p.lower() != "idle"]
        if poses: return poses
    except Exception as e:
        print("[labels] fallback:", e)
    return list(DEFAULT_POSES)

# ====== เกมความจำท่าทาง (logic อยู่ฝั่ง Python ทั้งหมด) ======
class SeqGame:
    IDLE, SHOW, INPUT, CLEAR, OVER, WIN = "idle", "show", "input", "clear", "over", "win"

    def __init__(self, poses):
        self.poses = list(poses) or list(DEFAULT_POSES)
        self.reset()

    def reset(self):
        self.phase = self.IDLE; self.phase_since = 0.0
        self.seq = []; self.idx = 0; self.score = 0
        self.deadline = 0.0; self.input_since = 0.0; self.detected = ""

    def start(self, now):
        self.score = 0; self.seq = []
        self._append(); self._begin_round(now)

    def _append(self):
        prev = self.seq[-1] if self.seq else None
        choices = [p for p in self.poses if p != prev] or self.poses
        self.seq.append(random.choice(choices))

    def _begin_round(self, now):
        self.phase = self.SHOW; self.phase_since = now

    def _enter(self, ph, now):
        self.phase = ph; self.phase_since = now

    def on_confirm(self, label, now):
        """ป้อนท่าที่ AI ยืนยัน (มีผลเฉพาะเฟส INPUT) -> คืน event ไว้สั่งเสียง"""
        if self.phase != self.INPUT:
            return None
        if now - self.input_since < INPUT_GRACE:
            return None                       # กัน confirm ค้างจากตอนโชว์
        expected = self.seq[self.idx]
        if label == expected:
            self.idx += 1
            self.deadline = now + STEP_TIMEOUT
            if self.idx >= len(self.seq):     # ทำครบลำดับ = ผ่านรอบ
                self.score += len(self.seq)
                if len(self.seq) >= MAX_LEN:
                    self._enter(self.WIN, now); return "win"
                self._enter(self.CLEAR, now); return "clear"
            return "correct"
        else:
            self._enter(self.OVER, now); return "wrong"

    def tick(self, now):
        """จัดการเวลา -> คืน event"""
        if self.phase == self.SHOW:
            if now - self.phase_since >= MEMORIZE_T:
                self._enter(self.INPUT, now)
                self.idx = 0; self.input_since = now
                self.deadline = now + STEP_TIMEOUT
                return "input_start"
        elif self.phase == self.INPUT:
            if now >= self.deadline:
                self._enter(self.OVER, now); return "timeout"
        elif self.phase == self.CLEAR:
            if now - self.phase_since >= CLEAR_PAUSE:
                self._append(); self._begin_round(now); return "new_pose"
        return None

    def showing(self):
        return self.seq[-1] if (self.phase == self.SHOW and self.seq) else None
    def time_left(self, now):
        if self.phase == self.SHOW: return max(0.0, MEMORIZE_T - (now - self.phase_since))
        if self.phase == self.INPUT: return max(0.0, self.deadline - now)
        return 0.0

_game = SeqGame(DEFAULT_POSES)

# เสียง: 1=tick/โชว์, 2=ถูก, 3=ชนะ, 4=ผิด/แพ้
_EVENT_BUZ = {"input_start": 1, "new_pose": 1, "correct": 2,
              "clear": 2, "win": 3, "wrong": 4, "timeout": 4}

def _render_game(now):
    g = _game; ph = g.phase
    n = len(g.seq); tl = g.time_left(now)
    if ph == g.SHOW:
        bar = int(round(tl / max(0.1, MEMORIZE_T) * 8))
        set_display("MEMORIZE this:", pose_name(g.showing()), bar, None)
    elif ph == g.INPUT:
        bar = int(round((g.idx / max(1, n)) * 8))
        seen = pose_name(g.detected) if g.detected else "..."
        set_display(f"DO {g.idx+1}/{n}  {int(tl)}s", f"see:{seen}", bar, None)
    elif ph == g.CLEAR:
        set_display("CORRECT!", f"Score {g.score}", 8, None)
    elif ph == g.WIN:
        set_display("YOU WIN!", f"Score {g.score}", 8, None)
    elif ph == g.OVER:
        set_display("GAME OVER", f"Score {g.score}", 0, None)
    else:
        set_display("Get ready...", "", 0, None)

# ====== โหมดต่าง ๆ ======
def deps_ready(): return requests is not None and cv2 is not None
def mode_needs(m):
    if m == 3: return cv2 is not None
    if m in (1, 2): return deps_ready()
    return True

def _consume_ai_result():
    global _last_result_sequence
    sequence, item = _results.latest()
    if item is None or sequence <= _last_result_sequence:
        return None
    _last_result_sequence = sequence
    with _lock:
        epoch = _mode_epoch
    if item.get("_epoch") != epoch:
        return None
    return item

def run_cam_check():
    h, w = _capture_shape
    snap = _metrics.snapshot()
    fps = float(snap.get("capture_fps", 0.0) or 0.0)
    if not w or not h:
        set_display("CAM: no frame", "reconnecting", 0, None); return
    set_display(f"CAM OK {w}x{h}", f"{fps:.1f} fps", 8, None)

def run_test_mode():
    res = _consume_ai_result()
    if res is None:
        if not _api_ok: set_display("API down", "reconnecting", 0, None)
        return
    pred = res.get("smoothed_prediction") or res.get("prediction") or {}
    conf = float(pred.get("confidence", 0))
    set_display(pose_name(pred.get("label")), f"Conf {int(conf*100)}%",
                int(round(conf*8)), 0)

def run_play_mode():
    now = time.time()
    res = _consume_ai_result()
    if res is not None:
        pred = res.get("smoothed_prediction") or res.get("prediction") or {}
        _game.detected = pred.get("label", "") or ""
        confirmed = res.get("confirmed")
        if confirmed:
            ev = _game.on_confirm(confirmed, now)
            if ev: set_display(_display["l1"], _display["l2"], _display["pix"],
                               _EVENT_BUZ.get(ev))
    ev = _game.tick(now)
    if ev:
        # ตั้งเสียงก่อน render (render ใช้ buz=None จะไม่ทับ)
        with _lock: _display["buz"] = _EVENT_BUZ.get(ev, 0)
    _render_game(now)

# ====== Workers ======
def capture_worker():
    global _frame_sequence, _capture_shape
    period = 1.0 / max(1, DISPLAY_FPS)
    while _running:
        started = time.monotonic()
        with _lock:
            mode, epoch = _mode, _mode_epoch
        if mode not in (1, 2, 3):
            time.sleep(0.05)
            continue
        frame = read_frame()
        if frame is not None:
            _frame_sequence += 1
            _capture_shape = frame.shape[:2]
            _metrics.mark_capture(time.monotonic())
            if mode in (1, 2):
                _frames.put(_frame_sequence, (epoch, frame))
                _metrics.set_value("dropped_frames", _frames.dropped)
        elapsed = time.monotonic() - started
        if elapsed < period:
            time.sleep(period - elapsed)

def _report_metrics():
    global _last_metrics_report
    now = time.monotonic()
    if _session is None or not API_BASE or now - _last_metrics_report < 2.0:
        return
    _last_metrics_report = now
    snap = _metrics.snapshot()
    payload = {
        "capture_fps": snap.get("capture_fps", 0.0),
        "ai_fps": snap.get("ai_fps", 0.0),
        "jpeg_ms": snap.get("jpeg_ms", 0.0),
        "request_ms": snap.get("request_ms", 0.0),
        "dropped_frames": snap.get("dropped_frames", 0),
    }
    try:
        _session.post(f"{API_BASE}/metrics/box", json=payload, timeout=1.0)
        print("[Perf] capture={capture_fps:.1f} ai={ai_fps:.1f} "
              "jpeg={jpeg_ms:.1f}ms request={request_ms:.1f}ms "
              "dropped={dropped_frames}".format(**payload))
    except Exception:
        pass

def ai_worker():
    global _api_ok
    period = 1.0 / max(1, AI_FPS)
    next_run = 0.0
    while _running:
        with _lock:
            mode = _mode
        if mode not in (1, 2):
            time.sleep(0.05)
            continue
        wait = next_run - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            sequence, payload = _frames.get(timeout=0.25)
        except queue.Empty:
            continue
        epoch, frame = payload
        with _lock:
            if epoch != _mode_epoch or _mode not in (1, 2):
                continue
        if not API_BASE and resolve_api_base() is None:
            _api_ok = False
            time.sleep(1.0)
            continue
        next_run = time.monotonic() + period
        try:
            started = time.perf_counter()
            jpeg = frame_to_jpeg(frame)
            _metrics.observe("jpeg_ms", (time.perf_counter() - started) * 1000.0)
            if jpeg is None:
                continue
            started = time.perf_counter()
            result = predict(jpeg, epoch)
            _metrics.observe("request_ms", (time.perf_counter() - started) * 1000.0)
            _metrics.mark_ai(time.monotonic())
            result["_epoch"] = epoch
            _results.publish(sequence, result)
            if not _api_ok:
                print("[API] recovered")
            _api_ok = True
            _report_metrics()
        except requests.exceptions.RequestException as e:
            if _api_ok:
                print("[API] unreachable:", e)
            _api_ok = False
            resolve_api_base()
            time.sleep(0.5)
        except Exception as e:
            print("[AI]", e)
            time.sleep(0.2)

def worker():
    global _last_result_sequence
    active = 0
    while _running:
        period = 1.0/max(1, DISPLAY_FPS)
        with _lock: mode = _mode
        if mode != active:
            try:
                if not mode_needs(mode) and mode != 0:
                    set_display("Missing deps", "need cv2/req", 0, 4)
                elif mode == 1:                       # Play game (logic ที่นี่)
                    resolve_api_base()
                    _game.poses = fetch_poses()
                    _game.start(time.time())
                    set_display("Game start!", "Get ready", 0, 2)
                elif mode == 2:                       # Test AI
                    resolve_api_base(); set_display("Test AI", "Show a pose", 0, 1)
                elif mode == 3:                       # Cam check
                    _camchk.update(n=0, t=time.time(), fps=0.0)
                    set_display("Cam Check", "opening cam..", 0, 1)
                else:                                 # Idle/เมนู
                    _game.reset(); release_camera()
                    set_display("Ong-Or Ready", "Select mode", 0, 0)
            except Exception as e:
                print("[switch]", e)
            active = mode; time.sleep(0.3); continue

        if mode == 0 or not mode_needs(mode):
            time.sleep(0.15); continue

        t0 = time.time()
        try:
            if mode == 3: run_cam_check()
            elif mode == 2: run_test_mode()
            elif mode == 1: run_play_mode()
        except Exception as e:
            print("[worker]", e); set_display("Error", "see log", 0, None); time.sleep(0.5)
        dt = time.time()-t0
        if dt < period: time.sleep(period-dt)

resolve_api_base()
threading.Thread(target=capture_worker, daemon=True).start()
threading.Thread(target=ai_worker, daemon=True).start()
threading.Thread(target=worker, daemon=True).start()
App.run()
