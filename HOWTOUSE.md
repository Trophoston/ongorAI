# HOW TO USE: Ong-Or FastAPI + Front-end + Arduino UNO Q Linux

This document is written for another AI agent (or developer) to onboard quickly.

## 1) Goal and System Shape

Run pose-classification game logic on Linux, expose it as FastAPI, let front-end connect by HTTP/WebSocket, and optionally bridge serial events to Arduino UNO Q.

Data flow:
1. Browser/mobile sends camera frames to backend (`/vision/predict`)
2. Backend predicts pose and returns `prediction` + `confirmed`
3. Front-end sends confirmed poses to game (`/game/confirm`)
4. Backend advances game state automatically (internal tick loop)
5. Front-end listens game events from `ws/events`
6. Backend can forward game events to Arduino serial if `ONGOR_SERIAL_PORT` is set

## 2) Files That Matter

- `api/main.py`: FastAPI service (game API, websocket, vision endpoint, serial bridge)
- `ongor/sequence_game.py`: core game state machine
- `ongor/pose_engine.py`: pose predictor + stabilizer
- `ongor/arduino_link.py`: serial transport to MCU
- `ongor/events.py`: event model + sinks
- `requirements.txt`: Python deps

## 3) Environment Setup

Use Python that has mediapipe + tensorflow installed.

### Prerequisite: install pip first

If `/usr/bin/python3 -m pip` fails with `No module named pip`, install pip before continuing:

```bash
sudo apt update
sudo apt install -y python3-pip
```

If your board does not use `apt`, install the package that provides `python3-pip` for your distro.

For the installer to create `.venv`, you also need the venv package:

```bash
sudo apt install -y python3-venv
```

The installer will create `/opt/ongorAI/.venv` and install `requirements.txt` there automatically.

Optional runtime env vars:

```bash
export ONGOR_SERIAL_PORT=/dev/ttyACM0
export CORS_ORIGINS=http://localhost:5173,http://192.168.1.20:3000
export GAME_TICK_HZ=20
```

Notes:
- `ONGOR_SERIAL_PORT` is optional. If unset, backend runs without MCU.
- `GAME_TICK_HZ` default is 20.

## 3.1) Copy From Your Drive To The Board

Use this when your project is still on a USB/drive mount and you want to install it onto the board filesystem.

### Step 1: Find the mounted drive

```bash
ls /media/arduino
```

If your drive is named `TrophosDisk`, the source path will usually be:

```bash
/media/arduino/TrophosDisk
```

### Step 2: Copy the project to /opt

```bash
sudo rm -rf /opt/ongorAI
sudo cp -r /media/arduino/TrophosDisk/ongorAI /opt/
```

If your project folder has a different name on the drive, replace `ongorAI` with that folder name.

### Step 3: Enter the project folder

```bash
cd /opt/ongorAI
ls
```

You should see files like `README.md`, `requirements.txt`, `api/`, `ongor/`, and `deploy/`.

### Step 4: Install dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv
```

### Step 5: Install and enable auto-start service

```bash
sudo bash ./install_systemd_service.sh --user ongor --group ongor --workdir /opt/ongorAI --python /usr/bin/python3
```

This script now creates `/opt/ongorAI/.venv`, installs `requirements.txt` there, and makes systemd run uvicorn from that virtualenv.

### Step 6: Edit runtime config

```bash
sudo nano /etc/default/ongor-fastapi
```

Recommended values:

```bash
ONGOR_HOST=0.0.0.0
ONGOR_PORT=8000
ONGOR_SERIAL_PORT=/dev/ttyACM0
CORS_ORIGINS=http://localhost:5173,http://192.168.1.20:3000
GAME_TICK_HZ=20
```

### Step 7: Start now and auto-start on boot

```bash
sudo systemctl daemon-reload
sudo systemctl enable ongor-fastapi
sudo systemctl restart ongor-fastapi
```

### Step 8: Check it works

```bash
sudo systemctl status ongor-fastapi --no-pager
curl http://127.0.0.1:8000/health
```

### Step 9: Watch logs

```bash
sudo journalctl -u ongor-fastapi -f
```

### Step 10: Open API docs in browser

```text
http://127.0.0.1:8000/docs
```

## 11) Troubleshooting

### Error: `/usr/bin/python3: No module named pip`

Install the venv package, then rerun the installer:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv
sudo bash ./install_systemd_service.sh --user ongor --group ongor --workdir /opt/ongorAI --python /usr/bin/python3
```

## 12) Start Backend

```bash
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Check:
- Swagger docs: `http://127.0.0.1:8000/docs`
- Health: `GET /health`

## 13) API Contract

### GET /health
Returns service status.

### GET /labels
Returns all model labels.

### GET /game/state
Returns current game state.

Example response:

```json
{
  "phase": "show",
  "score": 0,
  "round": 1,
  "sequence_length": 1,
  "next_expected": null,
  "showing": "prayHand",
  "time_left": 3.8
}
```

### POST /game/start
Starts a new game.

### POST /game/stop
Stops current game.

### POST /game/confirm
Input body:

```json
{ "label": "prayHand" }
```

Use when front-end has a confirmed pose event.

### POST /vision/predict
Multipart form-data with file field name `image` (jpg/png frame).
Returns:

```json
{
  "prediction": {
    "label": "prayHand",
    "confidence": 0.93,
    "thai": "พนมมือ"
  },
  "confirmed": "prayHand"
}
```

`confirmed` is null most of the time and non-null when stabilizer confirms hold-time.

### WebSocket /ws/events
Stream server game events in real-time.

Example event:

```json
{
  "id": 12,
  "type": "show_pose",
  "data": {"index": 0, "pose": "prayHand", "total": 1},
  "ts": 1780000000.123,
  "iso": "2026-06-07T10:11:12.123"
}
```

## 14) Front-end Integration Recipe

Recommended loop:
1. Connect WebSocket `/ws/events`
2. Call `POST /game/start`
3. Capture camera frame every 100-200 ms (5-10 fps)
4. Send frame to `POST /vision/predict`
5. If response `confirmed` is non-null, call `POST /game/confirm`
6. Render HUD from `GET /game/state` every 500 ms or drive UI from websocket events
7. Stop camera + call `POST /game/stop` when leaving page

Client-side rules:
- Throttle frame uploads (do not exceed 10 fps unless LAN + strong CPU)
- Reconnect websocket with backoff
- Ignore duplicate confirms for same pose within short window (e.g. 300 ms)

## 15) Minimal Front-end Pseudocode

```javascript
const API = "http://127.0.0.1:8000";
const ws = new WebSocket("ws://127.0.0.1:8000/ws/events");

ws.onmessage = (ev) => {
  const event = JSON.parse(ev.data);
  renderEvent(event);
};

async function startGame() {
  await fetch(`${API}/game/start`, { method: "POST" });
}

async function sendFrame(blob) {
  const fd = new FormData();
  fd.append("image", blob, "frame.jpg");
  const res = await fetch(`${API}/vision/predict`, { method: "POST", body: fd });
  const data = await res.json();

  if (data.confirmed) {
    await fetch(`${API}/game/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: data.confirmed })
    });
  }
}
```

## 16) Arduino UNO Q Linux Notes

- MCU communicates with backend by serial line protocol.
- Typical Linux device: `/dev/ttyACM0`.
- Backend listens for:
  - `BTN:START`
  - `BTN:STOP`
- Backend emits game events through `ArduinoSink` protocol in `ongor/events.py`.

If serial fails, backend still runs game + API (degraded mode).

## 17) Production Run (systemd template)

This repo now includes ready deployment files:

- `deploy/ongor-fastapi.service`
- `deploy/ongor-fastapi.env.example`
- `deploy/install_systemd_service.sh`

### 9.1 One-time install on board (auto start at boot)

1. Put project on board, example path: `/opt/ongorAI`
2. Open a shell inside the project root first:

```bash
cd /opt/ongorAI
```

3. Install Python packages in that environment
4. Run installer script as root

```bash
sudo bash ./install_systemd_service.sh \
  --user ongor \
  --group ongor \
  --workdir /opt/ongorAI \
  --python /usr/bin/python3
```

If the board still has an old copied installer, recopy the updated repo first. The old file cannot repair itself.

If you are not in the repo root, use the full path instead:

```bash
sudo bash /opt/ongorAI/deploy/install_systemd_service.sh \
  --user ongor \
  --group ongor \
  --workdir /opt/ongorAI \
  --python /usr/bin/python3
```

5. Edit env file

```bash
sudo nano /etc/default/ongor-fastapi
```

6. Restart after env changes

```bash
sudo systemctl restart ongor-fastapi
```

7. Verify service and logs

```bash
sudo systemctl status ongor-fastapi --no-pager
sudo journalctl -u ongor-fastapi -f
```

After this, backend will auto run every time board boots.

### 9.2 If you want manual unit setup (reference)

Create service manually (path example):

```ini
[Unit]
Description=Ong-Or FastAPI
After=network.target

[Service]
WorkingDirectory=/opt/ongor
Environment=ONGOR_SERIAL_PORT=/dev/ttyACM0
Environment=CORS_ORIGINS=http://192.168.1.50:5173
Environment=GAME_TICK_HZ=20
ExecStart=/usr/bin/python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ongor-fastapi
sudo systemctl start ongor-fastapi
sudo systemctl status ongor-fastapi
```

## 18) Quick Test Script (Manual)

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/labels
curl -X POST http://127.0.0.1:8000/game/start
curl http://127.0.0.1:8000/game/state
```

## 19) Optional Vision Feature

**Important**: On some boards (e.g., Python 3.13 ARM64), mediapipe may not have pre-built wheels. The backend gracefully handles this:

- **With mediapipe**: `/vision/predict` endpoint works normally
- **Without mediapipe**: 
  - Backend still launches and exposes game API
  - `/vision/predict` returns HTTP 503 "Vision unavailable"
  - Frontend must skip pose detection or implement client-side vision
  - Game can still receive poses via manual `/game/confirm` calls

**Check vision availability**:

```bash
curl http://127.0.0.1:8000/health
```

Response includes `"vision": true/false`:

```json
{
  "ok": true,
  "ts": 1234567890.5,
  "vision": false
}
```

If you need pose detection on a board without mediapipe wheels:
1. Train model on a different Python version and copy `.keras` model to board
2. Implement pose detection on the frontend (e.g., TensorFlow.js or client-side MediaPipe)
3. Send confirmed poses directly to `/game/confirm`

## 20) If You Are Another AI Agent

If asked to extend this stack, do it in this order:
1. Preserve current API contract unless requested breaking changes.
2. Add tests for endpoint behavior before refactor.
3. Keep serial optional; never hard-fail startup if serial missing.
4. Keep vision optional; never hard-fail startup if mediapipe missing.
5. Keep event names consistent with `ongor/events.py` constants.
6. Document any new endpoint in this file immediately.
