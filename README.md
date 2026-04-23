MediaPipe Gesture App Control
=============================

A small demo that uses MediaPipe's Hand Landmarker to control simple Windows apps (Notepad, Calculator, Paint) via hand gestures.

Quick start
-----------

1. Create a virtual environment and install dependencies from `requirements.txt`.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the script (default camera 0):

```bash
python control_apps_gestos_mediapipe.py --source 0
```

Flags
-----
- `--source`: camera index (`0`, `1`, ...) or a video file path. Default `0`.
- `--cooldown`: seconds between actions to avoid repeated triggers. Default `2.5`.

Notes
-----
- The first run downloads the MediaPipe `hand_landmarker.task` model into the script directory.
- Window position is persisted to `window_state.json`.
- Use the debug toggle (bottom-right) to show landmarks and per-finger state.