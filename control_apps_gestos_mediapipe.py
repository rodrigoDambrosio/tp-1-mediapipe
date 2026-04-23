import argparse
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from collections import deque, Counter

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import json
import ctypes
from ctypes import wintypes


APP_PRESETS = {
    "notepad": {
        "open_cmd": ["notepad.exe"],
        "process_name": "notepad.exe",
        "label": "Notepad",
    },
    "calculator": {
        "open_cmd": ["calc.exe"],
        "process_name": "CalculatorApp.exe",
        "label": "Calculator",
    },
    "paint": {
        "open_cmd": ["mspaint.exe"],
        "process_name": "mspaint.exe",
        "label": "Paint",
    },
}

GESTURE_ACTIONS = {
    "one": ("paint", "toggle"),
    "two": ("calculator", "toggle"),
    "fist": ("notepad", "toggle"),
}

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_FILENAME = "hand_landmarker.task"

# Detection and smoothing thresholds (tweakable)
Y_DELTA_THRESHOLD = 0.015
AVG_TIP_WRIST_THRESHOLD = 0.22
SMOOTHING_WINDOW = 5
SMOOTHING_THRESHOLD = 3
SEQUENCE_WINDOW = 2.0


def parse_source(source_arg: str):
    source_arg = source_arg.strip()
    if source_arg.isdigit():
        return int(source_arg)
    return source_arg


def open_capture(source):
    attempts = []

    if isinstance(source, int):
        candidates = [
            (source, None, f"index {source} (CAP_ANY)"),
            (source, cv2.CAP_MSMF, f"index {source} (CAP_MSMF)"),
            (source, cv2.CAP_DSHOW, f"index {source} (CAP_DSHOW)"),
        ]
    else:
        candidates = [(source, None, f"source '{source}'")]

    for src, backend, label in candidates:
        cap = cv2.VideoCapture(src) if backend is None else cv2.VideoCapture(src, backend)
        if not cap.isOpened():
            attempts.append(f"{label}: cannot open")
            cap.release()
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            attempts.append(f"{label}: opened but no frames")
            cap.release()
            continue

        attempts.append(f"{label}: OK")
        return cap, attempts

    return None, attempts


def ensure_hand_model_downloaded() -> Path:
    model_path = Path(__file__).with_name(HAND_MODEL_FILENAME)
    if model_path.exists():
        return model_path

    print("Downloading hand model (first time only)...")
    urllib.request.urlretrieve(HAND_MODEL_URL, model_path)
    print(f"Model saved at: {model_path}")
    return model_path


def create_landmarker(model_path: Path, max_hands: int = 1) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def _distance(a, b) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def classify_gesture(hand_landmarks, handedness_label: str | None = None) -> str | None:
    wrist = hand_landmarks[0]

    finger_pairs = [
        (8, 6),   # index
        (12, 10), # middle
        (16, 14), # ring
        (20, 18), # pinky
    ]

    extended = 0
    for tip_idx, pip_idx in finger_pairs:
        if hand_landmarks[tip_idx].y < hand_landmarks[pip_idx].y - Y_DELTA_THRESHOLD:
            extended += 1

    thumb_extended = False
    if handedness_label == "Right":
        thumb_extended = hand_landmarks[4].x < hand_landmarks[3].x - 0.015
    elif handedness_label == "Left":
        thumb_extended = hand_landmarks[4].x > hand_landmarks[3].x + 0.015
    else:
        thumb_extended = abs(hand_landmarks[4].x - hand_landmarks[3].x) > 0.05

    if thumb_extended:
        extended += 1

    tip_indices = [4, 8, 12, 16, 20]
    avg_tip_wrist = sum(_distance(hand_landmarks[idx], wrist) for idx in tip_indices) / len(tip_indices)

    # Detect open hand when most fingers (including thumb) are extended
    if extended >= 4 and avg_tip_wrist > AVG_TIP_WRIST_THRESHOLD:
        return "open"

    if extended == 0 and avg_tip_wrist < AVG_TIP_WRIST_THRESHOLD:
        return "fist"

    if extended == 1:
        return "one"

    if extended == 2:
        return "two"

    if extended == 3:
        return "three"

    return None


def draw_detection_overlay(
    frame,
    hand_landmarks,
    handedness_label,
    gesture_window,
    stable_gesture,
    smoothing_window,
    smoothing_threshold,
    last_seen_open_time,
    sequence_window,
):
    h, w = frame.shape[:2]
    now = time.monotonic()

    # color palette
    COLORS = {
        None: (200, 200, 200),
        "open": (0, 200, 0),
        "fist": (0, 0, 200),
        "one": (255, 0, 0),
        "two": (0, 180, 180),
        "three": (0, 180, 255),
    }

    # Draw landmarks and bones if available
    if hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        # simple skeleton connections (approximate MediaPipe topology)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]

        for a, b in connections:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (180, 180, 180), 1)

        for i, p in enumerate(pts):
            cv2.circle(frame, p, 3, (40, 40, 40), -1)

        # Per-finger extension visual (tip vs pip)
        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        for tip_idx, pip_idx in finger_pairs:
            tip = hand_landmarks[tip_idx]
            pip = hand_landmarks[pip_idx]
            extended = tip.y < pip.y - Y_DELTA_THRESHOLD
            tip_pt = (int(tip.x * w), int(tip.y * h))
            pip_pt = (int(pip.x * w), int(pip.y * h))
            col = (0, 200, 0) if extended else (0, 0, 200)
            cv2.line(frame, pip_pt, tip_pt, col, 3)
            cv2.putText(
                frame,
                "EXT" if extended else "FLX",
                (pip_pt[0] - 10, pip_pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                col,
                1,
            )

        # Thumb indicator
        thumb_tip = hand_landmarks[4]
        thumb_ip = hand_landmarks[3]
        if handedness_label == "Right":
            thumb_ext = thumb_tip.x < thumb_ip.x - 0.015
        elif handedness_label == "Left":
            thumb_ext = thumb_tip.x > thumb_ip.x + 0.015
        else:
            thumb_ext = abs(thumb_tip.x - thumb_ip.x) > 0.05
        tt = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        ti = (int(thumb_ip.x * w), int(thumb_ip.y * h))
        col = (0, 200, 0) if thumb_ext else (0, 0, 200)
        cv2.line(frame, ti, tt, col, 3)
        cv2.putText(frame, "T-EXT" if thumb_ext else "T-FLX", (ti[0] - 10, ti[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        # Wrist radius circle based on avg tip distance
        wrist = hand_landmarks[0]
        tip_indices = [4, 8, 12, 16, 20]
        avg_tip_wrist = sum(_distance(hand_landmarks[idx], wrist) for idx in tip_indices) / len(tip_indices)
        # scale normalized distance to pixels (rough)
        scale = (w + h) / 2.0
        radius_px = max(6, int(avg_tip_wrist * scale * 0.5))
        wrist_pt = (int(wrist.x * w), int(wrist.y * h))
        col = (0, 200, 0) if avg_tip_wrist > AVG_TIP_WRIST_THRESHOLD else (0, 0, 200)
        cv2.circle(frame, wrist_pt, radius_px, col, 2)
        cv2.putText(frame, f"r={avg_tip_wrist:.2f}", (wrist_pt[0] + 8, wrist_pt[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    # Draw smoothing timeline (top-right)
    box_w = 18
    box_h = 18
    spacing = 6
    start_x = frame.shape[1] - (box_w + spacing) * smoothing_window - 10
    y = 10
    counts = Counter(gesture_window)
    for i in range(smoothing_window):
        idx = max(0, len(gesture_window) - smoothing_window) + i
        g = gesture_window[idx] if idx < len(gesture_window) else None
        col = COLORS.get(g, (200, 200, 200))
        x = start_x + i * (box_w + spacing)
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), col, -1)
        if g:
            cv2.putText(frame, (g[0] if g else "-"), (x + 4, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Stable gesture and votes
    votes = counts.get(stable_gesture, 0) if stable_gesture else 0
    stable_col = COLORS.get(stable_gesture, (200, 200, 200))
    cv2.putText(frame, f"Stable: {stable_gesture or 'none'} ({votes}/{smoothing_window})", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stable_col, 2)

    # Sequence indicator
    seq_x = frame.shape[1] - 220
    seq_y = frame.shape[0] - 40
    cv2.putText(frame, "[OPEN] -> [FIST]", (seq_x, seq_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)
    if last_seen_open_time and (now - last_seen_open_time) <= sequence_window:
        left = sequence_window - (now - last_seen_open_time)
        cv2.putText(frame, f"wait {left:.1f}s", (seq_x + 10, seq_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 160, 255), 2)

    # Thresholds info
    cv2.putText(frame, "y_delta=0.015 | avg_r=0.22", (10, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)


@dataclass
class AppController:
    open_cmd: list[str]
    process_name: str | None

    def is_running(self) -> bool:
        if not self.process_name:
            return False

        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {self.process_name}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
        return self.process_name.lower() in result.stdout.lower()

    def open_app(self) -> bool:
        try:
            subprocess.Popen(self.open_cmd)
            return True
        except OSError:
            return False

    def close_app(self) -> bool:
        if not self.process_name:
            return False

        result = subprocess.run(
            ["taskkill", "/IM", self.process_name, "/F"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0


def build_app_controllers() -> dict[str, tuple[str, AppController]]:
    controllers = {}
    for app_key, data in APP_PRESETS.items():
        controllers[app_key] = (
            data["label"],
            AppController(open_cmd=data["open_cmd"], process_name=data["process_name"]),
        )
    return controllers


def perform_app_action(app_key: str, operation: str, controllers: dict[str, tuple[str, AppController]]):
    """Perform the requested operation for an app and return (action_info, last_action_label, success).

    - `operation` can be 'open', 'toggle', or other (treated as close).
    - Returns a tuple: (human-readable action_info, last_action_label_or_None, success_bool)
    """
    app_label, controller = controllers[app_key]

    if operation == "open":
        ok = controller.open_app()
        return (f"opened {app_label}" if ok else f"open {app_label} failed", f"open {app_label}" if ok else None, ok)

    if operation == "toggle":
        if controller.is_running():
            ok = controller.close_app()
            return (f"closed {app_label}" if ok else f"close {app_label} failed", f"close {app_label}" if ok else None, ok)
        else:
            ok = controller.open_app()
            return (f"opened {app_label}" if ok else f"open {app_label} failed", f"open {app_label}" if ok else None, ok)

    # fallback: try to close
    ok = controller.close_app()
    return (f"closed {app_label}" if ok else f"close {app_label} failed", f"close {app_label}" if ok else None, ok)


# Window position persistence
WINDOW_STATE_FILE = Path(__file__).with_name("window_state.json")


def load_window_pos() -> tuple[int, int] | None:
    if not WINDOW_STATE_FILE.exists():
        return None
    try:
        data = json.loads(WINDOW_STATE_FILE.read_text())
        return int(data.get("x")), int(data.get("y"))
    except Exception:
        return None


def save_window_pos(x: int, y: int) -> None:
    try:
        WINDOW_STATE_FILE.write_text(json.dumps({"x": int(x), "y": int(y)}))
    except Exception:
        pass


def get_window_pos_native(window_name: str) -> tuple[int, int] | None:
    try:
        FindWindow = ctypes.windll.user32.FindWindowW
        GetWindowRect = ctypes.windll.user32.GetWindowRect
        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        hwnd = FindWindow(None, window_name)
        if not hwnd:
            return None
        rect = RECT()
        res = GetWindowRect(hwnd, ctypes.byref(rect))
        if res == 0:
            return None
        return rect.left, rect.top
    except Exception:
        return None


def get_window_pos(window_name: str) -> tuple[int, int] | None:
    # Prefer OpenCV's API if available
    try:
        rect = cv2.getWindowImageRect(window_name)
        if rect and len(rect) >= 2:
            return int(rect[0]), int(rect[1])
    except Exception:
        pass
    return get_window_pos_native(window_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Control simple de apps con gestos usando MediaPipe Hands"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (0,1,2...) or video path. Default: 0",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.5,
        help="Seconds between actions to avoid repeated triggers",
    )
    args = parser.parse_args()

    source = parse_source(args.source)
    cap, attempts = open_capture(source)
    if cap is None:
        print("Could not open camera/video source")
        for item in attempts:
            print(f"  - {item}")
        return

    controllers = build_app_controllers()

    model_path = ensure_hand_model_downloaded()
    landmarker = create_landmarker(model_path=model_path, max_hands=1)
    last_action_time = 0.0
    last_action_label = "none"
    last_triggered_gesture = None
    last_seen_open_time = 0.0
    # Gesture smoothing window
    gesture_window = deque(maxlen=SMOOTHING_WINDOW)
    window_name = "MediaPipe Gesture App Control"
    # Keep a strictly increasing timestamp for MediaPipe's video API
    last_timestamp_ms = 0
    # Create window and restore last position if available
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    pos = load_window_pos()
    if pos:
        try:
            cv2.moveWindow(window_name, pos[0], pos[1])
        except Exception:
            pass

    print("Gesture control ready")
    print("Gesture mapping:")
    print("  - 1 finger  -> toggle Paint (open/close)")
    print("  - 2 fingers -> toggle Calculator (open/close)")
    print("  - Open then Fist -> toggle Notepad (open/close)")
    print("Keys: Q or ESC to exit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Use higher-resolution monotonic clock to reduce timestamp collisions
            timestamp_ms = time.monotonic_ns() // 1_000_000
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            gesture = None
            handedness_label = None
            if result.hand_landmarks:
                if result.handedness and result.handedness[0]:
                    handedness_label = result.handedness[0][0].category_name
                gesture = classify_gesture(result.hand_landmarks[0], handedness_label)

            # Append latest raw detection into the smoothing window
            gesture_window.append(gesture)

            # Determine stable gesture by majority vote over the window
            stable_gesture = None
            if len(gesture_window) > 0:
                counts = Counter(gesture_window)
                most = counts.most_common(1)
                if most:
                    candidate, votes = most[0]
                    if candidate is not None and votes >= SMOOTHING_THRESHOLD:
                        stable_gesture = candidate

            # Draw debug overlay showing landmarks, per-finger ext, smoothing timeline
            try:
                draw_detection_overlay(
                    frame,
                    result.hand_landmarks[0] if result.hand_landmarks else None,
                    handedness_label,
                    gesture_window,
                    stable_gesture,
                    SMOOTHING_WINDOW,
                    SMOOTHING_THRESHOLD,
                    last_seen_open_time,
                    SEQUENCE_WINDOW,
                )
            except Exception:
                pass

            now = time.monotonic()
            action_info = "waiting"
            # Use the smoothed/stable gesture for control decisions
            if stable_gesture is None:
                last_triggered_gesture = None
            elif stable_gesture == "open":
                # record the time we saw an open hand; waiting for fist next
                last_seen_open_time = now
            elif stable_gesture in GESTURE_ACTIONS and stable_gesture != last_triggered_gesture and (now - last_action_time) >= args.cooldown:
                # For fist, require open -> fist sequence within SEQUENCE_WINDOW
                if stable_gesture == "fist":
                    if last_seen_open_time == 0.0 or (now - last_seen_open_time) > SEQUENCE_WINDOW:
                        action_info = "waiting for open->fist sequence"
                    else:
                        app_key, operation = GESTURE_ACTIONS[stable_gesture]
                        action_info_res, last_label, ok = perform_app_action(app_key, operation, controllers)
                        action_info = action_info_res
                        if ok and last_label:
                            last_action_label = last_label
                            last_action_time = now
                            last_triggered_gesture = stable_gesture
                            last_seen_open_time = 0.0
                else:
                    app_key, operation = GESTURE_ACTIONS[stable_gesture]
                    action_info_res, last_label, ok = perform_app_action(app_key, operation, controllers)
                    action_info = action_info_res
                    if ok and last_label:
                        last_action_label = last_label
                    if ok:
                        last_action_time = now
                        last_triggered_gesture = stable_gesture

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 85), (245, 245, 245), -1)
            cv2.putText(
                frame,
                f"Gesture: {gesture or 'none'}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (30, 30, 30),
                2,
            )
            cv2.putText(
                frame,
                f"Last action: {last_action_label} | {action_info}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (30, 30, 30),
                2,
            )
            cv2.putText(
                frame,
                "1:P toggle | 2:C toggle | Open+Fist:N toggle",
                (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (50, 50, 50),
                1,
            )

            # Check whether the window still exists before showing a frame.
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                if get_window_pos_native(window_name) is None:
                    break

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            # Re-check after waitKey in case the user closed the window while it was shown.
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                if get_window_pos_native(window_name) is None:
                    break

            if key == 27 or key in (ord("q"), ord("Q")):
                break
    finally:
        # Save current window position
        try:
            pos = get_window_pos(window_name)
            if pos:
                save_window_pos(pos[0], pos[1])
        except Exception:
            pass
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
