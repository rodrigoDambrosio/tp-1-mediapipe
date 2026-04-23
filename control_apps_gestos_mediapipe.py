import argparse
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

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
        if hand_landmarks[tip_idx].y < hand_landmarks[pip_idx].y - 0.015:
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

    if extended == 0 and avg_tip_wrist < 0.22:
        return "fist"

    if extended == 1:
        return "one"

    if extended == 2:
        return "two"

    if extended == 3:
        return "three"

    return None


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
    print("  - Fist      -> toggle Notepad (open/close)")
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
            if result.hand_landmarks:
                handedness_label = None
                if result.handedness and result.handedness[0]:
                    handedness_label = result.handedness[0][0].category_name
                gesture = classify_gesture(result.hand_landmarks[0], handedness_label)

            now = time.monotonic()
            action_info = "waiting"
            if gesture is None:
                last_triggered_gesture = None
            elif gesture in GESTURE_ACTIONS and gesture != last_triggered_gesture and (now - last_action_time) >= args.cooldown:
                app_key, operation = GESTURE_ACTIONS[gesture]
                app_label, controller = controllers[app_key]

                if operation == "open":
                    if controller.open_app():
                        action_info = f"opened {app_label}"
                        last_action_label = f"open {app_label}"
                    else:
                        action_info = f"open {app_label} failed"
                elif operation == "toggle":
                    if controller.is_running():
                        if controller.close_app():
                            action_info = f"closed {app_label}"
                            last_action_label = f"close {app_label}"
                        else:
                            action_info = f"close {app_label} failed"
                    else:
                        if controller.open_app():
                            action_info = f"opened {app_label}"
                            last_action_label = f"open {app_label}"
                        else:
                            action_info = f"open {app_label} failed"
                else:
                    if controller.close_app():
                        action_info = f"closed {app_label}"
                        last_action_label = f"close {app_label}"
                    else:
                        action_info = f"close {app_label} failed"

                last_action_time = now
                last_triggered_gesture = gesture

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
                "1:P toggle | 2:C toggle | Fist:N toggle",
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
