from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from collections import deque, Counter
from typing import Any, Deque, Dict, List, Optional, Tuple

import ctypes

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ------------------------- Configuration ---------------------------------
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
NO_GESTURE_CLEAR_TIME = 1.0
THUMB_X_DELTA_THRESHOLD = 0.015
THUMB_X_FALLBACK = 0.05

WINDOW_STATE_FILE = Path(__file__).with_name("window_state.json")

# UI / layout constants
WINDOW_TITLE = "MediaPipe Gesture App Control"
HEADER_HEIGHT = 85
BTN_W, BTN_H = 140, 36

logger = logging.getLogger("gesture_app")


# ------------------------- Utility classes ---------------------------------
class CaptureManager:
    """Handle video source parsing and opening.

    The helpers here try multiple backends for camera indexes and verify a
    first frame can be read.
    """

    @staticmethod
    def parse_source(source_arg: str):
        source_arg = source_arg.strip()
        if source_arg.isdigit():
            return int(source_arg)
        return source_arg

    @staticmethod
    def open_capture(source) -> Tuple[Optional[cv2.VideoCapture], List[str]]:
        attempts: List[str] = []

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
                try:
                    cap.release()
                except Exception:
                    pass
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                attempts.append(f"{label}: opened but no frames")
                try:
                    cap.release()
                except Exception:
                    pass
                continue

            attempts.append(f"{label}: OK")
            return cap, attempts

        return None, attempts


class ModelManager:
    """Download and create MediaPipe hand landmarker.

    This isolates network IO and model creation from the main loop.
    """

    @staticmethod
    def ensure_hand_model_downloaded() -> Path:
        model_path = Path(__file__).with_name(HAND_MODEL_FILENAME)
        if model_path.exists():
            return model_path

        logger.info("Downloading hand model (first time only)...")
        urllib.request.urlretrieve(HAND_MODEL_URL, model_path)
        logger.info("Model saved at: %s", model_path)
        return model_path

    @staticmethod
    def create_landmarker(model_path: Path, max_hands: int = 1) -> Any:
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.HandLandmarker.create_from_options(options)


class GestureClassifier:
    """Pure logic for turning hand landmarks into gesture names and smoothing.
    """

    @staticmethod
    def _distance(a, b) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        dz = a.z - b.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    @classmethod
    def classify_gesture(cls, hand_landmarks, handedness_label: Optional[str] = None) -> Optional[str]:
        wrist = hand_landmarks[0]

        finger_pairs = [
            (8, 6),
            (12, 10),
            (16, 14),
            (20, 18),
        ]

        extended = 0
        for tip_idx, pip_idx in finger_pairs:
            if hand_landmarks[tip_idx].y < hand_landmarks[pip_idx].y - Y_DELTA_THRESHOLD:
                extended += 1

        thumb_extended = False
        if handedness_label == "Right":
            thumb_extended = hand_landmarks[4].x < hand_landmarks[3].x - THUMB_X_DELTA_THRESHOLD
        elif handedness_label == "Left":
            thumb_extended = hand_landmarks[4].x > hand_landmarks[3].x + THUMB_X_DELTA_THRESHOLD
        else:
            thumb_extended = abs(hand_landmarks[4].x - hand_landmarks[3].x) > THUMB_X_FALLBACK

        if thumb_extended:
            extended += 1

        tip_indices = [4, 8, 12, 16, 20]
        avg_tip_wrist = sum(cls._distance(hand_landmarks[idx], wrist) for idx in tip_indices) / len(tip_indices)

        if extended > 4 and avg_tip_wrist > AVG_TIP_WRIST_THRESHOLD:
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

    @staticmethod
    def get_stable_gesture(gesture_window: Deque[Optional[str]]) -> Optional[str]:
        non_none = [g for g in gesture_window if g is not None]
        if not non_none:
            return None
        counts = Counter(non_none)
        candidate, votes = counts.most_common(1)[0]
        return candidate if votes >= SMOOTHING_THRESHOLD else None


class Visualizer:
    """Drawing and UI helper functions grouped for clarity."""

    @staticmethod
    def draw_detection_overlay(frame, hand_landmarks, handedness_label):
        h, w = frame.shape[:2]

        if hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

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

            finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
            for tip_idx, pip_idx in finger_pairs:
                tip = hand_landmarks[tip_idx]
                pip = hand_landmarks[pip_idx]
                extended = tip.y < pip.y - Y_DELTA_THRESHOLD
                tip_pt = (int(tip.x * w), int(tip.y * h))
                pip_pt = (int(pip.x * w), int(pip.y * h))
                col = (0, 200, 0) if extended else (0, 0, 200)
                cv2.line(frame, pip_pt, tip_pt, col, 3)
                cv2.putText(frame, "EXT" if extended else "FLX", (pip_pt[0] - 10, pip_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

            # Thumb
            thumb_tip = hand_landmarks[4]
            thumb_ip = hand_landmarks[3]
            if handedness_label == "Right":
                thumb_ext = thumb_tip.x < thumb_ip.x - THUMB_X_DELTA_THRESHOLD
            elif handedness_label == "Left":
                thumb_ext = thumb_tip.x > thumb_ip.x + THUMB_X_DELTA_THRESHOLD
            else:
                thumb_ext = abs(thumb_tip.x - thumb_ip.x) > THUMB_X_FALLBACK
            tt = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            ti = (int(thumb_ip.x * w), int(thumb_ip.y * h))
            col = (0, 200, 0) if thumb_ext else (0, 0, 200)
            cv2.line(frame, ti, tt, col, 3)
            cv2.putText(frame, "T-EXT" if thumb_ext else "T-FLX", (ti[0] - 10, ti[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

            # Wrist radius
            wrist = hand_landmarks[0]
            tip_indices = [4, 8, 12, 16, 20]
            avg_tip_wrist = sum(GestureClassifier._distance(hand_landmarks[idx], wrist) for idx in tip_indices) / len(tip_indices)

            scale = (w + h) / 2.0
            radius_px = max(6, int(avg_tip_wrist * scale * 0.5))
            wrist_pt = (int(wrist.x * w), int(wrist.y * h))
            col = (0, 200, 0) if avg_tip_wrist > AVG_TIP_WRIST_THRESHOLD else (0, 0, 200)
            cv2.circle(frame, wrist_pt, radius_px, col, 2)
            cv2.putText(frame, f"r={avg_tip_wrist:.2f}", (wrist_pt[0] + 8, wrist_pt[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    @staticmethod
    def on_mouse(event, x, y, flags, param):
        state = param
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, x2, y2 = state.get("rect", (0, 0, 0, 0))
            if x1 <= x <= x2 and y1 <= y <= y2:
                state["enabled"] = not state.get("enabled", True)

    @staticmethod
    def compose_display(frame, window_name: str, overlay_state: Dict[str, Any]) -> np.ndarray:
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
        except Exception:
            win_w, win_h = frame.shape[1], frame.shape[0]

        scale = min(win_w / frame.shape[1], win_h / frame.shape[0])
        new_w = max(1, int(frame.shape[1] * scale))
        new_h = max(1, int(frame.shape[0] * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

        display = np.full((win_h, win_w, 3), 245, dtype=np.uint8)
        xoff = (win_w - new_w) // 2
        yoff = (win_h - new_h) // 2
        display[yoff : yoff + new_h, xoff : xoff + new_w] = resized

        btn_w, btn_h = 140, 36
        bx2 = win_w - 10
        by2 = win_h - 10
        bx1 = bx2 - btn_w
        by1 = by2 - btn_h
        overlay_state["rect"] = (bx1, by1, bx2, by2)
        if overlay_state.get("enabled", True):
            btn_color = (0, 200, 0)
            txt = "DEBUG: ON"
        else:
            btn_color = (80, 80, 80)
            txt = "DEBUG: OFF"
        cv2.rectangle(display, (bx1, by1), (bx2, by2), btn_color, -1)
        cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 0, 0), 1)
        text_y = by1 + int(btn_h * 0.65)
        cv2.putText(display, txt, (bx1 + 8, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return display


@dataclass
class AppController:
    open_cmd: List[str]
    process_name: Optional[str]

    def is_running(self) -> bool:
        if not self.process_name:
            return False

        result = subprocess.run([
            "tasklist",
            "/FI",
            f"IMAGENAME eq {self.process_name}",
        ], capture_output=True, text=True)
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

        result = subprocess.run(["taskkill", "/IM", self.process_name, "/F"], capture_output=True, text=True)
        return result.returncode == 0


class AppManager:
    """Manage AppController instances and expose simple actions."""

    def __init__(self, presets: Dict[str, Dict[str, Any]]):
        self.controllers: Dict[str, Tuple[str, AppController]] = {}
        for app_key, data in presets.items():
            self.controllers[app_key] = (
                data["label"],
                AppController(open_cmd=data["open_cmd"], process_name=data["process_name"]),
            )

    def perform_app_action(self, app_key: str, operation: str) -> Tuple[str, Optional[str], bool]:
        app_label, controller = self.controllers[app_key]

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

        # default to close
        ok = controller.close_app()
        return (f"closed {app_label}" if ok else f"close {app_label} failed", f"close {app_label}" if ok else None, ok)


class WindowManager:
    @staticmethod
    def load_window_pos() -> Optional[Tuple[int, int]]:
        if not WINDOW_STATE_FILE.exists():
            return None
        try:
            data = json.loads(WINDOW_STATE_FILE.read_text())
            return int(data.get("x")), int(data.get("y"))
        except Exception:
            return None

    @staticmethod
    def save_window_pos(x: int, y: int) -> None:
        try:
            WINDOW_STATE_FILE.write_text(json.dumps({"x": int(x), "y": int(y)}))
        except Exception:
            pass

    @staticmethod
    def get_window_pos_native(window_name: str) -> Optional[Tuple[int, int]]:
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

    @staticmethod
    def get_window_pos(window_name: str) -> Optional[Tuple[int, int]]:
        try:
            rect = cv2.getWindowImageRect(window_name)
            if rect and len(rect) >= 2:
                return int(rect[0]), int(rect[1])
        except Exception:
            pass
        return WindowManager.get_window_pos_native(window_name)


def show_frame(win_name: str, frame_img: np.ndarray, overlay: Dict[str, Any]) -> Tuple[int, bool]:
    """Compose, show and check the window visibility.

    Returns tuple `(key, window_alive)` where `key` is the result of `waitKey`.
    """
    display = Visualizer.compose_display(frame_img, win_name, overlay)
    cv2.imshow(win_name, display)
    key = cv2.waitKey(1) & 0xFF
    try:
        alive = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        alive = WindowManager.get_window_pos_native(win_name) is not None
    return key, alive


def draw_header(frame_img: np.ndarray, gesture_text: Optional[str], stable: Optional[str], gesture_window: Deque[Optional[str]], last_action_label: str, action_info: str) -> None:
    """Draw header UI elements on the provided frame in-place."""
    cv2.rectangle(frame_img, (0, 0), (frame_img.shape[1], HEADER_HEIGHT), (245, 245, 245), -1)
    cv2.putText(frame_img, f"Gesture: {gesture_text or 'none'}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    try:
        non_none = [g for g in gesture_window if g is not None]
        counts = Counter(non_none)
        votes = counts.get(stable, 0) if stable else 0
        stab_text = f"Stable: {stable or 'none'} ({votes}/{SMOOTHING_WINDOW})"
        text_size = cv2.getTextSize(stab_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        x = frame_img.shape[1] - text_size[0] - 12
        cv2.putText(frame_img, stab_text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 0) if stable else (120, 120, 120), 2)
    except Exception:
        pass
    cv2.putText(frame_img, f"Last action: {last_action_label} | {action_info}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    cv2.putText(frame_img, "1:P toggle | 2:C toggle | Open+Fist:N toggle", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)


def handle_gesture_action(stable_gesture: Optional[str], now_ts: float, state: Dict[str, Any]) -> str:
    """Encapsulate gesture -> app action logic and update `state` in-place.

    `state` must contain: `last_seen_open_time`, `last_action_time`, `last_action_label`, `last_triggered_gesture`, `app_manager`, `args`.
    Returns a human-readable `action_info` string describing the attempted action.
    """
    if stable_gesture is None:
        return "waiting"

    if stable_gesture == "open":
        state["last_seen_open_time"] = now_ts
        return "waiting"

    if stable_gesture in GESTURE_ACTIONS and stable_gesture != state.get("last_triggered_gesture") and (now_ts - state.get("last_action_time", 0.0)) >= state["args"].cooldown:
        if stable_gesture == "fist":
            if state.get("last_seen_open_time", 0.0) == 0.0 or (now_ts - state.get("last_seen_open_time", 0.0)) > SEQUENCE_WINDOW:
                return "waiting for open->fist sequence"
            else:
                app_key, operation = GESTURE_ACTIONS[stable_gesture]
                action_info_res, last_label, ok = state["app_manager"].perform_app_action(app_key, operation)
                if ok and last_label:
                    state["last_action_label"] = last_label
                    state["last_action_time"] = now_ts
                    state["last_triggered_gesture"] = stable_gesture
                    state["last_seen_open_time"] = 0.0
                return action_info_res
        else:
            app_key, operation = GESTURE_ACTIONS[stable_gesture]
            action_info_res, last_label, ok = state["app_manager"].perform_app_action(app_key, operation)
            if ok and last_label:
                state["last_action_label"] = last_label
            if ok:
                state["last_action_time"] = now_ts
                state["last_triggered_gesture"] = stable_gesture
            return action_info_res

    return "waiting"


def main() -> None:
    """Entry point: wire components and run the main loop."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Control simple de apps con gestos usando MediaPipe Hands")
    parser.add_argument("--source", default="0", help="Camera index (0,1,2...) or video path. Default: 0")
    parser.add_argument("--cooldown", type=float, default=2.5, help="Seconds between actions to avoid repeated triggers")
    args = parser.parse_args()

    source = CaptureManager.parse_source(args.source)
    cap, attempts = CaptureManager.open_capture(source)
    if cap is None:
        logger.error("Could not open camera/video source")
        for item in attempts:
            logger.error("  - %s", item)
        return

    app_manager = AppManager(APP_PRESETS)

    model_path = ModelManager.ensure_hand_model_downloaded()
    landmarker = ModelManager.create_landmarker(model_path=model_path, max_hands=1)

    last_action_time = 0.0
    last_action_label = "none"
    last_triggered_gesture = None
    last_seen_open_time = 0.0

    gesture_window: Deque[Optional[str]] = deque(maxlen=SMOOTHING_WINDOW)
    no_gesture_start = 0.0
    window_name = WINDOW_TITLE
    last_timestamp_ms = 0

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    overlay_state: Dict[str, Any] = {"enabled": True, "rect": (0, 0, 0, 0)}
    cv2.setMouseCallback(window_name, Visualizer.on_mouse, overlay_state)
    pos = WindowManager.load_window_pos()
    if pos:
        try:
            cv2.moveWindow(window_name, pos[0], pos[1])
        except Exception:
            pass

    logger.info("Gesture control ready")
    logger.info("Gesture mapping: 1->Paint toggle, 2->Calculator toggle, Open+Fist->Notepad toggle")

    

    try:
        state = {
            "last_seen_open_time": last_seen_open_time,
            "last_action_time": last_action_time,
            "last_action_label": last_action_label,
            "last_triggered_gesture": last_triggered_gesture,
            "app_manager": app_manager,
            "args": args,
        }

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

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
                gesture = GestureClassifier.classify_gesture(result.hand_landmarks[0], handedness_label)

            gesture_window.append(gesture)
            stable_gesture = GestureClassifier.get_stable_gesture(gesture_window)

            if overlay_state.get("enabled", True):
                try:
                    Visualizer.draw_detection_overlay(frame, result.hand_landmarks[0] if result.hand_landmarks else None, handedness_label)
                except Exception:
                    logger.debug("draw_detection_overlay failed", exc_info=True)

            now = time.monotonic()

            # Clear last-trigger on long no-gesture
            if stable_gesture is None:
                if no_gesture_start == 0.0:
                    no_gesture_start = now
                elif (now - no_gesture_start) > NO_GESTURE_CLEAR_TIME:
                    state["last_triggered_gesture"] = None
            else:
                no_gesture_start = 0.0

            action_info = handle_gesture_action(stable_gesture, now, state)

            # Update local variables from state (for persistence and display)
            last_action_time = state["last_action_time"]
            last_action_label = state["last_action_label"]
            last_triggered_gesture = state["last_triggered_gesture"]
            last_seen_open_time = state["last_seen_open_time"]

            draw_header(frame, gesture, stable_gesture, gesture_window, last_action_label, action_info)

            key, alive = show_frame(window_name, frame, overlay_state)
            if not alive:
                break

            if key == 27 or key in (ord("q"), ord("Q")):
                break
    finally:
        try:
            pos = WindowManager.get_window_pos(window_name)
            if pos:
                WindowManager.save_window_pos(pos[0], pos[1])
        except Exception:
            pass
        cap.release()
        try:
            landmarker.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
