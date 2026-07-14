"""
hand_mouse.py
-------------
Touchless virtual mouse controlled by hand gestures, using MediaPipe for hand
tracking, OpenCV for the camera feed, and PyAutoGUI to drive the system mouse.

Gestures
--------
Index finger only up                    -> Move cursor
Thumb + Index pinch (middle down)        -> Left click
  - hold the pinch                       -> Click-and-drag
  - two quick pinches                    -> Double click
Index + Middle up, pinch thumb to index  -> Right click
Index + Middle up, NOT pinching          -> Scroll (direction = hand height)

Controls
--------
q / ESC  -> quit
h        -> toggle on-screen overlay (landmarks + status text)
d        -> toggle debug numbers (live pinch ratio, finger states) so you can
            tune pinch_close / pinch_open in Config to match your own hand

Run:
    python hand_mouse.py
    python hand_mouse.py --camera 1
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
import pyautogui


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540

    # The active region of the camera frame that maps to the FULL screen.
    # Margins mean you don't have to reach the literal edge of the camera
    # frame to reach the edge of the screen.
    region_margin_x: float = 0.15
    region_margin_y: float = 0.15

    # Cursor smoothing. Higher = smoother movement but more lag.
    # Bumped up from 5 -> 8 to kill the jitter you saw on slight movement.
    smoothening: float = 8.0

    # Minimum pixel movement (in screen coordinates) before the cursor is
    # allowed to update at all. Filters out tiny landmark jitter from
    # MediaPipe so the cursor doesn't shake when your hand is basically still.
    cursor_dead_zone_px: float = 4.0

    # Pinch detection uses thumb-to-index distance DIVIDED by hand size
    # (wrist-to-middle-knuckle distance). This ratio stays roughly constant
    # regardless of camera resolution or how far your hand is from the
    # camera, unlike a fixed pixel threshold.
    # NOTE: if clicks/drags/right-click/scroll aren't firing at all, press
    # 'd' while running to see your live pinch ratio on screen, then adjust
    # these two numbers to match what you actually see (loosened from the
    # previous 0.45/0.65 since those were too strict).
    pinch_close: float = 0.60   # ratio below this  = considered "pinched"
    pinch_open: float = 0.70    # ratio above this  = considered "open"
    # (the gap between the two is a dead zone that prevents jitter)
    shape_debounce_frames: int = 3  # how many consecutive frames a gesture must hold before acting on it
    click_double_window: float = 0.6    # max seconds between pinches to count as a double-click
    drag_hold_time: float = 0.35        # how long a pinch must be held before it becomes a drag
    right_click_cooldown: float = 0.6   # min seconds between right-clicks
    scroll_cooldown: float = 0.05       # min seconds between scroll ticks
    scroll_amount: int = 60

    model_complexity: int = 1   # 0 = fastest, 1 = more accurate, less jittery landmarks
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.7


# --------------------------------------------------------------------------- #
# Hand detection helper
# --------------------------------------------------------------------------- #
class HandDetector:
    """Thin wrapper around MediaPipe Hands that returns convenient landmark data."""

    TIP_IDS = [8, 12, 16, 20]   # index, middle, ring, pinky fingertips
    PIP_IDS = [6, 10, 14, 18]   # the joint just below each fingertip

    def __init__(self, config: Config):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=1,
            model_complexity=config.model_complexity,
            min_detection_confidence=config.detection_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )
        self._draw_utils = mp.solutions.drawing_utils

    def process(self, frame_rgb):
        return self._hands.process(frame_rgb)

    def draw(self, frame, hand_landmarks) -> None:
        self._draw_utils.draw_landmarks(frame, hand_landmarks, self._mp_hands.HAND_CONNECTIONS)

    @staticmethod
    def fingers_up(landmarks) -> List[int]:
        """
        Returns [thumb, index, middle, ring, pinky], each 1 (extended) or 0
        (folded). Uses a small margin (not just tip.y < pip.y) so a finger
        that's borderline-curled doesn't flip-flop between up/down every
        frame — e.g. the middle finger naturally curls a bit while pinching
        thumb-to-index, which was causing LEFT CLICK / RIGHT CLICK to flicker.
        """
        margin = 0.02  # normalized (0-1) units; tune up if flicker persists
        fingers = [1 if landmarks[4].x < landmarks[3].x else 0]  # thumb (mirrored frame)
        for tip_id, pip_id in zip(HandDetector.TIP_IDS, HandDetector.PIP_IDS):
            fingers.append(1 if landmarks[tip_id].y < landmarks[pip_id].y - margin else 0)
        return fingers

    @staticmethod
    def hand_size_px(landmarks, frame_w: int, frame_h: int) -> float:
        """Wrist-to-middle-knuckle distance in pixels, used to normalize other distances."""
        wrist, mcp = landmarks[0], landmarks[9]
        return _pixel_distance(wrist, mcp, frame_w, frame_h)


def _pixel_distance(a, b, frame_w: int, frame_h: int) -> float:
    dx = (a.x - b.x) * frame_w
    dy = (a.y - b.y) * frame_h
    return math.hypot(dx, dy)


def map_to_screen(nx: float, ny: float, config: Config, screen_w: int, screen_h: int) -> tuple[float, float]:
    """Map a normalized (0-1) point inside the active region to full screen coordinates."""
    mx, my = config.region_margin_x, config.region_margin_y
    sx = (nx - mx) / (1 - 2 * mx)
    sy = (ny - my) / (1 - 2 * my)
    sx = min(max(sx, 0.0), 1.0)
    sy = min(max(sy, 0.0), 1.0)

    # Keep a 2px safety margin off the literal screen edges. PyAutoGUI's
    # built-in fail-safe (FAILSAFE = True) throws an exception and KILLS the
    # whole script the instant the cursor touches a true corner pixel (0,0
    # etc.) — this was almost certainly why the app appeared to randomly
    # freeze/stop mid-session, since the active region maps your hand across
    # the entire screen including its corners.
    x = min(max(sx * screen_w, 2), screen_w - 2)
    y = min(max(sy * screen_h, 2), screen_h - 2)
    return x, y


# --------------------------------------------------------------------------- #
# Gesture state machine -> mouse actions
# --------------------------------------------------------------------------- #
class GestureController:
    """
    Owns all gesture state (drag/pinch/cooldown timers) and turns finger
    positions into PyAutoGUI calls. Each gesture family is handled by its own
    method so cursor movement, clicking, dragging, and scrolling never step
    on each other.
    """

    def __init__(self, config: Config, screen_size: tuple[int, int]):
        self.cfg = config
        self.screen_w, self.screen_h = screen_size

        self.prev_x = self.prev_y = 0.0
        self._cursor_initialized = False

        # Left-click / drag / double-click state
        self.left_pinching = False
        self.left_pinch_start = 0.0
        self.dragging = False
        self.last_release_time = 0.0

        # Right-click state
        self.right_pinching = False
        self.last_right_click_time = 0.0

        self.last_scroll_time = 0.0

        # Exposed for the debug overlay so you can see the live numbers.
        self.last_pinch_ratio = 999.0

        # Debounce: require a gesture "shape" (one-finger vs two-finger) to
        # hold for a few consecutive frames before acting on it, so a single
        # noisy frame (e.g. middle finger briefly misread during a pinch)
        # can't flip LEFT CLICK into RIGHT CLICK or vice versa.
        self._pending_shape: Optional[str] = None
        self._pending_count = 0
        self._stable_shape: Optional[str] = None
        self.shape_debounce_frames = 3

        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = True

    def reset(self) -> None:
        """Called when no hand (or an unrecognized shape) is in frame, to avoid stuck states."""
        if self.dragging:
            pyautogui.mouseUp()
        self.dragging = False
        self.left_pinching = False
        self.right_pinching = False

    def update(self, landmarks, fingers: List[int], frame_w: int, frame_h: int,
               status: List[str]) -> None:
        now = time.time()
        index, thumb = landmarks[8], landmarks[4]

        hand_size = HandDetector.hand_size_px(landmarks, frame_w, frame_h)
        pinch_px = _pixel_distance(index, thumb, frame_w, frame_h)
        pinch_ratio = pinch_px / hand_size if hand_size > 1e-6 else 999.0
        self.last_pinch_ratio = pinch_ratio
        is_pinched = pinch_ratio < self.cfg.pinch_close
        is_open = pinch_ratio > self.cfg.pinch_open

        index_up, middle_up, ring_up, pinky_up = fingers[1], fingers[2], fingers[3], fingers[4]

        if index_up and middle_up and not ring_up and not pinky_up:
            raw_shape = "two_finger"
        elif index_up and not middle_up and not ring_up and not pinky_up:
            raw_shape = "one_finger"
        else:
            raw_shape = "none"

        shape = self._debounce_shape(raw_shape)

        if shape == "two_finger":
            # Two-finger shape: pinched = right click, open = scroll.
            self.left_pinching = False
            if is_pinched:
                self._handle_right_click(now, status)
            else:
                self.right_pinching = False
                if is_open:
                    self._handle_scroll(index.y, now, status)

        elif shape == "one_finger":
            # One-finger shape: cursor always follows the fingertip; pinching
            # (independently) drives click / drag / double-click.
            self.right_pinching = False
            self._move_cursor(index.x, index.y, status)
            self._handle_left_family(is_pinched, now, status)

        else:
            self.reset()

    def _debounce_shape(self, raw_shape: str) -> str:
        """
        Only switches the "stable" shape (the one actually acted on) once
        raw_shape has been seen for shape_debounce_frames in a row. This
        absorbs single-frame misreads (e.g. middle finger briefly flickering
        up/down mid-pinch) without adding noticeable input lag.
        """
        if raw_shape == self._pending_shape:
            self._pending_count += 1
        else:
            self._pending_shape = raw_shape
            self._pending_count = 1

        if self._pending_count >= self.shape_debounce_frames:
            self._stable_shape = raw_shape

        # First-ever reading: don't wait, just adopt it immediately.
        if self._stable_shape is None:
            self._stable_shape = raw_shape

        return self._stable_shape

    # -- gesture handlers ---------------------------------------------------

    def _move_cursor(self, nx: float, ny: float, status: List[str]) -> None:
        target_x, target_y = map_to_screen(nx, ny, self.cfg, self.screen_w, self.screen_h)
        if not self._cursor_initialized:
            self.prev_x, self.prev_y = target_x, target_y
            self._cursor_initialized = True

        smooth_x = self.prev_x + (target_x - self.prev_x) / self.cfg.smoothening
        smooth_y = self.prev_y + (target_y - self.prev_y) / self.cfg.smoothening

        # Dead zone: ignore movements too small to be intentional, so the
        # cursor doesn't shake from MediaPipe's natural landmark jitter when
        # your hand is basically still.
        moved = math.hypot(smooth_x - self.prev_x, smooth_y - self.prev_y)
        if moved < self.cfg.cursor_dead_zone_px:
            status.append("MOVE (idle)")
            return

        pyautogui.moveTo(smooth_x, smooth_y)
        self.prev_x, self.prev_y = smooth_x, smooth_y
        status.append("MOVE")

    def _handle_left_family(self, is_pinched: bool, now: float, status: List[str]) -> None:
        if is_pinched:
            if not self.left_pinching:
                self.left_pinching = True
                self.left_pinch_start = now
            elif now - self.left_pinch_start > self.cfg.drag_hold_time and not self.dragging:
                pyautogui.mouseDown()
                self.dragging = True
            if self.dragging:
                status.append("DRAG")
            else:
                status.append("PINCH")
        else:
            if self.left_pinching:
                hold_time = now - self.left_pinch_start
                if self.dragging:
                    pyautogui.mouseUp()
                    self.dragging = False
                    status.append("DROP")
                elif hold_time < self.cfg.drag_hold_time:
                    if now - self.last_release_time < self.cfg.click_double_window:
                        pyautogui.doubleClick()
                        status.append("DOUBLE CLICK")
                    else:
                        pyautogui.click()
                        status.append("LEFT CLICK")
                    self.last_release_time = now
            self.left_pinching = False

    def _handle_right_click(self, now: float, status: List[str]) -> None:
        if not self.right_pinching and now - self.last_right_click_time > self.cfg.right_click_cooldown:
            pyautogui.rightClick()
            self.last_right_click_time = now
            status.append("RIGHT CLICK")
        self.right_pinching = True

    def _handle_scroll(self, index_y: float, now: float, status: List[str]) -> None:
        if now - self.last_scroll_time < self.cfg.scroll_cooldown:
            return
        if index_y < 0.45:
            pyautogui.scroll(self.cfg.scroll_amount)
            status.append("SCROLL UP")
        elif index_y > 0.55:
            pyautogui.scroll(-self.cfg.scroll_amount)
            status.append("SCROLL DOWN")
        else:
            return
        self.last_scroll_time = now


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def draw_active_region(frame, config: Config) -> None:
    h, w, _ = frame.shape
    x1, y1 = int(w * config.region_margin_x), int(h * config.region_margin_y)
    x2, y2 = int(w * (1 - config.region_margin_x)), int(h * (1 - config.region_margin_y))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)


def draw_status(frame, status_lines: List[str], fps: float) -> None:
    y = 40
    for line in status_lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 35
    h, w, _ = frame.shape
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_debug(frame, config: Config, controller: "GestureController", fingers: Optional[List[int]]) -> None:
    """
    Shows live numbers so you can tune Config.pinch_close / pinch_open to
    your own hand instead of guessing. Toggle with 'd'.
    """
    h, w, _ = frame.shape
    lines = [
        f"pinch_ratio: {controller.last_pinch_ratio:.2f}  "
        f"(close<{config.pinch_close:.2f}  open>{config.pinch_open:.2f})",
    ]
    if fingers is not None:
        names = ["thumb", "index", "middle", "ring", "pinky"]
        state = " ".join(f"{n}:{v}" for n, v in zip(names, fingers))
        lines.append(state)

    y = h - 20 * len(lines) - 10
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        y += 22


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Hand-gesture virtual mouse")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=960, help="Capture width")
    parser.add_argument("--height", type=int, default=540, help="Capture height")
    parser.add_argument("--smoothening", type=float, default=8.0, help="Cursor smoothing factor")
    args = parser.parse_args()

    return Config(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        smoothening=args.smoothening,
    )


def main() -> None:
    config = parse_args()

    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {config.camera_index}. "
            "Check that a webcam is connected and not in use by another app."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    screen_w, screen_h = pyautogui.size()
    detector = HandDetector(config)
    controller = GestureController(config, (screen_w, screen_h))

    show_overlay = True
    show_debug = False
    prev_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Warning: failed to read a frame from the camera.")
                continue

            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)

            status_lines: List[str] = []
            current_fingers: Optional[List[int]] = None

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                landmarks = hand_landmarks.landmark
                fingers = HandDetector.fingers_up(landmarks)
                current_fingers = fingers

                try:
                    controller.update(landmarks, fingers, frame_w, frame_h, status_lines)
                except pyautogui.FailSafeException:
                    # Cursor hit a screen corner and PyAutoGUI's safety kill
                    # switch fired. Don't let it take the whole app down —
                    # just drop this frame's action and keep going.
                    status_lines.append("FAILSAFE TRIGGERED (cursor hit corner)")
                    controller.reset()

                if show_overlay:
                    detector.draw(frame, hand_landmarks)
                    index_px = (int(landmarks[8].x * frame_w), int(landmarks[8].y * frame_h))
                    cv2.circle(frame, index_px, 10, (255, 0, 255), -1)
            else:
                controller.reset()

            if show_overlay:
                draw_active_region(frame, config)
                now = time.time()
                fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
                prev_time = now
                draw_status(frame, status_lines, fps)

            if show_debug:
                draw_debug(frame, config, controller, current_fingers)

            cv2.imshow("Virtual Mouse", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):  # ESC or q
                break
            if key == ord("h"):
                show_overlay = not show_overlay
            if key == ord("d"):
                show_debug = not show_debug

    finally:
        controller.reset()  # release a stuck drag if we exit mid-gesture
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

