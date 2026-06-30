"""
test_gestures.py
-----------------
Safe diagnostic tool for hand_mouse.py. Shows everything the real app sees
(detection status, finger states, pinch ratio, which gesture WOULD fire) but
NEVER touches your actual mouse cursor or clicks anything — so you can test
freely without losing control of your computer mid-test.

Use this BEFORE running hand_mouse.py for real, to:
  1. Confirm your camera/lighting lets MediaPipe detect your hand at all.
  2. Find the right pinch_close / pinch_open values for YOUR hand.
  3. Confirm each gesture shape is recognized correctly.

Run: python test_gestures.py
Press ESC or q to quit.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional

import cv2
import mediapipe as mp

# --------------------------------------------------------------------------- #
# Same thresholds as hand_mouse.py's Config — edit these to match what you're
# testing there, so the numbers you see here transfer directly.
# --------------------------------------------------------------------------- #
PINCH_CLOSE = 0.55
PINCH_OPEN = 0.75
DETECTION_CONFIDENCE = 0.5   # lowered for this test to make detection easier
TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 1

TIP_IDS = [8, 12, 16, 20]
PIP_IDS = [6, 10, 14, 18]


def fingers_up(landmarks) -> List[int]:
    margin = 0.02  # matches hand_mouse.py's HandDetector.fingers_up
    fingers = [1 if landmarks[4].x < landmarks[3].x else 0]
    for tip_id, pip_id in zip(TIP_IDS, PIP_IDS):
        fingers.append(1 if landmarks[tip_id].y < landmarks[pip_id].y - margin else 0)
    return fingers


def pixel_distance(a, b, w: int, h: int) -> float:
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)


SHAPE_DEBOUNCE_FRAMES = 3


class ShapeDebouncer:
    """Mirrors GestureController._debounce_shape() in hand_mouse.py."""

    def __init__(self, frames_required: int = SHAPE_DEBOUNCE_FRAMES):
        self.frames_required = frames_required
        self._pending_shape: Optional[str] = None
        self._pending_count = 0
        self._stable_shape: Optional[str] = None

    def update(self, raw_shape: str) -> str:
        if raw_shape == self._pending_shape:
            self._pending_count += 1
        else:
            self._pending_shape = raw_shape
            self._pending_count = 1

        if self._pending_count >= self.frames_required:
            self._stable_shape = raw_shape
        if self._stable_shape is None:
            self._stable_shape = raw_shape
        return self._stable_shape


def raw_shape_from_fingers(fingers: List[int]) -> str:
    index_up, middle_up, ring_up, pinky_up = fingers[1], fingers[2], fingers[3], fingers[4]
    if index_up and middle_up and not ring_up and not pinky_up:
        return "two_finger"
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "one_finger"
    return "none"


def classify_gesture(fingers: List[int], pinch_ratio: float, stable_shape: str) -> str:
    """Mirrors the shape logic in GestureController.update(), read-only."""
    is_pinched = pinch_ratio < PINCH_CLOSE
    is_open = pinch_ratio > PINCH_OPEN

    if all(fingers):
        return "OPEN PALM"
    if not any(fingers):
        return "FIST"

    if stable_shape == "two_finger":
        if is_pinched:
            return "RIGHT CLICK shape (pinched)"
        if is_open:
            return "SCROLL shape (open)"
        return "TWO FINGERS (in pinch dead-zone, no action)"

    if stable_shape == "one_finger":
        if is_pinched:
            return "LEFT CLICK/DRAG shape (pinched)"
        return "MOVE shape (index only, not pinched)"

    return "UNRECOGNIZED shape"


def main() -> None:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera (index 0). Check it's connected and free.")

    prev_time = time.time()
    min_ratio_seen = 999.0
    max_ratio_seen = 0.0
    debouncer = ShapeDebouncer()

    print("=" * 60)
    print("SAFE GESTURE TEST — mouse will NOT move or click.")
    print("Watch the on-screen numbers. Press ESC or q to quit.")
    print("=" * 60)

    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0].landmark
            mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            fingers = fingers_up(landmarks)
            wrist, mcp = landmarks[0], landmarks[9]
            hand_size = pixel_distance(wrist, mcp, w, h)
            thumb, index = landmarks[4], landmarks[8]
            pinch_px = pixel_distance(index, thumb, w, h)
            pinch_ratio = pinch_px / hand_size if hand_size > 1e-6 else 999.0

            min_ratio_seen = min(min_ratio_seen, pinch_ratio)
            max_ratio_seen = max(max_ratio_seen, pinch_ratio)

            raw_shape = raw_shape_from_fingers(fingers)
            stable_shape = debouncer.update(raw_shape)
            gesture = classify_gesture(fingers, pinch_ratio, stable_shape)

            cv2.putText(frame, "HAND DETECTED", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {gesture}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"pinch_ratio: {pinch_ratio:.2f}  (close<{PINCH_CLOSE} open>{PINCH_OPEN})",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            cv2.putText(frame, f"min seen: {min_ratio_seen:.2f}   max seen: {max_ratio_seen:.2f}",
                        (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

            names = ["thumb", "index", "middle", "ring", "pinky"]
            state = " ".join(f"{n}:{v}" for n, v in zip(names, fingers))
            cv2.putText(frame, state, (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "NO HAND DETECTED", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "-> check lighting / distance / background",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.putText(frame, f"FPS: {int(fps)}", (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Gesture Test (safe - no mouse control)", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("=" * 60)
    print(f"Pinch ratio range observed this session: {min_ratio_seen:.2f} - {max_ratio_seen:.2f}")
    print(f"  -> Set pinch_close in Config to roughly {min_ratio_seen + 0.05:.2f}")
    print(f"  -> Set pinch_open  in Config to roughly {max_ratio_seen - 0.05:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()