"""
test_gestures_metrics.py
-------------------------
Metrics-logging twin of test_gestures.py. Mirrors the exact detection,
debouncing, and classification logic used in hand_mouse.py / test_gestures.py,
but instead of just printing gestures to screen, it logs one row per frame to
a CSV that analyze_metrics.py can consume directly.

Like test_gestures.py, this NEVER touches your real mouse.

How labeling works
-------------------
Hold one of these keys while you perform the matching gesture, so the frame
gets tagged with ground truth. Release the key (or press nothing) and the
frame is logged as "unlabeled" instead, which analyze_metrics.py's accuracy
report simply excludes.

    m  -> MOVE               (index-only, not pinched)
    l  -> LEFT_CLICK_FAMILY  (index-only, pinched / drag / double-click)
    r  -> RIGHT_CLICK        (two-finger, pinched)
    s  -> SCROLL             (two-finger, open)
    i  -> IDLE                (no hand / fist / open palm / anything else)

Because cv2.waitKey only reports a single keypress per call and doesn't give
a clean "key is currently held down" signal on all platforms, this script
uses a simple held-until-next-press model: press a label key once to start
labeling frames with it, press it again (or press a different label key) to
switch, press SPACE to go back to "unlabeled".

IMPORTANT - keyboard focus:
cv2.waitKey() only receives keypresses while the OpenCV preview window has
OS keyboard focus. Gesturing naturally pulls your hand (and often your
mouse, which steals focus on some window managers) away from that window,
so keys pressed mid-gesture can silently go nowhere -- every frame then
gets logged with whatever label was active before you started gesturing.
To avoid this, the script uses `pynput` (if installed) to listen for label
keys globally, independent of which window is focused. If pynput isn't
available it falls back to the old window-focused method and prints a
warning so you know to keep the preview window focused while labeling.

    pip install pynput

Run:
    python test_gestures_metrics.py
    python test_gestures_metrics.py --out my_session.csv --duration 120

Press ESC or q to stop early and save whatever was logged. Watch the
console -- every label change is printed there as "[label] -> X" so you
can confirm in real time that your keypresses are actually registering.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from typing import List, Optional

import cv2
import mediapipe as mp

try:
    from pynput import keyboard as pynput_keyboard
    HAVE_PYNPUT = True
except ImportError:
    HAVE_PYNPUT = False

# --------------------------------------------------------------------------- #
# Same thresholds/logic as test_gestures.py / hand_mouse.py — keep these in
# sync if you tune Config in hand_mouse.py, so the metrics stay meaningful.
# --------------------------------------------------------------------------- #
PINCH_CLOSE = 0.55
PINCH_OPEN = 0.75
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 1
SHAPE_DEBOUNCE_FRAMES = 3

TIP_IDS = [8, 12, 16, 20]
PIP_IDS = [6, 10, 14, 18]

CATEGORIES = ["MOVE", "LEFT_CLICK_FAMILY", "RIGHT_CLICK", "SCROLL", "IDLE"]
WINDOW_SECONDS = 15.0  # default per-gesture recording window for --guided mode; overridden by --window-seconds

# Maps the label keys to the ground-truth category strings analyze_metrics.py
# expects (see CATEGORIES in that script).
LABEL_KEYS = {
    ord("m"): "MOVE",
    ord("l"): "LEFT_CLICK_FAMILY",
    ord("r"): "RIGHT_CLICK",
    ord("s"): "SCROLL",
    ord("i"): "IDLE",
}
CSV_FIELDS = [
    "timestamp",
    "fps",
    "frame_time_ms",
    "hand_detected",
    "pinch_ratio",
    "stable_shape",
    "ground_truth",
    "predicted_category",
]


class GlobalLabelListener:
    """
    Captures label keypresses independent of which window has OS focus.

    Needed because performing hand gestures naturally pulls your hand (and
    often mouse-follow-focus) away from the OpenCV preview window, which
    silently stops cv2.waitKey() from ever seeing your keypresses -- this is
    why earlier sessions ended up with every single frame carrying whatever
    label happened to be active before gesturing started.

    Prints every label change to the console so you can visually confirm
    keypresses are registering in real time.
    """

    def __init__(self):
        self.current_label = "unlabeled"
        self._listener = pynput_keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key) -> None:
        label: Optional[str] = None
        if key == pynput_keyboard.Key.space:
            label = "unlabeled"
        else:
            char = getattr(key, "char", None)
            if char:
                label = LABEL_KEYS.get(ord(char.lower()))
        if label is not None and label != self.current_label:
            print(f"[label] -> {label}")
            self.current_label = label

    def stop(self) -> None:
        self._listener.stop()


def fingers_up(landmarks) -> List[int]:
    margin = 0.02  # matches hand_mouse.py's HandDetector.fingers_up
    fingers = [1 if landmarks[4].x < landmarks[3].x else 0]
    for tip_id, pip_id in zip(TIP_IDS, PIP_IDS):
        fingers.append(1 if landmarks[tip_id].y < landmarks[pip_id].y - margin else 0)
    return fingers


def pixel_distance(a, b, w: int, h: int) -> float:
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)


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


def predicted_category(stable_shape: str, pinch_ratio: float) -> str:
    """
    Maps the same rule-based logic hand_mouse.py's GestureController.update()
    uses into one of analyze_metrics.py's CATEGORIES, so predictions and
    ground-truth labels live in the same label space.
    """
    is_pinched = pinch_ratio < PINCH_CLOSE
    is_open = pinch_ratio > PINCH_OPEN

    if stable_shape == "two_finger":
        if is_pinched:
            return "RIGHT_CLICK"
        if is_open:
            return "SCROLL"
        return "IDLE"  # dead-zone: two fingers up but ambiguous pinch state
    if stable_shape == "one_finger":
        return "LEFT_CLICK_FAMILY" if is_pinched else "MOVE"
    return "IDLE"


def parse_args():
    p = argparse.ArgumentParser(description="Log gesture metrics to CSV (no mouse control)")
    p.add_argument("--out", type=str, default="gesture_metrics.csv", help="Output CSV path")
    p.add_argument("--camera", type=int, default=0, help="Camera device index")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Auto-stop after N seconds (0 = run until q/ESC). Ignored in --guided mode.")
    p.add_argument("--guided", action="store_true",
                   help="Timed protocol: press ENTER before each gesture, then perform it "
                        "hands-free for a fixed window while labels apply automatically. "
                        "No keypress needed during gesturing -- avoids window-focus issues "
                        "entirely and doesn't require pynput.")
    p.add_argument("--window-seconds", type=float, default=15.0,
                   help="Seconds to record per gesture in --guided mode (default 15)")
    return p.parse_args()


GUIDED_SEQUENCE = [
    ("MOVE", "Point with index finger only and move your hand around the frame."),
    ("LEFT_CLICK_FAMILY", "Pinch thumb-to-index repeatedly (include a hold-drag and a double-pinch)."),
    ("RIGHT_CLICK", "Hold up index+middle, pinch thumb-to-index a few times."),
    ("SCROLL", "Hold up index+middle, keep them open (not pinched), move hand up/down."),
    ("IDLE", "Rest your hand naturally / keep it out of frame / make a fist."),
]


def run_guided_labeling(cap, hands, mp_draw, mp_hands) -> list:
    """
    Terminal-driven timed protocol. You press ENTER once per gesture, while
    your hands are still free, THEN perform the gesture -- no key ever has
    to register while you're mid-gesture, so this works even if pynput is
    unavailable or blocked by OS permissions (macOS Accessibility, Wayland).
    """
    rows = []
    debouncer = ShapeDebouncer()
    prev_time = time.time()

    for label, instructions in GUIDED_SEQUENCE:
        print("\n" + "-" * 60)
        print(f"NEXT: {label}")
        print(f"  {instructions}")
        input("  Press ENTER, THEN start the gesture immediately... ")
        print(f"  Recording {label} now...")
        window_start = time.time()

        while True:
            loop_start = time.time()
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

            hand_detected = False
            pinch_ratio = -1.0
            stable_shape = "none"
            pred = "IDLE"

            if result.multi_hand_landmarks:
                hand_detected = True
                landmarks = result.multi_hand_landmarks[0].landmark
                mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                fingers = fingers_up(landmarks)
                wrist, mcp = landmarks[0], landmarks[9]
                hand_size = pixel_distance(wrist, mcp, w, h)
                thumb, index = landmarks[4], landmarks[8]
                pinch_px = pixel_distance(index, thumb, w, h)
                pinch_ratio = pinch_px / hand_size if hand_size > 1e-6 else 999.0
                raw_shape = raw_shape_from_fingers(fingers)
                stable_shape = debouncer.update(raw_shape)
                pred = predicted_category(stable_shape, pinch_ratio)
            else:
                debouncer = ShapeDebouncer()

            frame_time_ms = (time.time() - loop_start) * 1000.0
            elapsed = now - window_start
            remaining = WINDOW_SECONDS - elapsed

            rows.append({
                "timestamp": now,
                "fps": round(fps, 2),
                "frame_time_ms": round(frame_time_ms, 3),
                "hand_detected": hand_detected,
                "pinch_ratio": round(pinch_ratio, 4),
                "stable_shape": stable_shape,
                "ground_truth": label,
                "predicted_category": pred,
            })

            cv2.putText(frame, f"RECORDING: {label}  ({remaining:0.1f}s left)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"PRED: {pred}" if hand_detected else "PRED: (no hand)",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Gesture Metrics Logger (safe - no mouse control)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("Stopped early by user.")
                return rows

            if elapsed >= WINDOW_SECONDS:
                break

    return rows


def main() -> None:
    args = parse_args()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}. Check it's connected and free.")

    if args.guided:
        global WINDOW_SECONDS
        WINDOW_SECONDS = args.window_seconds
        print("=" * 60)
        print("GUIDED LABELING MODE — no keypresses needed during gestures.")
        print(f"Each gesture is recorded for {WINDOW_SECONDS:.0f}s after you press ENTER.")
        print("Press q or ESC anytime to stop early and save what's logged so far.")
        print("=" * 60)
        rows = run_guided_labeling(cap, hands, mp_draw, mp_hands)
        cap.release()
        cv2.destroyAllWindows()

        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        print("=" * 60)
        print(f"Saved {len(rows)} frames to {args.out}")
        labeled = sum(1 for r in rows if r["ground_truth"] != "unlabeled")
        print(f"Labeled frames: {labeled} / {len(rows)}")
        print(f"Next: python analyze_metrics.py --csv {args.out}")
        print("=" * 60)
        return

    debouncer = ShapeDebouncer()
    current_label = "unlabeled"
    rows = []

    prev_time = time.time()
    start_time = prev_time

    label_listener: Optional[GlobalLabelListener] = None
    if HAVE_PYNPUT:
        label_listener = GlobalLabelListener()

    print("=" * 60)
    print("METRICS LOGGING — mouse will NOT move or click.")
    print("Press m/l/r/s/i to label the current gesture, SPACE to clear label.")
    if HAVE_PYNPUT:
        print("Global key capture ON (pynput) — labels register even if the")
        print("preview window isn't focused. Watch console for '[label] -> X'.")
    else:
        print("WARNING: pynput not installed, falling back to window-focused")
        print("keys — you MUST keep the preview window focused while pressing")
        print("label keys, or your labels won't register at all.")
        print("Install with: pip install pynput")
    print("Press ESC or q to stop and save.")
    print("=" * 60)

    try:
        while True:
            loop_start = time.time()
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

            hand_detected = False
            pinch_ratio = -1.0
            stable_shape = "none"
            pred = "IDLE"

            if result.multi_hand_landmarks:
                hand_detected = True
                landmarks = result.multi_hand_landmarks[0].landmark
                mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                fingers = fingers_up(landmarks)
                wrist, mcp = landmarks[0], landmarks[9]
                hand_size = pixel_distance(wrist, mcp, w, h)
                thumb, index = landmarks[4], landmarks[8]
                pinch_px = pixel_distance(index, thumb, w, h)
                pinch_ratio = pinch_px / hand_size if hand_size > 1e-6 else 999.0

                raw_shape = raw_shape_from_fingers(fingers)
                stable_shape = debouncer.update(raw_shape)
                pred = predicted_category(stable_shape, pinch_ratio)
            else:
                # No hand -> treat as a fresh debounce state next time a hand appears
                debouncer = ShapeDebouncer()

            frame_time_ms = (time.time() - loop_start) * 1000.0

            if label_listener is not None:
                current_label = label_listener.current_label

            rows.append({
                "timestamp": now,
                "fps": round(fps, 2),
                "frame_time_ms": round(frame_time_ms, 3),
                "hand_detected": hand_detected,
                "pinch_ratio": round(pinch_ratio, 4),
                "stable_shape": stable_shape,
                "ground_truth": current_label,
                "predicted_category": pred,
            })

            # ---- overlay ----
            cv2.putText(frame, f"LABEL: {current_label}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"PRED:  {pred}" if hand_detected else "PRED:  (no hand)",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {int(fps)}   frames logged: {len(rows)}",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("Gesture Metrics Logger (safe - no mouse control)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if label_listener is None:
                # Fallback path only (no pynput): window must be focused.
                if key == ord(" "):
                    current_label = "unlabeled"
                elif key in LABEL_KEYS:
                    current_label = LABEL_KEYS[key]

            if args.duration > 0 and (now - start_time) >= args.duration:
                print(f"Reached --duration {args.duration}s, stopping.")
                break

    finally:
        if label_listener is not None:
            label_listener.stop()
        cap.release()
        cv2.destroyAllWindows()

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 60)
    print(f"Saved {len(rows)} frames to {args.out}")
    labeled = sum(1 for r in rows if r["ground_truth"] != "unlabeled")
    print(f"Labeled frames: {labeled} / {len(rows)}")
    print(f"Next: python analyze_metrics.py --csv {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()