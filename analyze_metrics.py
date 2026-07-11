"""
analyze_metrics.py
-------------------
Reads the CSV produced by test_gestures_metrics.py and prints a full metrics
report:
  - FPS (mean / min / max / std)
  - Frame processing time (mean / p95)
  - Hand detection rate
  - Pinch ratio jitter (std dev of ratio while a stable one_finger/two_finger
    shape is held with no ground-truth transition — good proxy for how noisy
    your setup is)
  - Gesture recognition accuracy: confusion matrix + per-class precision/
    recall/F1, computed only over labeled frames (ground_truth != unlabeled)

Requires: pandas, scikit-learn, numpy
    pip install pandas scikit-learn numpy

Run:
    python analyze_metrics.py --csv gesture_metrics.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


CATEGORIES = ["MOVE", "LEFT_CLICK_FAMILY", "RIGHT_CLICK", "SCROLL", "IDLE"]


def parse_args():
    p = argparse.ArgumentParser(description="Analyze gesture metrics CSV")
    p.add_argument("--csv", type=str, required=True, help="Path to CSV from test_gestures_metrics.py")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    print("=" * 70)
    print(f"METRICS REPORT: {args.csv}")
    print("=" * 70)

    # ---- FPS / timing ----
    print("\n--- Performance ---")
    print(f"Frames logged:        {len(df)}")
    print(f"Mean FPS:              {df['fps'].mean():.2f}")
    print(f"Min FPS:                {df['fps'].min():.2f}")
    print(f"Max FPS:                {df['fps'].max():.2f}")
    print(f"Std FPS:                {df['fps'].std():.2f}")
    print(f"Mean frame time:        {df['frame_time_ms'].mean():.2f} ms")
    print(f"P95 frame time:         {np.percentile(df['frame_time_ms'], 95):.2f} ms")

    # ---- Detection rate ----
    print("\n--- Hand Detection ---")
    detection_rate = df["hand_detected"].mean() * 100
    print(f"Detection rate:         {detection_rate:.1f}% ({df['hand_detected'].sum()}/{len(df)} frames)")

    # ---- Pinch ratio jitter ----
    print("\n--- Pinch Ratio Stability ---")
    detected = df[df["hand_detected"] == True]  # noqa: E712
    valid_ratio = detected[detected["pinch_ratio"] >= 0]
    if len(valid_ratio) > 0:
        print(f"Pinch ratio mean:       {valid_ratio['pinch_ratio'].mean():.3f}")
        print(f"Pinch ratio std (all):  {valid_ratio['pinch_ratio'].std():.3f}")
        # jitter within each contiguous run of the same (stable_shape, ground_truth)
        # gives a cleaner "how noisy is the ratio while doing ONE thing" number
        runs = (valid_ratio["stable_shape"] != valid_ratio["stable_shape"].shift()).cumsum()
        run_stds = valid_ratio.groupby(runs)["pinch_ratio"].std().dropna()
        if len(run_stds) > 0:
            print(f"Mean within-run std:    {run_stds.mean():.3f}  (lower = more stable pinch reading)")
    else:
        print("No valid pinch ratio samples (hand never detected).")

    # ---- Gesture accuracy ----
    print("\n--- Gesture Recognition Accuracy ---")
    labeled = df[df["ground_truth"] != "unlabeled"].copy()
    if len(labeled) == 0:
        print("No labeled frames found — re-run test_gestures_metrics.py and hold")
        print("a label key (m/l/r/s/i) while performing each gesture.")
    else:
        y_true = labeled["ground_truth"]
        y_pred = labeled["predicted_category"]
        present = sorted(set(y_true) | set(y_pred), key=lambda c: CATEGORIES.index(c) if c in CATEGORIES else 99)

        print(f"Labeled frames used:    {len(labeled)} / {len(df)} total")
        print(f"Overall accuracy:       {(y_true == y_pred).mean() * 100:.1f}%\n")

        print("Confusion matrix (rows=ground truth, cols=predicted):")
        cm = confusion_matrix(y_true, y_pred, labels=present)
        header = "".join(f"{c[:10]:>12}" for c in present)
        print(" " * 20 + header)
        for label, row in zip(present, cm):
            print(f"{label[:18]:>18}  " + "".join(f"{v:>12}" for v in row))

        print("\nPer-class precision/recall/F1:")
        print(classification_report(y_true, y_pred, labels=present, zero_division=0))

        # False-positive rate specifically for the LEFT vs RIGHT click
        # confusion your debounce logic was designed to prevent
        lr = labeled[labeled["ground_truth"].isin(["LEFT_CLICK_FAMILY", "RIGHT_CLICK"])]
        if len(lr) > 0:
            mix = lr[lr["ground_truth"] != lr["predicted_category"]]
            print(f"\nLeft/Right-click family misfires: {len(mix)}/{len(lr)} "
                  f"({len(mix)/len(lr)*100:.1f}%) — flicker between click types")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()