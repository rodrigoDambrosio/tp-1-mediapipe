DEV NOTES — Gesture thresholds and tuning
=======================================

This file documents the primary thresholds and how to tune them.

Constants
---------
- `Y_DELTA_THRESHOLD` (0.015): vertical delta between tip and pip joints used to
  consider a finger "extended". Increase to make extension detection stricter.

- `THUMB_X_DELTA_THRESHOLD` (0.015) and `THUMB_X_FALLBACK` (0.05): used to
  detect thumb extension in the X axis. For side-facing hands, adjust these
  upwards if thumbs are misclassified.

- `AVG_TIP_WRIST_THRESHOLD` (0.22): average distance of finger tips to wrist
  used to discriminate open versus compact hands. Decrease if open hands are
  not being recognized at typical camera distances; increase if false
  positives occur.

- `SMOOTHING_WINDOW` (5) and `SMOOTHING_THRESHOLD` (3): sliding window length
  and vote threshold for the majority-based stable gesture. Increase the
  window for more stability but higher latency.

- `SEQUENCE_WINDOW` (2.0): seconds allowed between an `open` and `fist` to
  trigger the open->fist gesture used for the Notepad toggle.

Tuning recommendations
----------------------
1. Enable debug overlay and watch the wrist radius (`r=...`) and per-finger
   EXT/FLX labels to calibrate `Y_DELTA_THRESHOLD` and `AVG_TIP_WRIST_THRESHOLD`.

2. If gestures trigger too easily, raise `SMOOTHING_THRESHOLD` or increase
   `SMOOTHING_WINDOW`.
""