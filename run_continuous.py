#!/usr/bin/env python
"""Real-time continuous sign recognition from MediaPipe landmarks + CSLR model.

Usage:
  python run_continuous.py
  python run_continuous.py --model artifacts/cslr/lstm_best.keras
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.config import CSLR_DEFAULT_MODEL, CSLR_LABELS, CSLR_META  # noqa: E402
from app.cslr.continuous import ContinuousRecognizer  # noqa: E402
from app.landmarks import LandmarkTracker, primary_hand  # noqa: E402


def run(
    model_path: Path | None = None,
    meta_path: Path | None = None,
    camera: int = 0,
    arch: str = "lstm",
) -> None:
    model_path = model_path or CSLR_DEFAULT_MODEL
    meta_path = meta_path or CSLR_META
    labels = list(CSLR_LABELS)
    seq_len = 48
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        labels = meta.get("labels", labels)
        seq_len = int(meta.get("seq_len", seq_len))
        arch = meta.get("architecture", arch)

    if not model_path.exists():
        print(
            f"Model not found at {model_path}.\n"
            "Generate data + train first:\n"
            "  python scripts/generate_synthetic_sequences.py\n"
            "  python scripts/train_cslr.py --arch lstm\n"
            "Running with an untrained scaffold (predictions will be random)."
        )

    recognizer = ContinuousRecognizer.from_saved(
        model_path,
        labels,
        architecture=arch,
        seq_len=seq_len,
    )
    tracker = LandmarkTracker(max_hands=1)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")

    print("Continuous mode — Q to quit, C to clear sentence")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        hands = tracker.process(frame)
        hand = primary_hand(hands)
        pred = None
        if hand:
            for px, py in hand.landmarks_px:
                cv2.circle(frame, (px, py), 3, (0, 200, 120), -1)
            x1, y1, x2, y2 = hand.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 180, 90), 2)
            pred = recognizer.push_landmarks(hand.landmarks, handedness=hand.handedness)

        live = f"{pred.label} {pred.confidence:.0%}" if pred else "…"
        cv2.putText(frame, live, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 220, 120), 2)
        cv2.putText(
            frame,
            recognizer.sentence() or "—",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (240, 240, 240),
            2,
        )
        cv2.putText(
            frame,
            "continuous CSLR · MediaPipe landmarks",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )
        cv2.imshow("Continuous Sign Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("c"):
            recognizer.reset()

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Continuous Sign Language Recognition")
    parser.add_argument("--model", type=Path, default=CSLR_DEFAULT_MODEL)
    parser.add_argument("--meta", type=Path, default=CSLR_META)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--arch", default="lstm")
    args, _ = parser.parse_known_args()
    run(model_path=args.model, meta_path=args.meta, camera=args.camera, arch=args.arch)


if __name__ == "__main__":
    main()
