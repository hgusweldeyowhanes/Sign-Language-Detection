#!/usr/bin/env python
"""Record MediaPipe landmark sequences for continuous SLR training.

Controls:
  1-8 / keys mapped to gloss labels — select active class
  SPACE — start/stop recording a clip
  S — save last clip
  Q / ESC — quit

Saves NPZ sequences under data/sequences/ and updates manifest.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import CSLR_LABELS  # noqa: E402
from app.cslr.features import LANDMARK_DIM, landmarks_to_vector  # noqa: E402
from app.landmarks import LandmarkTracker, primary_hand  # noqa: E402

DEFAULT_LABELS = list(CSLR_LABELS)


def load_or_init_manifest(data_dir: Path) -> dict:
    path = data_dir / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "labels": DEFAULT_LABELS,
        "feature_dim": LANDMARK_DIM,
        "seq_len_default": 48,
        "synthetic": False,
        "samples": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "sequences")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = load_or_init_manifest(args.out)
    labels = manifest["labels"]
    active = 0
    recording = False
    frames: list[np.ndarray] = []

    tracker = LandmarkTracker(max_hands=1)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")

    print("Labels:", ", ".join(f"{i+1}:{n}" for i, n in enumerate(labels)))
    print("SPACE=record  S=save  Q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        hands = tracker.process(frame)
        hand = primary_hand(hands)
        status = f"{labels[active]} | {'REC' if recording else 'idle'} | buf={len(frames)}"
        if hand:
            for px, py in hand.landmarks_px:
                cv2.circle(frame, (px, py), 3, (0, 200, 120), -1)
            if recording:
                frames.append(landmarks_to_vector(hand.landmarks, handedness=hand.handedness))
        cv2.putText(frame, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 120), 2)
        cv2.imshow("CSLR sequence collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if ord("1") <= key <= ord(str(min(9, len(labels)))):
            active = key - ord("1")
        if key == ord(" "):
            recording = not recording
            if recording:
                frames = []
        if key == ord("s") and frames:
            arr = np.stack(frames, axis=0)
            idx = sum(1 for s in manifest["samples"] if s["label"] == labels[active])
            rel = f"{args.split}/{labels[active]}_{idx:03d}_{int(time.time())}.npz"
            path = args.out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, seq=arr)
            manifest["samples"].append(
                {
                    "path": rel.replace("\\", "/"),
                    "label": labels[active],
                    "label_id": active,
                    "split": args.split,
                    "frames": int(arr.shape[0]),
                }
            )
            manifest["synthetic"] = False
            (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Saved {path} ({arr.shape[0]} frames)")
            frames = []
            recording = False

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
