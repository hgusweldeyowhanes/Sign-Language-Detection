from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .features import SequenceBuffer, landmarks_to_vector


@dataclass
class GlossPrediction:
    label: str
    confidence: float
    label_id: int
    activity: float


class ContinuousRecognizer:
    """Streaming continuous recognizer: MediaPipe landmarks → sliding window → model."""

    def __init__(
        self,
        model,
        labels: list[str],
        *,
        seq_len: int = 48,
        conf_threshold: float = 0.55,
        activity_threshold: float = 0.015,
        stable_windows: int = 3,
        cooldown_windows: int = 8,
    ):
        self.model = model
        self.labels = labels
        self.seq_len = seq_len
        self.conf_threshold = conf_threshold
        self.activity_threshold = activity_threshold
        self.stable_windows = stable_windows
        self.cooldown_windows = cooldown_windows
        self.buffer = SequenceBuffer(window=seq_len)
        self._pending: Optional[str] = None
        self._pending_count = 0
        self._cooldown = 0
        self.last: Optional[GlossPrediction] = None
        self.committed: list[str] = []

    @classmethod
    def from_saved(
        cls,
        model_path: Path | str,
        labels: list[str],
        architecture: str = "lstm",
        seq_len: int = 48,
        feat_dim: int = 63,
        **kwargs,
    ) -> "ContinuousRecognizer":
        import tensorflow as tf

        from .models import build_model

        path = Path(model_path)
        if path.exists():
            model = tf.keras.models.load_model(path)
        else:
            # untrained scaffold for UI wiring / dry-run
            model = build_model(architecture, seq_len, feat_dim, len(labels))
        return cls(model, labels, seq_len=seq_len, **kwargs)

    def reset(self) -> None:
        self.buffer.clear()
        self._pending = None
        self._pending_count = 0
        self._cooldown = 0
        self.last = None
        self.committed.clear()

    def push_landmarks(
        self,
        landmarks,
        *,
        handedness: str = "Right",
        run_infer: bool = True,
    ) -> Optional[GlossPrediction]:
        vec = landmarks_to_vector(landmarks, handedness=handedness)
        self.buffer.push(vec)
        if not run_infer or not self.buffer.ready():
            return None
        if self._cooldown > 0:
            self._cooldown -= 1
            return self.last

        activity = self.buffer.activity()
        if activity < self.activity_threshold:
            self._pending = None
            self._pending_count = 0
            return None

        x = self.buffer.as_array()[None, ...]
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.labels[idx]
        pred = GlossPrediction(label=label, confidence=conf, label_id=idx, activity=activity)
        self.last = pred

        if conf < self.conf_threshold:
            self._pending = None
            self._pending_count = 0
            return pred

        if label == self._pending:
            self._pending_count += 1
        else:
            self._pending = label
            self._pending_count = 1

        if self._pending_count >= self.stable_windows:
            self.committed.append(label)
            self._cooldown = self.cooldown_windows
            self._pending = None
            self._pending_count = 0
            # soft-clear buffer to encourage next gloss boundary
            self.buffer.clear()
        return pred

    def sentence(self) -> str:
        return " ".join(self.committed)
