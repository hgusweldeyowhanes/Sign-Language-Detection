from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Optional, Sequence

import numpy as np

# MediaPipe Hands: 21 landmarks × (x, y, z)
LANDMARK_DIM = 21 * 3


def landmarks_to_vector(
    landmarks: Sequence[tuple[float, float, float]] | None,
    *,
    handedness: str = "Right",
) -> np.ndarray:
    """Flatten 21 hand landmarks to a 63-D vector, wrist-centered & scale-normalized."""
    vec = np.zeros(LANDMARK_DIM, dtype=np.float32)
    if not landmarks or len(landmarks) < 21:
        return vec
    pts = np.asarray(landmarks[:21], dtype=np.float32)
    wrist = pts[0].copy()
    pts = pts - wrist
    # scale by palm size (wrist → middle MCP)
    scale = float(np.linalg.norm(pts[9]) + 1e-6)
    pts = pts / scale
    if handedness.lower().startswith("left"):
        pts[:, 0] *= -1.0  # mirror to right-hand canonical space
    vec[:] = pts.reshape(-1)
    return vec


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Per-sequence mean/std normalize (robust to signing speed offsets)."""
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError("sequence must be (T, F)")
    mean = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-5
    return (seq - mean) / std


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Time-resample to fixed length (handles different signing speeds)."""
    seq = np.asarray(seq, dtype=np.float32)
    t, f = seq.shape
    if t == target_len:
        return seq
    if t == 0:
        return np.zeros((target_len, f), dtype=np.float32)
    old_idx = np.linspace(0.0, 1.0, num=t)
    new_idx = np.linspace(0.0, 1.0, num=target_len)
    out = np.zeros((target_len, f), dtype=np.float32)
    for i in range(f):
        out[:, i] = np.interp(new_idx, old_idx, seq[:, i])
    return out


class SequenceBuffer:
    """Sliding window of landmark feature frames for continuous inference."""

    def __init__(self, window: int = 48, feature_dim: int = LANDMARK_DIM):
        self.window = window
        self.feature_dim = feature_dim
        self._buf: Deque[np.ndarray] = deque(maxlen=window)

    def clear(self) -> None:
        self._buf.clear()

    def push(self, frame_vec: np.ndarray) -> None:
        v = np.asarray(frame_vec, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.feature_dim:
            raise ValueError(f"expected {self.feature_dim}-D frame, got {v.shape[0]}")
        self._buf.append(v)

    def ready(self) -> bool:
        return len(self._buf) >= max(8, self.window // 3)

    def as_array(self, pad: bool = True) -> np.ndarray:
        if not self._buf:
            return np.zeros((self.window, self.feature_dim), dtype=np.float32)
        arr = np.stack(list(self._buf), axis=0)
        if pad and arr.shape[0] < self.window:
            pad_n = self.window - arr.shape[0]
            arr = np.concatenate(
                [np.repeat(arr[:1], pad_n, axis=0), arr],
                axis=0,
            )
        elif arr.shape[0] > self.window:
            arr = arr[-self.window :]
        return normalize_sequence(arr)

    def activity(self) -> float:
        """Mean frame-to-frame motion — used to gate idle vs signing."""
        if len(self._buf) < 2:
            return 0.0
        arr = np.stack(list(self._buf), axis=0)
        deltas = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        return float(deltas.mean())
