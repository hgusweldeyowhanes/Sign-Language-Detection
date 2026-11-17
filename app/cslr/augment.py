from __future__ import annotations

import numpy as np


def augment_sequence(
    seq: np.ndarray,
    *,
    speed_jitter: float = 0.25,
    noise_std: float = 0.02,
    dropout_prob: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate signing-speed variation, sensor noise, and brief occlusions."""
    rng = rng or np.random.default_rng()
    seq = np.asarray(seq, dtype=np.float32)
    t, f = seq.shape

    # speed: resample to slightly different length then back
    factor = 1.0 + float(rng.uniform(-speed_jitter, speed_jitter))
    new_t = max(8, int(round(t * factor)))
    old_idx = np.linspace(0.0, 1.0, num=t)
    mid_idx = np.linspace(0.0, 1.0, num=new_t)
    tmp = np.zeros((new_t, f), dtype=np.float32)
    for i in range(f):
        tmp[:, i] = np.interp(mid_idx, old_idx, seq[:, i])
    back_idx = np.linspace(0.0, 1.0, num=t)
    out = np.zeros_like(seq)
    for i in range(f):
        out[:, i] = np.interp(back_idx, mid_idx, tmp[:, i])

    out = out + rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)
    if dropout_prob > 0:
        mask = rng.random(t) > dropout_prob
        if mask.any():
            # hold previous frame on dropout
            for i in range(1, t):
                if not mask[i]:
                    out[i] = out[i - 1]
    return out.astype(np.float32)
