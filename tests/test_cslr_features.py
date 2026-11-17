import numpy as np
import pytest

from app.cslr.features import SequenceBuffer, landmarks_to_vector, normalize_sequence, resample_sequence


def _fake_hand():
    return [(0.5 + 0.01 * i, 0.4 + 0.005 * i, 0.0) for i in range(21)]


def test_landmarks_to_vector_shape():
    v = landmarks_to_vector(_fake_hand(), handedness="Right")
    assert v.shape == (63,)
    assert np.isfinite(v).all()


def test_sequence_buffer_and_resample():
    buf = SequenceBuffer(window=16)
    for _ in range(10):
        buf.push(landmarks_to_vector(_fake_hand()))
    arr = buf.as_array()
    assert arr.shape == (16, 63)
    short = np.random.randn(20, 63).astype(np.float32)
    assert resample_sequence(short, 48).shape == (48, 63)
    assert normalize_sequence(arr).shape == arr.shape


def test_build_models_lightweight():
    tf = pytest.importorskip("tensorflow")
    from app.cslr.models import build_model, count_params

    for arch in ("cnn1d", "lstm", "transformer"):
        model = build_model(arch, seq_len=48, feat_dim=63, n_classes=8)
        assert count_params(model) > 1000
        out = model.predict(np.zeros((1, 48, 63), dtype=np.float32), verbose=0)
        assert out.shape == (1, 8)
