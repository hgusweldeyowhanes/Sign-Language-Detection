from app.asl_alphabet import classify_letter
from tests.landmarks_factory import make_hand


def _letter(lm, handedness="Right"):
    guess = classify_letter(lm, handedness=handedness)
    assert guess is not None, "expected a letter guess"
    return guess.letter


def test_l_shape():
    lm = make_hand(up=("thumb", "index"))
    assert _letter(lm) == "L"


def test_y_shape():
    lm = make_hand(up=("thumb", "pinky"))
    assert _letter(lm) == "Y"


def test_i_shape():
    lm = make_hand(up=("pinky",))
    assert _letter(lm) == "I"


def test_b_shape():
    lm = make_hand(up=("index", "middle", "ring", "pinky"))
    assert _letter(lm) == "B"


def test_w_shape():
    lm = make_hand(up=("index", "middle", "ring"))
    assert _letter(lm) == "W"


def test_v_spread():
    lm = make_hand(up=("index", "middle"), spread=0.14)
    assert _letter(lm) == "V"


def test_u_together():
    lm = make_hand(up=("index", "middle"), spread=0.03)
    assert _letter(lm) == "U"


def test_a_thumb_out_fist():
    lm = make_hand(up=("thumb",))
    assert _letter(lm) == "A"


def test_d_index_up():
    lm = make_hand(up=("index",), index_axis="up")
    assert _letter(lm) == "D"


def test_left_hand_is_mirrored_to_l():
    lm = make_hand(up=("thumb", "index"))
    # Mirror x so it looks like a left hand in camera space
    mirrored = [(1.0 - x, y, z) for x, y, z in lm]
    assert _letter(mirrored, handedness="Left") == "L"


def test_low_structure_returns_none():
    lm = [[0.5, 0.5, 0.0] for _ in range(21)]
    assert classify_letter(lm) is None
