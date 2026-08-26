"""Synthetic MediaPipe-style landmarks for alphabet tests."""
from __future__ import annotations

WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)


def _pt(x, y):
    return [float(x), float(y), 0.0]


def make_hand(*, up=(), pinch=False, spread=0.06, thumb_x=None, index_axis="up"):
    """
    Upright right hand in image coordinates (y grows downward).
    `up` is a subset of: thumb, index, middle, ring, pinky.
    """
    lm = [_pt(0.50, 0.90) for _ in range(21)]
    lm[0] = _pt(0.50, 0.90)

    mcp_x = {"thumb": 0.40, "index": 0.44, "middle": 0.50, "ring": 0.56, "pinky": 0.62}
    mcp_y = 0.64
    for name, chain in (
        ("thumb", THUMB),
        ("index", INDEX),
        ("middle", MIDDLE),
        ("ring", RING),
        ("pinky", PINKY),
    ):
        mcp, pip, dip, tip = chain
        x = mcp_x[name]
        lm[mcp] = _pt(x, mcp_y)
        if name in up and name != "thumb":
            if name == "index" and index_axis == "side":
                lm[pip] = _pt(x + 0.10, mcp_y)
                lm[dip] = _pt(x + 0.18, mcp_y)
                lm[tip] = _pt(x + 0.28, mcp_y)
            elif name == "index" and index_axis == "down":
                lm[pip] = _pt(x, mcp_y + 0.10)
                lm[dip] = _pt(x, mcp_y + 0.18)
                lm[tip] = _pt(x, mcp_y + 0.28)
            else:
                offset = 0.0
                if name == "index":
                    offset = -spread / 2
                elif name == "middle":
                    offset = spread / 2
                # Keep U tips near the same x; V fans them apart.
                base_x = 0.47 if name in ("index", "middle") and spread < 0.08 else x
                lm[pip] = _pt(base_x + offset * 0.4, 0.48)
                lm[dip] = _pt(base_x + offset * 0.7, 0.36)
                lm[tip] = _pt(base_x + offset, 0.24)
        elif name != "thumb":
            lm[pip] = _pt(x, 0.70)
            lm[dip] = _pt(x, 0.72)
            lm[tip] = _pt(x + 0.01, 0.74)

    # Thumb
    if "thumb" in up:
        tx = 0.26 if thumb_x is None else thumb_x
        lm[1] = _pt(0.42, 0.78)
        lm[2] = _pt(0.36, 0.72)
        lm[3] = _pt(0.30, 0.70)
        lm[4] = _pt(tx, 0.68)
    else:
        lm[1] = _pt(0.44, 0.78)
        lm[2] = _pt(0.44, 0.70)
        lm[3] = _pt(0.45, 0.66)
        lm[4] = _pt(0.46, 0.64)

    if pinch:
        # Pull index tip to the thumb
        lm[6] = _pt(0.40, 0.62)
        lm[7] = _pt(0.36, 0.64)
        lm[8] = _pt(lm[4][0] + 0.02, lm[4][1] + 0.01)

    return lm
