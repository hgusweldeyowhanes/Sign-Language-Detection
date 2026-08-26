"""Geometric ASL alphabet classifier from 21 hand landmarks.

Letters J and Z are motion signs; the static pose is scored as I and D.
Left hands are mirrored into right-hand space before scoring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import ALPHABET, LETTER_MIN_CONFIDENCE

# MediaPipe Hands
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


@dataclass(frozen=True)
class LetterGuess:
    letter: str
    confidence: float
    alternatives: tuple[tuple[str, float], ...]


def _dist(lm, i: int, j: int) -> float:
    ax, ay = lm[i][0], lm[i][1]
    bx, by = lm[j][0], lm[j][1]
    return math.hypot(ax - bx, ay - by)


def _as_right_hand(landmarks, handedness: str | None):
    pts = [(float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0) for p in landmarks]
    if handedness and str(handedness).lower().startswith("left"):
        pts = [(1.0 - x, y, z) for x, y, z in pts]
    return pts


def _extended(lm, tip: int, pip: int, palm: float) -> bool:
    return _dist(lm, tip, WRIST) > _dist(lm, pip, WRIST) + palm * 0.07


def _thumb_out(lm, palm: float) -> bool:
    return _dist(lm, THUMB_TIP, INDEX_MCP) > palm * 0.55 and _dist(
        lm, THUMB_TIP, WRIST
    ) > _dist(lm, THUMB_IP, WRIST)


def _finger_flags(lm, palm: float) -> tuple[bool, bool, bool, bool, bool]:
    thumb = _thumb_out(lm, palm)
    index = _extended(lm, INDEX_TIP, INDEX_PIP, palm)
    middle = _extended(lm, MIDDLE_TIP, MIDDLE_PIP, palm)
    ring = _extended(lm, RING_TIP, RING_PIP, palm)
    pinky = _extended(lm, PINKY_TIP, PINKY_PIP, palm)
    return thumb, index, middle, ring, pinky


def _pointing_axis(lm) -> str:
    """Where the index tip sits relative to the wrist: up, down, or side."""
    dx = lm[INDEX_TIP][0] - lm[WRIST][0]
    dy = lm[INDEX_TIP][1] - lm[WRIST][1]
    if abs(dy) >= abs(dx) * 1.05:
        return "down" if dy > 0 else "up"
    return "side"


def classify_letter(
    landmarks,
    handedness: str | None = "Right",
    min_confidence: float = LETTER_MIN_CONFIDENCE,
) -> LetterGuess | None:
    if landmarks is None or len(landmarks) < 21:
        return None

    lm = _as_right_hand(landmarks, handedness)
    palm = _dist(lm, WRIST, MIDDLE_MCP)
    if palm < 1e-6:
        return None

    t, i, m, r, p = _finger_flags(lm, palm)
    pinch = _dist(lm, THUMB_TIP, INDEX_TIP) / palm
    spread_im = _dist(lm, INDEX_TIP, MIDDLE_TIP) / palm
    thumb_index_mcp = _dist(lm, THUMB_TIP, INDEX_MCP) / palm
    thumb_middle_mcp = _dist(lm, THUMB_TIP, MIDDLE_MCP) / palm
    thumb_ring_mcp = _dist(lm, THUMB_TIP, RING_MCP) / palm
    index_curl = _dist(lm, INDEX_TIP, INDEX_MCP) / palm
    axis = _pointing_axis(lm)
    index_middle_cross = lm[INDEX_TIP][0] > lm[MIDDLE_TIP][0] + palm * 0.015

    scores = {letter: 0.0 for letter in ALPHABET}

    # Open / distinctive shapes
    if not t and i and m and r and p:
        scores["B"] += 3.4
    if t and i and not m and not r and not p:
        if axis == "side":
            scores["G"] += 3.0
            scores["L"] += 1.1
        elif axis == "down":
            scores["Q"] += 3.1
        else:
            scores["L"] += 3.4
            scores["G"] += 0.8
    if t and p and not i and not m and not r:
        scores["Y"] += 3.5
    if p and not i and not m and not r and not t:
        scores["I"] += 3.4
        scores["J"] += 1.4
    if i and m and r and not p:
        scores["W"] += 3.4
        if t:
            scores["W"] += 0.2

    # F: OK-shape, three fingers up
    if pinch < 0.38 and m and r and p and not i:
        scores["F"] += 3.6
    elif pinch < 0.42 and m and r and p:
        scores["F"] += 2.4

    # Two-finger group: U V K R H P
    if i and m and not r and not p:
        if t and spread_im > 0.32:
            if axis == "down":
                scores["P"] += 3.2
            else:
                scores["K"] += 3.1
        if axis == "side":
            scores["H"] += 2.8
        if spread_im > 0.38:
            scores["V"] += 3.3
        elif spread_im < 0.22 and index_middle_cross:
            scores["R"] += 3.2
        elif spread_im < 0.32:
            scores["U"] += 3.1
        else:
            scores["U"] += 1.4
            scores["V"] += 1.6

    # Single index: D X Z G Q
    if i and not m and not r and not p:
        if index_curl < 0.55:
            scores["X"] += 3.0
        elif axis == "up":
            scores["D"] += 3.2
            scores["Z"] += 1.2
        elif axis == "down":
            scores["Q"] += 2.6
            scores["Z"] += 1.0
        else:
            scores["G"] += 2.4
            if t:
                scores["G"] += 0.6

    # C vs O vs open five
    if t and i and m and r and p:
        if 0.45 < pinch < 1.05:
            scores["C"] += 3.0
        elif pinch >= 1.05:
            scores["C"] += 1.2
            scores["B"] += 0.6
        else:
            scores["O"] += 1.5

    if pinch < 0.40 and not m and not r and not p:
        scores["O"] += 3.1
        if not i:
            scores["O"] += 0.4

    # Fist family: A S E T M N
    if not i and not m and not r and not p:
        if t and thumb_index_mcp > 0.50:
            scores["A"] += 3.3
        elif thumb_ring_mcp < 0.38 and thumb_middle_mcp < 0.48:
            scores["M"] += 3.1
        elif thumb_middle_mcp < 0.40:
            scores["N"] += 3.1
        elif thumb_index_mcp < 0.42:
            scores["T"] += 3.0
        elif pinch < 0.45:
            scores["O"] += 2.2
            scores["E"] += 1.6
        else:
            scores["S"] += 2.4
            scores["E"] += 2.2
            if thumb_index_mcp < 0.55:
                scores["E"] += 0.6
            else:
                scores["S"] += 0.5

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_letter, best_score = ranked[0]
    if best_score < 1.15:
        return None

    confidence = min(0.99, best_score / 3.6)
    if confidence < min_confidence:
        return None

    alternatives = tuple(
        (lab, min(0.99, sc / 3.6)) for lab, sc in ranked[:3] if sc > 0.8
    )
    return LetterGuess(letter=best_letter, confidence=confidence, alternatives=alternatives)
