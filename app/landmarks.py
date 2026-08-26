"""MediaPipe hand landmarks for fingerspelling (up to two hands)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2

from .asl_alphabet import LetterGuess, classify_letter


@dataclass
class TrackedHand:
    landmarks: list[tuple[float, float, float]]
    landmarks_px: list[tuple[int, int]]
    handedness: str
    score: float
    bbox: tuple[int, int, int, int]


class LandmarkTracker:
    def __init__(self, max_hands: int = 2):
        self.max_hands = max_hands
        self._hands = None

    def _ensure(self):
        if self._hands is not None:
            return
        import mediapipe as mp

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.55,
        )

    def close(self):
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def process(self, frame_bgr) -> list[TrackedHand]:
        self._ensure()
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        hands: list[TrackedHand] = []
        if not result.multi_hand_landmarks:
            return hands

        handedness_list = result.multi_handedness or []
        for idx, hand_lms in enumerate(result.multi_hand_landmarks):
            pts_norm = []
            pts_px = []
            xs, ys = [], []
            for lm in hand_lms.landmark:
                pts_norm.append((lm.x, lm.y, lm.z))
                px, py = int(lm.x * w), int(lm.y * h)
                pts_px.append((px, py))
                xs.append(px)
                ys.append(py)
            label = "Right"
            score = 0.0
            if idx < len(handedness_list):
                cat = handedness_list[idx].classification[0]
                label = cat.label
                score = float(cat.score)
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            pad = 18
            bbox = (
                max(0, x1 - pad),
                max(0, y1 - pad),
                min(w, x2 + pad),
                min(h, y2 + pad),
            )
            hands.append(
                TrackedHand(
                    landmarks=pts_norm,
                    landmarks_px=pts_px,
                    handedness=label,
                    score=score,
                    bbox=bbox,
                )
            )
        hands.sort(key=lambda hand: hand.score, reverse=True)
        return hands


def primary_hand(hands: list[TrackedHand]) -> TrackedHand | None:
    return hands[0] if hands else None


def guess_from_hand(hand: TrackedHand | None) -> LetterGuess | None:
    if hand is None:
        return None
    return classify_letter(hand.landmarks, handedness=hand.handedness)


def draw_fingerspell_overlay(frame_bgr, hands: list[TrackedHand], guess: LetterGuess | None, word: str):
    annotated = frame_bgr.copy()
    for hand in hands:
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 160, 80), 2)
        for px, py in hand.landmarks_px:
            cv2.circle(annotated, (px, py), 3, (80, 200, 120), -1)
        tag = f"{hand.handedness} {hand.score:.0%}"
        cv2.putText(
            annotated,
            tag,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 160, 80),
            2,
        )

    letter = guess.letter if guess else "…"
    conf = f"{guess.confidence:.0%}" if guess else ""
    cv2.putText(
        annotated,
        f"{letter}  {conf}",
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (40, 200, 90) if guess else (180, 180, 180),
        3,
    )
    cv2.putText(
        annotated,
        f"word: {word or '—'}",
        (24, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (240, 240, 240),
        2,
    )
    if len(hands) > 1:
        cv2.putText(
            annotated,
            "2 hands — using the stronger detection",
            (24, 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 80),
            1,
        )
    return annotated
