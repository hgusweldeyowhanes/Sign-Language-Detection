"""Webcam hand crop + CNN prediction."""
from __future__ import annotations

import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector

from .config import CONFIDENCE_THRESHOLD, IMG_SIZE, LABELS, OFFSET


class SignDetector:
    def __init__(self, model, labels=None, threshold: float = CONFIDENCE_THRESHOLD):
        self.model = model
        self.labels = list(labels or LABELS)
        self.threshold = threshold
        self.hand_detector = HandDetector(maxHands=1)

    def predict_frame(self, frame_bgr):
        """
        Returns dict: hands_found, label, confidence, probs, annotated_frame
        """
        annotated = frame_bgr.copy()
        hands, annotated = self.hand_detector.findHands(annotated)
        result = {
            "hands_found": bool(hands),
            "label": None,
            "confidence": 0.0,
            "probs": {lab: 0.0 for lab in self.labels},
            "annotated_frame": annotated,
        }
        if not hands:
            return result

        hand = hands[0]
        x, y, w, h = hand["bbox"]
        y1, y2 = max(0, y - OFFSET), min(frame_bgr.shape[0], y + h + OFFSET)
        x1, x2 = max(0, x - OFFSET), min(frame_bgr.shape[1], x + w + OFFSET)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return result

        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        batch = np.expand_dims(resized, axis=0)
        prediction = self.model.predict(batch, verbose=0)[0]
        index = int(np.argmax(prediction))
        confidence = float(prediction[index])
        for i, lab in enumerate(self.labels):
            if i < len(prediction):
                result["probs"][lab] = float(prediction[i])

        result["confidence"] = confidence
        if confidence >= self.threshold:
            result["label"] = self.labels[index]

        # Overlay
        color = (40, 160, 80) if result["label"] else (40, 40, 200)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{result['label'] or '...'} {confidence:.0%}"
        cv2.putText(
            annotated,
            text,
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        result["annotated_frame"] = annotated
        return result
