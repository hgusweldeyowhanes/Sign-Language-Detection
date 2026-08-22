"""Text-to-speech helper for Conversation Mode."""
from __future__ import annotations

import threading


class Speaker:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._engine = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._engine is not None:
            return
        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 175)
        except Exception:
            self._engine = False  # mark unavailable

    def speak(self, text: str) -> bool:
        if not self.enabled or not text:
            return False
        self._ensure()
        if not self._engine:
            return False

        def _run():
            with self._lock:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception:
                    pass

        threading.Thread(target=_run, daemon=True).start()
        return True
