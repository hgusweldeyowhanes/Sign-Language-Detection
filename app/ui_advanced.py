"""Advanced Conversation Studio UI (Tkinter)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .config import LABELS, ROOT
from .detector import SignDetector
from .fingerspell import FingerspellEngine, SpellEvent
from .landmarks import LandmarkTracker, draw_fingerspell_overlay, guess_from_hand, primary_hand
from .model_loader import load_classifier
from .phrase_engine import PhraseEngine
from .practice import PracticeSession, SpellPractice
from .tts import Speaker

MODE_PHRASE = "phrases"
MODE_SPELL = "fingerspell"


class ConversationStudio:
    def __init__(self, model_path=None, mode: str = MODE_PHRASE):
        self.root = tk.Tk()
        self.root.title("Sign Studio — Conversation Mode")
        self.root.geometry("1240x760")
        self.root.minsize(1040, 680)

        try:
            self.model = load_classifier(model_path)
            self.model_error = None
        except FileNotFoundError as e:
            self.model = None
            self.model_error = str(e)

        self.detector = SignDetector(self.model) if self.model else None
        self.tracker = LandmarkTracker(max_hands=2)
        self.engine = PhraseEngine()
        self.speller = FingerspellEngine()
        self.practice = PracticeSession()
        self.spell_practice = SpellPractice()
        self.speaker = Speaker(enabled=True)
        self.cap = None
        self._photo = None
        self.mode = tk.StringVar(value=mode if mode in (MODE_PHRASE, MODE_SPELL) else MODE_PHRASE)
        self.practice_on = tk.BooleanVar(value=False)
        self.tts_on = tk.BooleanVar(value=True)
        self.tts_letters_on = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Ready")
        self.sentence_var = tk.StringVar(value="")
        self.word_var = tk.StringVar(value="")
        self.live_label = tk.StringVar(value="—")
        self.live_conf = tk.StringVar(value="")
        self.alts_var = tk.StringVar(value="")
        self.practice_msg = tk.StringVar(value=self.practice.message)
        self.prob_vars = {lab: tk.DoubleVar(value=0.0) for lab in LABELS}

        self._build_ui()
        self._sync_mode()
        if self.model_error and self.mode.get() == MODE_PHRASE:
            self.status.set(self.model_error)
            messagebox.showwarning(
                "Model missing",
                f"{self.model_error}\n\nFingerspell A–Z still works without the CNN. "
                "Switch to Fingerspell mode, or place best_model.keras in the project root.",
            )
        self._start_camera()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="both", expand=True)

        left = ttk.Frame(top)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ttk.Label(left, text="Camera", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.camera_label = ttk.Label(left)
        self.camera_label.pack(fill="both", expand=True, pady=6)

        live_row = ttk.Frame(left)
        live_row.pack(fill="x")
        ttk.Label(live_row, textvariable=self.live_label, font=("Segoe UI", 16, "bold")).pack(
            side="left"
        )
        ttk.Label(live_row, textvariable=self.live_conf, foreground="#555").pack(
            side="left", padx=10
        )
        ttk.Label(live_row, textvariable=self.alts_var, foreground="#666").pack(
            side="left", padx=6
        )

        self.meters_frame = ttk.Frame(left)
        self.meters_frame.pack(fill="x")
        ttk.Label(self.meters_frame, text="Class confidence", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=(8, 2)
        )
        meters = ttk.Frame(self.meters_frame)
        meters.pack(fill="x")
        for lab in LABELS:
            row = ttk.Frame(meters)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=lab, width=14).pack(side="left")
            bar = ttk.Progressbar(row, maximum=1.0, variable=self.prob_vars[lab], length=280)
            bar.pack(side="left", fill="x", expand=True)

        right = ttk.Frame(top, width=440)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        ttk.Label(right, text="Conversation Studio", font=("Segoe UI", 14, "bold")).pack(
            anchor="w"
        )

        mode_row = ttk.Frame(right)
        mode_row.pack(fill="x", pady=(4, 8))
        ttk.Radiobutton(
            mode_row,
            text="Phrase signs",
            value=MODE_PHRASE,
            variable=self.mode,
            command=self._sync_mode,
        ).pack(side="left")
        ttk.Radiobutton(
            mode_row,
            text="Fingerspell A–Z",
            value=MODE_SPELL,
            variable=self.mode,
            command=self._sync_mode,
        ).pack(side="left", padx=8)

        self.hint_var = tk.StringVar()
        ttk.Label(right, textvariable=self.hint_var, wraplength=420).pack(anchor="w", pady=(0, 8))

        self.word_block = ttk.Frame(right)
        ttk.Label(self.word_block, text="Current word", font=("Segoe UI", 10, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            self.word_block,
            textvariable=self.word_var,
            wraplength=400,
            font=("Segoe UI", 16, "bold"),
            background="#e8f5e9",
            padding=8,
        ).pack(fill="x", pady=4)
        word_btns = ttk.Frame(self.word_block)
        word_btns.pack(fill="x", pady=(0, 6))
        ttk.Button(word_btns, text="Commit word", command=self._commit_word).pack(
            side="left", padx=2
        )
        ttk.Button(word_btns, text="Backspace letter", command=self._backspace).pack(
            side="left", padx=2
        )

        self.sentence_heading = ttk.Label(right, text="Sentence", font=("Segoe UI", 10, "bold"))
        self.sentence_heading.pack(anchor="w")
        self.sentence_label = ttk.Label(
            right,
            textvariable=self.sentence_var,
            wraplength=400,
            font=("Segoe UI", 13),
            background="#f7f1e6",
            padding=8,
        )
        self.sentence_label.pack(fill="x", pady=4)

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Undo", command=self._undo).pack(side="left", padx=2)
        ttk.Button(btns, text="Clear", command=self._clear).pack(side="left", padx=2)
        ttk.Button(btns, text="Speak sentence", command=self._speak_sentence).pack(
            side="left", padx=2
        )

        ttk.Checkbutton(
            right, text="Speak new phrases / words (TTS)", variable=self.tts_on
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            right, text="Speak each fingerspelled letter", variable=self.tts_letters_on
        ).pack(anchor="w")

        ttk.Separator(right).pack(fill="x", pady=10)
        ttk.Label(right, text="Practice Mode", font=("Segoe UI", 12, "bold")).pack(
            anchor="w"
        )
        ttk.Checkbutton(
            right, text="Enable guided practice", variable=self.practice_on
        ).pack(anchor="w")
        ttk.Button(right, text="Next target", command=self._next_practice).pack(
            anchor="w", pady=4
        )
        ttk.Label(right, textvariable=self.practice_msg, wraplength=400).pack(
            anchor="w", pady=4
        )

        ttk.Separator(right).pack(fill="x", pady=10)
        ttk.Label(right, text="Transcript", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.transcript = tk.Text(right, height=12, wrap="word", font=("Consolas", 10))
        self.transcript.pack(fill="both", expand=True, pady=4)

        export_row = ttk.Frame(right)
        export_row.pack(fill="x", pady=4)
        ttk.Button(export_row, text="Export TXT", command=lambda: self._export("txt")).pack(
            side="left", padx=2
        )
        ttk.Button(export_row, text="Export JSON", command=lambda: self._export("json")).pack(
            side="left", padx=2
        )

        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.status).pack(anchor="w")

    def _is_spell(self) -> bool:
        return self.mode.get() == MODE_SPELL

    def _active_engine(self):
        return self.speller if self._is_spell() else self.engine

    def _sync_mode(self):
        if self._is_spell():
            self.hint_var.set(
                "Hold a letter still to add it. Drop your hand briefly before a double letter "
                "(LL in HELLO). Pause ~1s or press Commit word to finish the word. "
                "J looks like I; Z looks like D (they are motion signs)."
            )
            self.word_block.pack(fill="x", pady=(0, 6), before=self.sentence_heading)
            self.meters_frame.pack_forget()
            self.practice_msg.set(self.spell_practice.message)
            self.word_var.set(self.speller.current_word or "—")
            self.sentence_var.set(self.speller.sentence())
            self.status.set("Fingerspell mode — A–Z from hand landmarks (no CNN needed).")
            return

        self.hint_var.set(
            "Stable phrase signs become sentence tokens. Speak, practice, and export."
        )
        self.word_block.pack_forget()
        self.meters_frame.pack(fill="x")
        self.alts_var.set("")
        self.practice_msg.set(self.practice.message)
        self.sentence_var.set(self.engine.sentence())
        if self.model_error:
            self.status.set(self.model_error)
        else:
            self.status.set("Phrase mode — HELLO, THANK YOU, I LOVE YOU, GOODBYE.")

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status.set("Could not open webcam.")
            messagebox.showerror("Camera", "Could not open webcam (index 0).")
            return
        self._tick()

    def _tick(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(30, self._tick)
            return

        frame = cv2.flip(frame, 1)
        if self._is_spell():
            annotated = self._tick_fingerspell(frame)
        else:
            annotated = self._tick_phrases(frame)

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = image.resize((720, 540))
        self._photo = ImageTk.PhotoImage(image)
        self.camera_label.configure(image=self._photo)
        self.root.after(20, self._tick)

    def _tick_phrases(self, frame):
        if self.detector is None:
            cv2.putText(
                frame,
                "Place best_model.keras or switch to Fingerspell",
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (40, 40, 200),
                2,
            )
            return frame

        result = self.detector.predict_frame(frame)
        for lab, val in result["probs"].items():
            if lab in self.prob_vars:
                self.prob_vars[lab].set(val)

        if result["label"]:
            self.live_label.set(result["label"])
            self.live_conf.set(f"{result['confidence']:.0%}")
        else:
            self.live_label.set("…")
            self.live_conf.set("")

        committed = self.engine.update(result["label"], result["confidence"])
        if committed:
            self._on_phrase_committed(committed, result["confidence"])
        return result["annotated_frame"]

    def _tick_fingerspell(self, frame):
        hands = self.tracker.process(frame)
        hand = primary_hand(hands)
        guess = guess_from_hand(hand)
        letter = guess.letter if guess else None
        confidence = guess.confidence if guess else 0.0

        if guess:
            self.live_label.set(guess.letter)
            self.live_conf.set(f"{guess.confidence:.0%}")
            extras = "  ".join(f"{lab} {conf:.0%}" for lab, conf in guess.alternatives[1:3])
            self.alts_var.set(extras)
        else:
            self.live_label.set("…")
            self.live_conf.set("")
            self.alts_var.set("" if hands else "Show one hand to the camera")

        event = self.speller.update(letter, confidence)
        self.word_var.set(self.speller.current_word or "—")
        self.sentence_var.set(self.speller.sentence())
        if event:
            self._on_spell_event(event)

        return draw_fingerspell_overlay(frame, hands, guess, self.speller.current_word)

    def _on_phrase_committed(self, phrase: str, _confidence: float):
        self.sentence_var.set(self.engine.sentence())
        self._append_transcript(phrase)
        self.status.set(self.engine.last_event)
        if self.tts_on.get():
            self.speaker.speak(phrase)
        if self.practice_on.get():
            done = self.practice.on_phrase(phrase)
            self.practice_msg.set(self.practice.message)
            if done and self.tts_on.get():
                self.speaker.speak("Perfect")

    def _on_spell_event(self, event):
        self.word_var.set(self.speller.current_word or "—")
        self.sentence_var.set(self.speller.sentence())
        self.status.set(self.speller.last_event)
        if event.kind == "letter":
            self._append_transcript(f"  {event.value}")
            if self.tts_letters_on.get():
                self.speaker.speak(event.value)
            if self.practice_on.get():
                done = self.spell_practice.on_letter(event.value)
                self.practice_msg.set(self.spell_practice.message)
                if done and self.tts_on.get():
                    self.speaker.speak("Perfect")
        elif event.kind == "word":
            self._append_transcript(event.value)
            if self.tts_on.get():
                self.speaker.speak(event.value)

    def _append_transcript(self, text: str):
        self.transcript.insert("end", f"[{datetime.now():%H:%M:%S}] {text}\n")
        self.transcript.see("end")

    def _commit_word(self):
        word = self.speller.commit_word()
        if not word:
            self.status.set("No letters in the current word.")
            return
        self._on_spell_event(SpellEvent("word", word))

    def _backspace(self):
        self.speller.backspace()
        self.word_var.set(self.speller.current_word or "—")
        self.sentence_var.set(self.speller.sentence())
        self.status.set(self.speller.last_event)

    def _undo(self):
        self._active_engine().undo()
        self.word_var.set(self.speller.current_word or "—")
        self.sentence_var.set(self._active_engine().sentence())
        self.status.set(self._active_engine().last_event)

    def _clear(self):
        self._active_engine().clear()
        self.sentence_var.set("")
        self.word_var.set("—")
        self.transcript.delete("1.0", "end")
        self.status.set(self._active_engine().last_event)

    def _speak_sentence(self):
        text = self._active_engine().end_sentence()
        if not text:
            self.status.set("Nothing to speak yet.")
            return
        self.word_var.set(self.speller.current_word or "—")
        self.sentence_var.set(self._active_engine().sentence())
        self.speaker.speak(text)
        self.status.set(f"Speaking: {text}")

    def _next_practice(self):
        self.practice_on.set(True)
        if self._is_spell():
            self.spell_practice.next_target()
            self.practice_msg.set(self.spell_practice.message)
            self.status.set(self.spell_practice.message)
        else:
            self.practice.next_target()
            self.practice_msg.set(self.practice.message)
            self.status.set(self.practice.message)

    def _export(self, kind: str):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = ROOT / f"transcript_{stamp}.{kind}"
        path = filedialog.asksaveasfilename(
            defaultextension=f".{kind}",
            initialfile=default.name,
            filetypes=[(kind.upper(), f"*.{kind}"), ("All", "*.*")],
        )
        if not path:
            return
        out = Path(path)
        engine = self._active_engine()
        if kind == "txt":
            engine.export_txt(out)
        else:
            engine.export_json(out)
        self.status.set(engine.last_event)

    def _on_close(self):
        if self.cap is not None:
            self.cap.release()
        self.tracker.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(model_path=None, mode: str = MODE_PHRASE):
    ConversationStudio(model_path=model_path, mode=mode).run()
