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
from .model_loader import load_classifier
from .phrase_engine import PhraseEngine
from .practice import PracticeSession
from .tts import Speaker


class ConversationStudio:
    def __init__(self, model_path=None):
        self.root = tk.Tk()
        self.root.title("Sign Studio — Conversation Mode")
        self.root.geometry("1180x720")
        self.root.minsize(1000, 640)

        try:
            self.model = load_classifier(model_path)
            self.model_error = None
        except FileNotFoundError as e:
            self.model = None
            self.model_error = str(e)

        self.detector = SignDetector(self.model) if self.model else None
        self.engine = PhraseEngine()
        self.practice = PracticeSession()
        self.speaker = Speaker(enabled=True)
        self.cap = None
        self._photo = None
        self.practice_on = tk.BooleanVar(value=False)
        self.tts_on = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Ready")
        self.sentence_var = tk.StringVar(value="")
        self.live_label = tk.StringVar(value="—")
        self.live_conf = tk.StringVar(value="")
        self.practice_msg = tk.StringVar(value=self.practice.message)
        self.prob_vars = {lab: tk.DoubleVar(value=0.0) for lab in LABELS}

        self._build_ui()
        if self.model_error:
            self.status.set(self.model_error)
            messagebox.showwarning("Model missing", self.model_error)
        else:
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

        ttk.Label(left, text="Class confidence", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=(8, 2)
        )
        meters = ttk.Frame(left)
        meters.pack(fill="x")
        for lab in LABELS:
            row = ttk.Frame(meters)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=lab, width=14).pack(side="left")
            bar = ttk.Progressbar(row, maximum=1.0, variable=self.prob_vars[lab], length=280)
            bar.pack(side="left", fill="x", expand=True)

        right = ttk.Frame(top, width=420)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        ttk.Label(right, text="Conversation Mode", font=("Segoe UI", 14, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            right,
            text="Stable signs become sentence tokens. Speak, practice, and export.",
            wraplength=400,
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(right, text="Sentence", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(
            right,
            textvariable=self.sentence_var,
            wraplength=400,
            font=("Segoe UI", 13),
            background="#f7f1e6",
            padding=8,
        ).pack(fill="x", pady=4)

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Undo", command=self._undo).pack(side="left", padx=2)
        ttk.Button(btns, text="Clear", command=self._clear).pack(side="left", padx=2)
        ttk.Button(btns, text="Speak sentence", command=self._speak_sentence).pack(
            side="left", padx=2
        )

        ttk.Checkbutton(
            right, text="Speak each new phrase (TTS)", variable=self.tts_on
        ).pack(anchor="w", pady=4)

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
        self.transcript = tk.Text(right, height=14, wrap="word", font=("Consolas", 10))
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

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status.set("Could not open webcam.")
            messagebox.showerror("Camera", "Could not open webcam (index 0).")
            return
        self._tick()

    def _tick(self):
        if self.cap is None or self.detector is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(30, self._tick)
            return

        frame = cv2.flip(frame, 1)
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
            self._on_committed(committed, result["confidence"])

        rgb = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = image.resize((720, 540))
        self._photo = ImageTk.PhotoImage(image)
        self.camera_label.configure(image=self._photo)
        self.root.after(20, self._tick)

    def _on_committed(self, phrase: str, confidence: float):
        self.sentence_var.set(self.engine.sentence())
        self.transcript.insert("end", f"[{datetime.now():%H:%M:%S}] {phrase}\n")
        self.transcript.see("end")
        self.status.set(self.engine.last_event)
        if self.tts_on.get():
            self.speaker.speak(phrase)
        if self.practice_on.get():
            done = self.practice.on_phrase(phrase)
            self.practice_msg.set(self.practice.message)
            if done and self.tts_on.get():
                self.speaker.speak("Perfect")

    def _undo(self):
        self.engine.undo()
        self.sentence_var.set(self.engine.sentence())
        self.status.set(self.engine.last_event)

    def _clear(self):
        self.engine.clear()
        self.sentence_var.set("")
        self.transcript.delete("1.0", "end")
        self.status.set(self.engine.last_event)

    def _speak_sentence(self):
        text = self.engine.end_sentence()
        if not text:
            self.status.set("Nothing to speak yet.")
            return
        self.speaker.speak(text)
        self.status.set(f"Speaking: {text}")

    def _next_practice(self):
        self.practice_on.set(True)
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
        if kind == "txt":
            self.engine.export_txt(out)
        else:
            self.engine.export_json(out)
        self.status.set(self.engine.last_event)

    def _on_close(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main(model_path=None):
    ConversationStudio(model_path=model_path).run()
