"""Offline call analyzer: OpenAI Whisper large-v3 + local Korean Q&A."""
from __future__ import annotations

import gc
import hashlib
import os
import queue
import shutil
import sys
import threading
from pathlib import Path
from urllib.request import Request, urlopen
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


def app_root() -> Path:
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


ROOT = app_root()
LOCAL_DATA = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Callnote"
MODEL_DIR = LOCAL_DATA / "models"
WHISPER_MODEL = MODEL_DIR / "large-v3.pt"
LLM_MODEL = MODEL_DIR / "qwen3-4b-instruct-2507"
SUPPORT_SYSTEM = """당신은 한국어 고객센터 상담 품질 분석가입니다. 대화 내용에 근거해 답하고, 없는 사실은 만들지 마세요. 통화의 핵심을 다음 관점에서 정리하세요: 고객 문의/불편, 상담원의 안내와 공감, 해결 결과와 미해결 사항, 고객 및 상담원이 약속한 후속 조치와 기한, 감정 흐름, 품질 개선 제안. 발화자나 약속 주체가 불분명하면 불분명하다고 쓰세요. 한국어로 읽기 쉽게 답하세요."""
QA_SYSTEM = "당신은 고객센터 통화 기록 도우미입니다. 제공된 전사 내용만 근거로 한국어로 답하세요. 찾을 수 없는 정보는 통화에서 확인되지 않는다고 말하세요. 약속 주체, 일정, 금액은 추측하지 마세요."
events: queue.Queue = queue.Queue()
_llm_model = None
_llm_tokenizer = None


def transcribe(audio_path: str) -> str:
    if not WHISPER_MODEL.is_file():
        raise RuntimeError("Whisper large-v3 모델을 찾을 수 없습니다. 인터넷 연결 후 모델 다운로드를 다시 시도해 주세요.")
    try:
        import imageio_ffmpeg
        ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
    packaged_ffmpeg = ROOT / "bin"
    if packaged_ffmpeg.exists():
        os.environ["PATH"] = str(packaged_ffmpeg) + os.pathsep + os.environ.get("PATH", "")
    events.put(("progress", "OpenAI Whisper large-v3 모델을 불러오는 중…"))
    import whisper
    model = whisper.load_model(str(WHISPER_MODEL), device="cpu")
    events.put(("progress", "large-v3로 한국어 통화를 전사하고 있어요…"))
    result = model.transcribe(audio_path, language="ko", task="transcribe", fp16=False, verbose=False)
    transcript = result["text"].strip()
    del model, result
    gc.collect()
    return transcript


def generate(system: str, prompt: str) -> str:
    global _llm_model, _llm_tokenizer
    if not (LLM_MODEL / "config.json").is_file() or not any(LLM_MODEL.glob("*.safetensors")):
        raise RuntimeError("질문 답변용 로컬 모델이 없습니다. 앱을 다시 시작해 모델 다운로드를 완료해 주세요.")
    import torch
    if _llm_model is None:
        events.put(("progress", "로컬 상담 분석 모델을 불러오는 중…"))
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, local_files_only=True)
        _llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    input_ids = _llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    events.put(("progress", "답변을 생성하고 있어요…"))
    with torch.inference_mode():
        output = _llm_model.generate(input_ids, max_new_tokens=700, do_sample=False, repetition_penalty=1.08)
    return _llm_tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()


def _download_whisper_checkpoint():
    import whisper
    url = whisper._MODELS["large-v3"]
    expected_sha256 = url.split("/")[-2]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if WHISPER_MODEL.exists():
        h = hashlib.sha256()
        with WHISPER_MODEL.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() == expected_sha256:
            return
        WHISPER_MODEL.unlink()

    partial = WHISPER_MODEL.with_suffix(".pt.part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urlopen(request, timeout=60) as response:
        resumed = offset > 0 and response.status == 206
        if offset and not resumed:
            offset = 0
        length = response.headers.get("Content-Length")
        total = offset + int(length) if length and length.isdigit() else 0
        mode = "ab" if resumed else "wb"
        current = offset
        digest = hashlib.sha256()
        if resumed:
            with partial.open("rb") as existing:
                for prefix in iter(lambda: existing.read(8 * 1024 * 1024), b""):
                    digest.update(prefix)
        with partial.open(mode) as output:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                current += len(chunk)
                events.put(("download_progress", "Whisper large-v3", current, total))
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError("Whisper large-v3 파일의 체크섬이 맞지 않습니다. 다시 시도해 주세요.")
    partial.replace(WHISPER_MODEL)


def _download_qwen_model():
    if (LLM_MODEL / "config.json").is_file() and any(LLM_MODEL.glob("*.safetensors")):
        return
    events.put(("download_indeterminate", "Qwen3 4B 상담 분석 모델 다운로드 중 (약 8.1GB)…"))
    from huggingface_hub import snapshot_download
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        local_dir=str(LLM_MODEL),
        ignore_patterns=["*.onnx", "*.tflite", "*.msgpack", "*.gguf"],
    )
    cache = LLM_MODEL / ".cache"
    if cache.exists():
        shutil.rmtree(cache)


def ensure_models():
    events.put(("download_indeterminate", "첫 실행용 Whisper large-v3 모델 다운로드 준비 중 (약 2.9GB)…"))
    _download_whisper_checkpoint()
    _download_qwen_model()
    events.put(("models_ready",))


class CallnoteApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("통화 인사이트 | Callnote")
        self.root.geometry("1120x780")
        self.root.minsize(850, 650)
        self.root.configure(bg="#f6f5f1")
        self.audio_path: str | None = None
        self.transcript = ""
        self.busy = False
        self.models_ready = False
        self.model_download_running = False
        self._styles()
        self._layout()
        self.root.after(100, self._poll_events)
        self._start_model_setup()

    def _styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f6f5f1")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("TLabel", background="#f6f5f1", foreground="#252a24", font=("Segoe UI", 10))
        style.configure("Muted.TLabel", foreground="#858a80", font=("Segoe UI", 9))
        style.configure("Eyebrow.TLabel", foreground="#71856d", font=("Segoe UI", 9, "bold"))
        style.configure("Title.TLabel", foreground="#222720", font=("Segoe UI", 24, "bold"))
        style.configure("Section.TLabel", background="#ffffff", foreground="#344a36", font=("Segoe UI", 11, "bold"))
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=(12, 8))
        style.configure("Primary.TButton", background="#476d4e", foreground="#ffffff")
        style.map("Primary.TButton", background=[("active", "#365a3d"), ("disabled", "#bbc7b8")])

    def _layout(self):
        outer = ttk.Frame(self.root, padding=(24, 14))
        outer.pack(fill="both", expand=True)
        top = ttk.Frame(outer); top.pack(fill="x", pady=(0, 14))
        ttk.Label(top, text="◒  callnote", font=("Segoe UI", 16, "bold"), foreground="#304b34").pack(side="left")
        ttk.Label(top, text="●  로컬 처리 · 파일과 대화 내용은 PC에만 보관", style="Muted.TLabel").pack(side="right")

        ttk.Label(outer, text="CUSTOMER SUPPORT  ·  CALL INTELLIGENCE", style="Eyebrow.TLabel").pack(anchor="w")
        ttk.Label(outer, text="통화의 맥락을 읽고 다음 행동을 정리하세요.", style="Title.TLabel", justify="left").pack(anchor="w", pady=(5, 3))
        ttk.Label(outer, text="OpenAI Whisper large-v3 음성 인식  ·  고객센터 상담 요약  ·  통화 내용 질의응답", style="Muted.TLabel").pack(anchor="w", pady=(0, 12))

        file_card = ttk.Frame(outer, style="Card.TFrame", padding=(15, 12)); file_card.pack(fill="x", pady=(0, 11))
        ttk.Label(file_card, text="WAV 통화 파일", style="Section.TLabel").pack(side="left", padx=(0, 16))
        self.file_label = ttk.Label(file_card, text="파일을 선택해 주세요 · WAV", style="Muted.TLabel")
        self.file_label.pack(side="left", fill="x", expand=True)
        self.choose_button = ttk.Button(file_card, text="파일 선택", command=self.choose_file); self.choose_button.pack(side="left", padx=5)
        self.retry_button = ttk.Button(file_card, text="모델 다시 받기", command=self._start_model_setup)
        self.analyze_button = ttk.Button(file_card, text="통화 분석 시작  →", style="Primary.TButton", command=self.analyze, state="disabled")
        self.analyze_button.pack(side="left")
        self.status = ttk.Label(outer, text="첫 실행에 음성 인식과 상담 분석 모델을 자동으로 다운로드합니다 (총 약 10.9GB).", style="Muted.TLabel")
        self.status.pack(anchor="w", pady=(0, 8))
        self.model_progress = ttk.Progressbar(outer, mode="determinate", maximum=100, length=300)
        self.model_progress.pack(fill="x", pady=(0, 8))

        panes = ttk.Panedwindow(outer, orient="horizontal"); panes.pack(fill="both", expand=True)
        left = ttk.Frame(panes, style="Card.TFrame", padding=14)
        right = ttk.Frame(panes, style="Card.TFrame", padding=14)
        panes.add(left, weight=11); panes.add(right, weight=9)
        ttk.Label(left, text="01    상담 분석", style="Section.TLabel").pack(anchor="w", pady=(0, 7))
        self.summary = ScrolledText(left, wrap="word", height=12, font=("Segoe UI", 9), bg="#fcfcfa", fg="#343a32", relief="solid", bd=1, padx=9, pady=8)
        self.summary.pack(fill="both", expand=True, pady=(0, 14)); self.summary.insert("1.0", "분석 결과가 여기에 표시됩니다.\n\n고객 문의와 불편, 상담원의 안내와 공감, 해결 여부, 후속 약속, 감정 흐름과 개선 제안을 정리합니다."); self.summary.configure(state="disabled")
        ttk.Label(left, text="02    전체 전사", style="Section.TLabel").pack(anchor="w", pady=(0, 7))
        self.transcript_view = ScrolledText(left, wrap="word", height=11, font=("Segoe UI", 9), bg="#fcfcfa", fg="#343a32", relief="solid", bd=1, padx=9, pady=8)
        self.transcript_view.pack(fill="both", expand=True); self.transcript_view.configure(state="disabled")

        ttk.Label(right, text="03    통화에 질문하기", style="Section.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Label(right, text="상담 내용에서 확인되는 정보만 답변합니다.", style="Muted.TLabel").pack(anchor="w", pady=(0, 10))
        self.conversation_panes = ttk.Panedwindow(right, orient="vertical")
        self.conversation_panes.pack(fill="both", expand=True, pady=(0, 8))
        question_panel = ttk.Frame(self.conversation_panes, style="Card.TFrame", padding=(0, 2, 0, 7))
        answer_panel = ttk.Frame(self.conversation_panes, style="Card.TFrame", padding=(0, 7, 0, 0))
        self.conversation_panes.add(question_panel, weight=1)
        self.conversation_panes.add(answer_panel, weight=1)
        question_head = ttk.Frame(question_panel, style="Card.TFrame")
        question_head.pack(fill="x", pady=(0, 6))
        ttk.Label(question_head, text="질문 작성", style="Section.TLabel").pack(side="left")
        self.ask_button = ttk.Button(question_head, text="답변 받기", style="Primary.TButton", command=self.ask)
        self.ask_button.pack(side="right")
        self.question = ScrolledText(question_panel, wrap="word", height=4, font=("Segoe UI", 9), bg="#fcfcfa", fg="#343a32", relief="solid", bd=1, padx=9, pady=8)
        self.question.pack(fill="both", expand=True)
        self.question.insert("1.0", "통화 내용에 대해 질문하세요. 여러 줄로 작성할 수 있습니다.")
        self.question.configure(fg="#92978d")
        self.question.bind("<FocusIn>", self._clear_question_hint)
        self.question.bind("<Control-Return>", self._send_shortcut)
        ttk.Label(question_panel, text="Ctrl+Enter로 보낼 수도 있어요.", style="Muted.TLabel").pack(anchor="w", pady=(5, 0))
        ttk.Label(answer_panel, text="답변 / 대화 기록", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        self.chat = ScrolledText(answer_panel, wrap="word", height=4, font=("Segoe UI", 9), bg="#fcfcfa", fg="#343a32", relief="solid", bd=1, padx=9, pady=8, state="disabled")
        self.chat.pack(fill="both", expand=True)
        self.root.after(250, self._balance_question_answer)
        ttk.Label(outer, text="Whisper: OpenAI 원본 구현 · large-v3     |     요약·질문 답변: 로컬 Qwen3 4B     |     인터넷 연결 없이 사용", style="Muted.TLabel").pack(pady=(8, 0))

    def choose_file(self):
        path = filedialog.askopenfilename(title="WAV 통화 파일 선택", filetypes=[("WAV audio", "*.wav")])
        if path:
            self.audio_path = path
            self.file_label.configure(text=Path(path).name)
            self.analyze_button.configure(state="normal" if self.models_ready and not self.busy else "disabled")

    def _set_text(self, widget, text):
        widget.configure(state="normal"); widget.delete("1.0", "end"); widget.insert("1.0", text); widget.configure(state="disabled")

    def _clear_question_hint(self, _event=None):
        if self.question.get("1.0", "end-1c") == "통화 내용에 대해 질문하세요. 여러 줄로 작성할 수 있습니다.":
            self.question.delete("1.0", "end")
            self.question.configure(fg="#343a32")

    def _balance_question_answer(self):
        self.root.update_idletasks()
        height = self.conversation_panes.winfo_height()
        if height > 100:
            self.conversation_panes.sashpos(0, height // 2)

    def _send_shortcut(self, _event=None):
        self.ask()
        return "break"

    def _busy(self, value):
        self.busy = value
        state = "disabled" if value else "normal"
        self.choose_button.configure(state=state)
        self.analyze_button.configure(state="normal" if self.audio_path and self.models_ready and not value else "disabled")
        self.ask_button.configure(state="disabled" if value else "normal")
        self.question.configure(state="disabled" if value else "normal")

    def analyze(self):
        if not self.audio_path or self.busy: return
        if not self.models_ready:
            self.status.configure(text="필요한 음성·상담 모델을 다운로드 중입니다. 완료된 뒤 분석할 수 있어요.")
            return
        self.transcript = ""; self._set_text(self.summary, ""); self._set_text(self.transcript_view, "")
        self._set_text(self.chat, "")
        self._busy(True)
        threading.Thread(target=self._analysis_job, args=(self.audio_path,), daemon=True).start()

    def _analysis_job(self, path):
        try:
            transcript = transcribe(path)
            if not transcript: raise RuntimeError("음성에서 내용을 인식하지 못했습니다.")
            events.put(("progress", "고객센터 상담 관점으로 내용을 정리하고 있어요…"))
            summary = generate(SUPPORT_SYSTEM, f"다음 상담 통화를 분석하세요.\n\n{transcript}")
            events.put(("analysis_done", transcript, summary))
        except Exception as exc:
            events.put(("error", str(exc)))

    def ask(self):
        prompt = self.question.get("1.0", "end-1c").strip()
        if prompt == "통화 내용에 대해 질문하세요. 여러 줄로 작성할 수 있습니다.":
            prompt = ""
        if not prompt or self.busy: return
        if not self.transcript:
            self.status.configure(text="먼저 WAV 파일을 선택하고 ‘통화 분석 시작’을 눌러 주세요.")
            return
        self._append_chat("나", prompt); self.question.delete("1.0", "end"); self._busy(True)
        threading.Thread(target=self._question_job, args=(prompt, self.transcript), daemon=True).start()

    def _question_job(self, prompt, transcript):
        try:
            answer = generate(QA_SYSTEM, f"통화 전사:\n{transcript}\n\n질문: {prompt}")
            events.put(("answer_done", answer))
        except Exception as exc:
            events.put(("error", str(exc)))

    def _start_model_setup(self):
        if self.model_download_running or self.models_ready:
            return
        self.model_download_running = True
        self.retry_button.pack_forget()
        self.model_progress.configure(mode="indeterminate")
        self.model_progress.start(12)
        self.status.configure(text="처음 실행에 모델을 다운로드하고 있습니다. 총 용량은 약 10.9GB입니다.")
        threading.Thread(target=self._model_setup_job, daemon=True).start()

    @staticmethod
    def _model_setup_job():
        try:
            ensure_models()
        except Exception as exc:
            events.put(("model_error", str(exc)))

    def _append_chat(self, speaker, text):
        self.chat.configure(state="normal"); self.chat.insert("end", f"{speaker}: {text}\n\n"); self.chat.see("end"); self.chat.configure(state="disabled")

    def _poll_events(self):
        try:
            while True:
                event = events.get_nowait()
                if event[0] == "progress": self.status.configure(text=event[1])
                elif event[0] == "download_indeterminate":
                    self.model_progress.configure(mode="indeterminate"); self.model_progress.start(12)
                    self.status.configure(text=event[1])
                elif event[0] == "download_progress":
                    _, name, current, total = event
                    if total:
                        self.model_progress.stop(); self.model_progress.configure(mode="determinate", maximum=total, value=current)
                        self.status.configure(text=f"{name} 다운로드 중 · {current / 1024**3:.2f} / {total / 1024**3:.2f} GB")
                elif event[0] == "models_ready":
                    self.model_download_running = False; self.models_ready = True
                    self.model_progress.stop(); self.model_progress.configure(mode="determinate", maximum=100, value=100)
                    self.status.configure(text="모델 준비가 끝났어요. WAV 통화를 선택해 분석할 수 있습니다.")
                    self.analyze_button.configure(state="normal" if self.audio_path and not self.busy else "disabled")
                elif event[0] == "model_error":
                    self.model_download_running = False; self.model_progress.stop(); self.model_progress.configure(value=0)
                    self.status.configure(text="모델 다운로드를 완료하지 못했습니다. 인터넷 연결을 확인한 뒤 다시 시도하세요.")
                    self.retry_button.pack(side="left", padx=5)
                    messagebox.showerror("모델 다운로드 오류", event[1], parent=self.root)
                elif event[0] == "analysis_done":
                    self.transcript = event[1]
                    self._set_text(self.transcript_view, event[1]); self._set_text(self.summary, event[2])
                    self.status.configure(text="분석이 끝났어요. 상담 내용에 대해 질문해 보세요."); self._busy(False)
                elif event[0] == "answer_done":
                    self._append_chat("도우미", event[1]); self._busy(False)
                elif event[0] == "error":
                    self.status.configure(text="작업을 완료하지 못했어요."); self._busy(False)
                    messagebox.showerror("오류", event[1], parent=self.root)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)


def main():
    root = tk.Tk()
    CallnoteApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
