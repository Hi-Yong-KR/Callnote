import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import app


class Tokenizer:
    eos_token_id = 0

    def apply_chat_template(self, *args, **kwargs):
        return {"input_ids": torch.tensor([[900, 901]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)}

    def decode(self, tokens, **kwargs):
        return "".join(f"항목{int(token)} " for token in tokens if int(token) != 0)


class Model:
    device = torch.device("cpu")
    config = SimpleNamespace(max_position_embeddings=4096)
    generation_config = SimpleNamespace(eos_token_id=0)

    def __init__(self, count=350, fail=False, stop_after=None, fail_after=None):
        self.count, self.fail, self.stop_after, self.fail_after = count, fail, stop_after, fail_after

    def generate(self, inputs, *, max_new_tokens, streamer, stopping_criteria, **kwargs):
        if self.fail:
            raise RuntimeError("test inference failure")
        streamer.put(inputs)
        generated = []
        for token in (list(range(1, self.count + 1)) + [0])[:max_new_tokens]:
            generated.append(token)
            streamer.put(torch.tensor([token]))
            if self.fail_after and len(generated) == self.fail_after:
                raise RuntimeError("test failure after partial output")
            if self.stop_after and len(generated) == self.stop_after:
                app._generation_stop.set()
            if token == 0 or bool(stopping_criteria(inputs, None)[0]):
                break
        streamer.end()
        return torch.cat((inputs, torch.tensor([generated])), dim=1)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp.name)
        (self.model_path / "config.json").touch()
        (self.model_path / "model.safetensors").touch()
        self.patches = [patch.object(app, "LLM_MODEL", self.model_path),
                        patch.object(app, "_llm_model", Model()),
                        patch.object(app, "_llm_tokenizer", Tokenizer()),
                        patch.object(app, "events", queue.Queue())]
        for replacement in self.patches:
            replacement.start()

    def tearDown(self):
        for replacement in reversed(self.patches):
            replacement.stop()
        app._generation_stop.clear()
        self.temp.cleanup()

    def test_answer_and_summary_exceed_old_limits_and_reach_eos(self):
        for system in (app.QA_SYSTEM, app.SUPPORT_SYSTEM):
            with self.subTest(system=system):
                result = app.generate(system, "검증용 질문")
                self.assertEqual(result.finish_reason, "eos")
                self.assertIn("항목350", result.text)
        events = list(app.events.queue)
        self.assertTrue(any(event[0] == "generation_text" for event in events))
        streamed = "".join(event[2] for event in events if event[0] == "generation_text" and event[1] == "answer")
        self.assertEqual(streamed.strip(), result.text)

    def test_worker_error_propagates_without_hanging_stream(self):
        app._llm_model = Model(fail=True)
        result = []

        def run():
            try:
                app.generate(app.QA_SYSTEM, "오류 검증")
            except Exception as exc:
                result.append(exc)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive(), "Streamer hung after model failure")
        self.assertIsInstance(result[0], RuntimeError)

    def test_user_stop_keeps_partial_text_and_next_request_can_finish(self):
        app._llm_model = Model(stop_after=7)
        result = app.generate(app.QA_SYSTEM, "중지 검증")
        self.assertEqual(result.finish_reason, "stopped")
        self.assertIn("항목7", result.text)
        self.assertNotIn("항목8 ", result.text)
        app._llm_model = Model(count=10)
        self.assertEqual(app.generate(app.QA_SYSTEM, "다시 생성").finish_reason, "eos")

    def test_context_exhaustion_is_reported_explicitly(self):
        app._llm_model.config = SimpleNamespace(max_position_embeddings=5)
        result = app.generate(app.QA_SYSTEM, "컨텍스트 검증")
        self.assertEqual(result.finish_reason, "context_limit")
        app._llm_model.config = SimpleNamespace(max_position_embeddings=2)
        with self.assertRaisesRegex(RuntimeError, "최대 처리 길이"):
            app.generate(app.QA_SYSTEM, "입력 초과 검증")

    def test_error_after_partial_output_keeps_streamed_chunks(self):
        app._llm_model = Model(fail_after=3)
        with self.assertRaisesRegex(RuntimeError, "partial output"):
            app.generate(app.QA_SYSTEM, "부분 출력 오류 검증")
        streamed = "".join(e[2] for e in app.events.queue if e[0] == "generation_text")
        self.assertIn("항목3", streamed)


class TranscriptTests(unittest.TestCase):
    def test_segment_paragraphs_timestamps_and_empty_segments(self):
        result = {"text": "평탄화된 원문", "segments": [
            {"start": 0.2, "text": " 안녕하세요. "},
            {"start": 5.8, "text": "배송 조회를 부탁드립니다."},
            {"start": 6, "text": " "},
            {"start": 3661, "text": "확인해 드리겠습니다."}]}
        self.assertEqual(app.format_transcript(result),
                         "[00:00] 안녕하세요.\n\n[00:05] 배송 조회를 부탁드립니다.\n\n[01:01:01] 확인해 드리겠습니다.")

    def test_plain_text_fallback_keeps_decimals(self):
        self.assertEqual(app.format_transcript({"text": "3.5일 걸립니다. 감사합니다."}),
                         "3.5일 걸립니다.\n\n감사합니다.")
        self.assertEqual(app.format_transcript({"text": "", "segments": []}), "")


class DisplayTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.withdraw()
        self.replacements = [patch.object(app, "events", queue.Queue()),
                             patch.object(app.CallnoteApp, "_start_model_setup")]
        for replacement in self.replacements:
            replacement.start()
        self.ui = app.CallnoteApp(self.root)

    def tearDown(self):
        for callback in self.root.tk.call("after", "info"):
            self.root.after_cancel(callback)
        self.root.destroy()
        for replacement in reversed(self.replacements):
            replacement.stop()

    def test_stream_finalization_does_not_duplicate_or_erase_previous_answers(self):
        for number in (1, 2):
            self.ui._append_chat("나", f"질문{number}")
            app.events.put(("generation_start", "answer"))
            app.events.put(("generation_text", "answer", "일부 "))
            app.events.put(("generation_text", "answer", f"답변{number}."))
            app.events.put(("answer_done", f"일부 답변{number}.", "eos"))
            self.ui._poll_events()
        text = self.ui.chat.get("1.0", "end-1c")
        for number in (1, 2):
            self.assertEqual(text.count(f"도우미: 일부 답변{number}."), 1)

    def test_stopped_summary_preserves_transcript_and_partial_summary(self):
        transcript = "[00:00] 안녕하세요.\n\n[00:05] 문의드립니다."
        for event in [("transcript_ready", transcript), ("generation_start", "summary"),
                      ("generation_text", "summary", "요약 일부"),
                      ("analysis_done", transcript, "요약 일부", "stopped")]:
            app.events.put(event)
        self.ui._poll_events()
        self.assertEqual(self.ui.transcript, transcript)
        self.assertEqual(self.ui.summary.get("1.0", "end-1c"), "요약 일부")
        self.assertIn("중지", self.ui.status.cget("text"))
        self.assertFalse(self.ui.busy)

    def test_live_output_stop_control_and_error_preserve_partial_answer(self):
        app.events.put(("generation_start", "answer"))
        app.events.put(("generation_text", "answer", "먼저 나온 답변"))
        self.ui._poll_events()
        self.assertIn("먼저 나온 답변", self.ui.chat.get("1.0", "end-1c"))
        self.assertEqual(self.ui.stop_button.winfo_manager(), "pack")
        app.events.put(("error", "검증용 오류"))
        with patch.object(app.messagebox, "showerror") as dialog:
            self.ui._poll_events()
        self.assertIn("먼저 나온 답변", self.ui.chat.get("1.0", "end-1c"))
        self.assertIn("오류", self.ui.chat.get("1.0", "end-1c"))
        self.assertEqual(self.ui.stop_button.winfo_manager(), "")
        dialog.assert_called_once()


if __name__ == "__main__":
    unittest.main()
