"""Offline tests for the local inference layer: no Ollama server or OCR model needed."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import base64
import json
import os
import unittest

import pymupdf

import app.bills.local_inference as li
from app.bills.bill_extraction import BoundaryReview, PageAnalysis, PAGE_ANALYSIS_INSTRUCTIONS
from app.bills.bill_value_extraction import cache_path_for

NO_OCR = replace(li.LocalInferenceConfig(), ocr_enabled=False)
ANALYSIS = {
    "page_role": "bill_start", "utility_types": ["water"], "utility_company": "City",
    "evidence": "Heading", "account_number": None, "service_start": None,
    "service_end": None, "printed_page_number": 1, "printed_total_pages": 4,
    "bill_date": "07/21/2021", "invoice_number": None, "billing_period_label": None,
    "billing_period_key": None,
}


def make_pdf(folder: Path, name: str, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page(width=612, height=792)
        if text:
            page.insert_text((72, 72), text, fontsize=11)
    path = folder / name
    doc.save(path)
    doc.close()
    return path


def reply(content, done_reason="stop"):
    return {"message": {"role": "assistant", "content": content}, "done_reason": done_reason}


class ReadyClient(li.LocalInferenceClient):
    """Skips the Ollama readiness probe."""

    def ensure_ready(self, model=None, *, vision=True):
        return None


class OcrLayoutTests(unittest.TestCase):
    def test_lines_grouped_by_row_and_columns_kept_apart(self):
        box = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        text = li.assemble_ocr_text(
            [box(1500, 781, 1800, 811), box(100, 780, 400, 810), box(100, 830, 420, 860),
             box(2200, 832, 2400, 862), box(100, 900, 300, 930)],
            ["$3,743.63", "Previous Balance", "Payments Received", "-3,743.63", "smudge"],
            [0.99, 1.0, 0.98, 0.99, 0.2],
            min_score=0.5,
        )
        self.assertEqual(text.splitlines(), [
            "Previous Balance    $3,743.63",
            "Payments Received    -3,743.63",
        ])

    def test_empty_ocr(self):
        self.assertEqual(li.assemble_ocr_text([], [], [], 0.5), "")


class RenderingTests(unittest.TestCase):
    def test_text_layer_used_before_ocr_and_images_sized_to_budget(self):
        with TemporaryDirectory() as folder:
            pdf = make_pdf(Path(folder), "digital.pdf", [
                "City of Phoenix City Services Bill Account Number 123 Amount Due $4,521.55",
                "",
            ])
            with patch.object(li, "ocr_page", return_value="OCR TEXT") as ocr:
                pages = li.load_pdf_pages([pdf], li.LocalInferenceConfig())
        self.assertEqual([p.transcript_source for p in pages], ["PDF text layer", "OCR"])
        self.assertIn("$4,521.55", pages[0].transcript)
        self.assertEqual(pages[1].transcript, "OCR TEXT")
        ocr.assert_called_once()
        self.assertTrue(pages[0].image_png.startswith(b"\x89PNG"))
        self.assertLessEqual(sum(p.image_tokens for p in pages),
                             li.LocalInferenceConfig().image_token_budget * 1.1)

    def test_image_side_shrinks_with_page_count(self):
        config = li.LocalInferenceConfig()
        sides = [li.image_side_for(n, config) for n in (1, 2, 4, 8, 40)]
        self.assertEqual(sides[0], config.image_max_side)
        self.assertEqual(sides, sorted(sides, reverse=True))
        self.assertGreaterEqual(sides[-1], 640)

    def test_boundary_labels_follow_each_pdf(self):
        with TemporaryDirectory() as folder:
            first = make_pdf(Path(folder), "1.pdf", [""])
            second = make_pdf(Path(folder), "2.pdf", [""])
            pages = li.load_pdf_pages([first, second], NO_OCR, ["Earlier page", "Later page"])
        self.assertEqual([p.label for p in pages], ["Earlier page", "Later page"])


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.pdf = make_pdf(Path(self.temp.name), "page.pdf", [""])
        self.client = ReadyClient(NO_OCR)

    def test_request_is_local_schema_constrained_and_validated(self):
        with patch.object(li, "_post_json", return_value=reply(json.dumps(ANALYSIS))) as post:
            result = self.client.parse_pdf(
                PageAnalysis, PAGE_ANALYSIS_INSTRUCTIONS, "Analyze this supplied page.", [self.pdf]
            )
        self.assertIsInstance(result, PageAnalysis)
        self.assertEqual(result.model_dump(), ANALYSIS)
        url, payload, _timeout = post.call_args.args
        self.assertEqual(url, "http://localhost:11434/api/chat")
        self.assertEqual(payload["model"], "qwen2.5vl:latest")
        self.assertEqual(payload["format"], PageAnalysis.model_json_schema())
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["temperature"], 0)
        system, user = payload["messages"]
        self.assertIn(PAGE_ANALYSIS_INSTRUCTIONS.strip(), system["content"])
        self.assertIn("transcript", system["content"])
        self.assertIn("Analyze this supplied page.", user["content"])
        self.assertEqual(len(user["images"]), 1)
        self.assertTrue(base64.b64decode(user["images"][0]).startswith(b"\x89PNG"))

    def test_invalid_json_is_retried_with_the_error(self):
        bad = dict(ANALYSIS, page_role="first_page")
        responses = [reply(json.dumps(bad)), reply("```json\n" + json.dumps(ANALYSIS) + "\n```")]
        with patch.object(li, "_post_json", side_effect=responses) as post:
            result = self.client.parse_pdf(PageAnalysis, "rules", "prompt", [self.pdf])
        self.assertEqual(result.page_role, "bill_start")
        retry_messages = post.call_args_list[1].args[1]["messages"]
        self.assertEqual([m["role"] for m in retry_messages], ["system", "user", "assistant", "user"])
        self.assertIn("page_role", retry_messages[-1]["content"])

    def test_gives_up_after_configured_attempts(self):
        with patch.object(li, "_post_json", return_value=reply("not json")) as post, \
             self.assertRaisesRegex(RuntimeError, "did not return valid BoundaryReview"):
            self.client.parse_pdf(BoundaryReview, "rules", "prompt", [self.pdf])
        self.assertEqual(post.call_count, NO_OCR.max_attempts)

    def test_truncated_output_asks_for_more_tokens(self):
        responses = [reply('{"relationship": "same', "length"),
                     reply(json.dumps({"relationship": "same_bill", "evidence": "x"}))]
        with patch.object(li, "_post_json", side_effect=responses) as post:
            self.client.parse_pdf(BoundaryReview, "rules", "prompt", [self.pdf], max_output_tokens=512)
        self.assertEqual(post.call_args_list[1].args[1]["options"]["num_predict"], 1024)

    def test_text_only_request_has_no_images(self):
        with patch.object(li, "_post_json", return_value=reply(json.dumps(ANALYSIS))) as post:
            self.client.parse_text(PageAnalysis, "rules", "page text")
        self.assertNotIn("images", post.call_args.args[1]["messages"][1])

    def test_context_grows_for_long_requests_within_limit(self):
        self.assertEqual(self.client._context_size(1000, 2000, 2048), NO_OCR.num_ctx)
        self.assertEqual(self.client._context_size(3200 * 5, 6000, 8192), 20480)
        with self.assertRaisesRegex(RuntimeError, "OLLAMA_MAX_NUM_CTX"):
            self.client._context_size(3200 * 40, 6000, 8192)


class ReadinessAndConfigTests(unittest.TestCase):
    def test_missing_model_names_the_pull_command(self):
        client = li.LocalInferenceClient(li.LocalInferenceConfig())
        with patch.object(li, "_get_json", return_value={"models": [{"name": "llama3.1:8b"}]}), \
             self.assertRaisesRegex(RuntimeError, r"ollama pull qwen2\.5vl:latest"):
            client.ensure_ready()

    def test_server_down_explains_how_to_start_it(self):
        client = li.LocalInferenceClient(li.LocalInferenceConfig())
        with patch.object(li, "_get_json", side_effect=OSError("refused")), \
             self.assertRaisesRegex(RuntimeError, "Cannot reach Ollama"):
            client.ensure_ready()

    def test_text_only_model_is_rejected_for_images(self):
        client = li.LocalInferenceClient(li.LocalInferenceConfig())
        with patch.object(li, "_get_json", return_value={"models": [{"name": "qwen2.5vl:latest"}]}), \
             patch.object(li, "_post_json", return_value={"capabilities": ["completion"]}), \
             self.assertRaisesRegex(RuntimeError, "cannot read images"):
            client.ensure_ready()

    def test_env_settings_and_host_forms(self):
        env = {"OLLAMA_HOST": "0.0.0.0", "LOCAL_VLM_MODEL": "qwen2.5vl:7b", "OLLAMA_NUM_CTX": "8192",
               "LOCAL_OCR": "false"}
        with patch.dict(os.environ, env):
            config = li.LocalInferenceConfig.from_env()
        self.assertEqual(config.host, "http://127.0.0.1:11434")
        self.assertEqual((config.model, config.num_ctx, config.ocr_enabled), ("qwen2.5vl:7b", 8192, False))
        with patch.dict(os.environ, {"OLLAMA_NUM_CTX": "lots"}), self.assertRaisesRegex(ValueError, "whole number"):
            li.LocalInferenceConfig.from_env()

    def test_cache_label_is_a_valid_cache_folder(self):
        label = li.cache_label("qwen2.5vl:latest")
        self.assertEqual(label, "ollama-qwen2.5vl-latest")
        self.assertEqual(cache_path_for(Path("root"), "abc", label).parent.name, label)


if __name__ == "__main__":
    unittest.main()
