"""Provider layer: OpenAI Responses, OpenAI-compatible and Ollama, with no network.

The OpenAI path is exercised end to end against a mocked HTTP client that
returns a response shaped like the real Responses API's, so adding an API key is
the only step left to run it for real.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from app.bills import figures
from app.bills.bill_schema import BillFigures, strict_json_schema
from app.bills.llm_providers import (
    OllamaProvider, OpenAICompatibleProvider, OpenAIResponsesProvider, ProviderConfig,
    ProviderError, make_provider,
)
from app.bills.local_inference import LocalInferenceConfig
from app.config import get_settings
from .synthetic import Month, figures_for, make_bill_pdf

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (100).to_bytes(4, "big") + (130).to_bytes(4, "big")


def settings(**changes):
    return get_settings().model_copy(update=changes)


class FakeResponse:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body) if not isinstance(body, str) else body

    def json(self):
        return self._body


class FakeHttp:
    """Records requests; replies from a queue."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def post(self, url, headers=None, json=None):
        self.requests.append({"url": url, "headers": headers, "json": json})
        return self.replies.pop(0)


def responses_api_reply(answer: dict, *, status="completed") -> dict:
    """A body shaped like POST /v1/responses returns for a reasoning model."""
    return {
        "id": "resp_68c1f0a2b3c4", "object": "response", "created_at": 1790000000,
        "status": status, "model": "gpt-6-luna",
        "incomplete_details": None if status == "completed" else {"reason": "max_output_tokens"},
        "output": [
            {"id": "rs_68c1f0a2", "type": "reasoning", "summary": []},
            {"id": "msg_68c1f0a3", "type": "message", "status": "completed", "role": "assistant",
             "content": [{"type": "output_text", "annotations": [], "logprobs": [],
                          "text": json.dumps(answer)}]},
        ],
        "reasoning": {"effort": "medium", "summary": None},
        "text": {"format": {"type": "json_schema", "name": "BillFigures", "strict": True}},
        "usage": {"input_tokens": 9416, "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens": 2873, "output_tokens_details": {"reasoning_tokens": 1792},
                  "total_tokens": 12289},
    }


class ConfigTests(unittest.TestCase):
    def test_openai_defaults_to_gpt_6_luna_on_the_responses_api(self):
        config = ProviderConfig.from_settings(settings(bills_llm_provider="openai",
                                                       bills_llm_model="", bills_llm_base_url=""))
        self.assertEqual(config.model, "gpt-6-luna")
        self.assertEqual(config.base_url, "https://api.openai.com/v1")
        self.assertEqual(config.cache_label, "openai-gpt-6-luna")

    def test_ollama_keeps_the_local_vision_model(self):
        config = ProviderConfig.from_settings(settings(bills_llm_provider="ollama",
                                                       bills_llm_model=""))
        self.assertEqual(config.model, get_settings().bills_vlm_model)
        self.assertTrue(config.cache_label.startswith("ollama-"))

    def test_cache_labels_never_collide_across_providers_or_models(self):
        labels = {
            ProviderConfig("openai", "gpt-6-luna", base_url="https://api.openai.com/v1").cache_label,
            ProviderConfig("openai", "gpt-6-sol", base_url="https://api.openai.com/v1").cache_label,
            ProviderConfig("ollama", "qwen2.5vl:latest").cache_label,
            ProviderConfig("openai_compatible", "openai/gpt-6-luna",
                           base_url="https://openrouter.ai/api/v1").cache_label,
        }
        self.assertEqual(len(labels), 4)

    def test_openai_compatible_needs_a_base_url(self):
        with self.assertRaises(ValueError):
            ProviderConfig.from_settings(settings(bills_llm_provider="openai_compatible",
                                                  bills_llm_model="x", bills_llm_base_url=""))

    def test_unknown_provider_and_effort_are_refused(self):
        with self.assertRaises(ValueError):
            ProviderConfig.from_settings(settings(bills_llm_provider="gemini"))
        with self.assertRaises(ValueError):
            ProviderConfig.from_settings(settings(bills_llm_provider="openai",
                                                  bills_llm_reasoning_effort="extreme"))

    def test_one_setting_switches_the_implementation(self):
        self.assertIsInstance(make_provider(settings(bills_llm_provider="ollama")), OllamaProvider)
        self.assertIsInstance(make_provider(settings(bills_llm_provider="openai")),
                              OpenAIResponsesProvider)
        self.assertIsInstance(make_provider(settings(
            bills_llm_provider="openai_compatible", bills_llm_model="openai/gpt-6-luna",
            bills_llm_base_url="https://openrouter.ai/api/v1")), OpenAICompatibleProvider)

    def test_a_missing_key_fails_with_the_fix(self):
        provider = make_provider(settings(bills_llm_provider="openai", bills_llm_api_key=""))
        with self.assertRaisesRegex(ProviderError, "BILLS_LLM_API_KEY"):
            provider.ensure_ready()


class StrictSchemaTests(unittest.TestCase):
    def test_every_object_is_closed_and_fully_required(self):
        schema = strict_json_schema(BillFigures)
        objects = []

        def walk(node):
            if isinstance(node, dict):
                self.assertNotIn("default", node)
                if node.get("type") == "object":
                    objects.append(node)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(schema)
        self.assertGreaterEqual(len(objects), 4)
        for node in objects:
            self.assertIs(node["additionalProperties"], False)
            self.assertEqual(sorted(node["required"]), sorted(node["properties"]))


class OpenAIResponsesTests(unittest.TestCase):
    def provider(self, http, effort="medium"):
        return OpenAIResponsesProvider(ProviderConfig(
            "openai", "gpt-6-luna", api_key="sk-test", base_url="https://api.openai.com/v1",
            reasoning_effort=effort, max_attempts=3), http=http)

    def test_request_follows_the_responses_api(self):
        answer = figures_for(Month("01/16/2026"))
        http = FakeHttp(FakeResponse(200, responses_api_reply(answer)))
        result = self.provider(http).extract([PNG, PNG], "page text", BillFigures,
                                             instructions="read the bill")
        self.assertEqual(result, answer)
        request = http.requests[0]
        self.assertEqual(request["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(request["headers"]["Authorization"], "Bearer sk-test")
        body = request["json"]
        self.assertEqual(body["model"], "gpt-6-luna")
        self.assertEqual(body["instructions"], "read the bill")
        self.assertEqual(body["reasoning"], {"effort": "medium"})
        self.assertNotIn("temperature", body)  # rejected when reasoning is on
        self.assertFalse(body["store"])
        fmt = body["text"]["format"]
        self.assertEqual((fmt["type"], fmt["name"], fmt["strict"]),
                         ("json_schema", "BillFigures", True))
        self.assertEqual(fmt["schema"], strict_json_schema(BillFigures))
        content = body["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "page text"})
        images = [part for part in content if part["type"] == "input_image"]
        self.assertEqual(len(images), 2)
        self.assertTrue(images[0]["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(images[0]["detail"], "high")

    def test_usage_is_recorded_for_costing(self):
        http = FakeHttp(FakeResponse(200, responses_api_reply(figures_for(Month("01/16/2026")))))
        provider = self.provider(http)
        provider.extract([PNG], "t", BillFigures, instructions="i")
        self.assertEqual((provider.usage.input_tokens, provider.usage.output_tokens,
                          provider.usage.reasoning_tokens), (9416, 2873, 1792))

    def test_no_reasoning_allows_temperature_zero(self):
        http = FakeHttp(FakeResponse(200, responses_api_reply(figures_for(Month("01/16/2026")))))
        self.provider(http, effort="none").extract([PNG], "t", BillFigures, instructions="i")
        self.assertEqual(http.requests[0]["json"]["temperature"], 0)
        self.assertEqual(http.requests[0]["json"]["reasoning"], {"effort": "none"})

    def test_a_rate_limit_is_retried_and_a_bad_key_is_not(self):
        answer = figures_for(Month("01/16/2026"))
        http = FakeHttp(FakeResponse(429, {"error": {"message": "slow down"}}),
                        FakeResponse(200, responses_api_reply(answer)))
        with patch("app.bills.llm_providers.time.sleep"):
            self.assertEqual(self.provider(http).extract([PNG], "t", BillFigures,
                                                         instructions="i"), answer)
        http = FakeHttp(FakeResponse(401, {"error": {"message": "bad key"}}))
        with patch("app.bills.llm_providers.time.sleep"), \
                self.assertRaisesRegex(ProviderError, "401"):
            self.provider(http).extract([PNG], "t", BillFigures, instructions="i")
        self.assertEqual(len(http.requests), 1)

    def test_refusal_and_truncation_are_errors_not_answers(self):
        refusal = responses_api_reply({})
        refusal["output"][1]["content"] = [{"type": "refusal", "refusal": "no"}]
        with self.assertRaisesRegex(ProviderError, "refused"):
            self.provider(FakeHttp(FakeResponse(200, refusal))).extract(
                [PNG], "t", BillFigures, instructions="i")
        cut = responses_api_reply({}, status="incomplete")
        with self.assertRaisesRegex(ProviderError, "max_output_tokens"):
            self.provider(FakeHttp(FakeResponse(200, cut))).extract(
                [PNG], "t", BillFigures, instructions="i")

    def test_a_bill_read_through_openai_is_verified_and_cached_per_model(self):
        """The OpenAI path end to end: page text, images, request, verify, cache."""
        month = Month("01/16/2026")
        answer = figures_for(month)
        answer["readings"].append({"label": "Electricity (kWh)", "value": "25000", "unit": "kWh",
                                   "kind": "energy", "period": "none", "page": 1})
        http = FakeHttp(FakeResponse(200, responses_api_reply(answer)))
        provider = self.provider(http)
        with tempfile.TemporaryDirectory() as folder:
            pdf = make_bill_pdf(Path(folder), "bill.pdf", month)
            fingerprint = sha256(pdf.read_bytes()).hexdigest()
            inference = LocalInferenceConfig(ocr_enabled=False)
            verified, status = figures.read_bill(pdf, fingerprint, Path(folder), provider, inference)
            again, second = figures.read_bill(pdf, fingerprint, Path(folder), provider, inference)
            cached = figures.cache_path(Path(folder), fingerprint, "openai-gpt-6-luna")
            self.assertTrue(cached.is_file())
        self.assertEqual((status, second), ("extracted", "cached"))
        self.assertEqual(len(http.requests), 1)
        values = [r.value for r in verified.figures.readings]
        self.assertNotIn("25000", values)      # the axis tick the model was fooled by
        self.assertIn("16,720.000", values)
        self.assertTrue(any("'25000' is not printed" in f for f in verified.flags))
        self.assertEqual(verified.status, "matched")
        self.assertEqual(verified.total, Decimal("3406.63"))
        text = http.requests[0]["json"]["input"][0]["content"][0]["text"]
        self.assertNotIn("25000", text)
        self.assertEqual(again.figures, verified.figures)


class GroupingThroughOpenAITests(unittest.TestCase):
    def test_page_analysis_uses_the_same_provider_and_schema(self):
        from app.bills.bill_extraction import PAGE_ANALYSIS_INSTRUCTIONS, PageAnalysis
        from app.bills.llm_providers import ProviderClient

        analysis = {"page_role": "bill_start", "utility_types": ["electricity"],
                    "utility_company": "PNM", "evidence": "Page 1 of 2", "account_number": None,
                    "service_start": None, "service_end": None, "printed_page_number": 1,
                    "printed_total_pages": 2, "bill_date": "01/16/2026", "invoice_number": None,
                    "billing_period_label": None, "billing_period_key": None}
        http = FakeHttp(FakeResponse(200, responses_api_reply(analysis)))
        provider = OpenAIResponsesProvider(ProviderConfig(
            "openai", "gpt-6-luna", api_key="k", base_url="https://api.openai.com/v1"), http=http)
        client = ProviderClient(provider, LocalInferenceConfig(ocr_enabled=False))
        result = client.parse_text(PageAnalysis, PAGE_ANALYSIS_INSTRUCTIONS, "PNM Page: 1 of 2")
        self.assertEqual(result.printed_page_number, 1)
        self.assertEqual(client.cache_label, "openai-gpt-6-luna")
        fmt = http.requests[0]["json"]["text"]["format"]
        self.assertEqual(fmt["schema"], strict_json_schema(PageAnalysis))


class OpenAICompatibleTests(unittest.TestCase):
    def test_openrouter_style_request_and_reply(self):
        answer = figures_for(Month("01/16/2026"))
        http = FakeHttp(FakeResponse(200, {
            "id": "gen-1", "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(answer)}}],
            "usage": {"prompt_tokens": 9000, "completion_tokens": 2000},
        }))
        provider = OpenAICompatibleProvider(ProviderConfig(
            "openai_compatible", "openai/gpt-6-luna", api_key="or-key",
            base_url="https://openrouter.ai/api/v1", reasoning_effort="low"), http=http)
        self.assertEqual(provider.extract([PNG], "text", BillFigures, instructions="sys"), answer)
        body = http.requests[0]["json"]
        self.assertEqual(http.requests[0]["url"], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(body["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(body["messages"][1]["content"][1]["type"], "image_url")
        self.assertEqual(body["reasoning_effort"], "low")


class OllamaTests(unittest.TestCase):
    def test_ollama_goes_through_the_local_client_with_the_schema(self):
        answer = figures_for(Month("01/16/2026"))
        provider = make_provider(settings(bills_llm_provider="ollama", bills_llm_model=""))
        reply = {"message": {"role": "assistant", "content": json.dumps(answer)},
                 "done_reason": "stop"}
        with patch("app.bills.local_inference._post_json", return_value=reply) as post:
            result = provider.extract([PNG], "text", BillFigures, instructions="sys")
        self.assertEqual(BillFigures.model_validate(result), BillFigures.model_validate(answer))
        url, payload, _ = post.call_args.args
        self.assertTrue(url.endswith("/api/chat"))
        self.assertEqual(payload["format"], BillFigures.model_json_schema())
        self.assertEqual(len(payload["messages"][1]["images"]), 1)


if __name__ == "__main__":
    unittest.main()
