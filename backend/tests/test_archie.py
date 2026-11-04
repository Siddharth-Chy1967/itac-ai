"""ARchie demo features: industry codes, question routing, the OpenAI chat provider.

All offline: the provider is exercised against a fake HTTP transport, never the API.
"""
import asyncio
import json

import httpx
import pytest

from app.chat.orchestrator import (
    _classify_mode, _field_topic, _industry_header, _parse_checklist_query,
)
from app.chat.prompts import BASE_SYSTEM_PROMPT, assemble_system_prompt
from app.llm import openai as openai_mod
from app.llm.openai import OpenAIProvider
from app.models import Message
from app.rag.industry import describe_sic, naics_to_sic, sic_entry


# --- industry codes ---------------------------------------------------------

def test_every_level_of_a_sic_is_named():
    entry = sic_entry("2431")
    assert entry.sic4_name == "Millwork"
    assert entry.sic3 == "243" and entry.sic2 == "24"
    assert entry.division_name == "Manufacturing"


def test_families_and_wildcards_are_described():
    assert "Furniture And Fixtures" in describe_sic("25xx")
    assert describe_sic("259").startswith("SIC group 259")


def test_a_code_missing_from_the_manual_is_said_so():
    assert "not a code in the 1987 SIC manual" in describe_sic("3028")


def test_naics_maps_back_to_sic():
    assert naics_to_sic("332710") == ("Machine Shops", ["3599"])
    assert "2011" in naics_to_sic("311611")[1]
    assert naics_to_sic("999999") is None


# --- routing ----------------------------------------------------------------

@pytest.mark.parametrize("message, topic", [
    ("what do I need for VFD AR?", "VFD"),
    ("What do I need to know about the compressed air leak AR", "compressed air leak"),
    ("what equipment should I bring for a boiler tune-up AR", "boiler tune-up"),
    ("what do i need for the occupancy sensor recommendation", "occupancy sensor"),
])
def test_any_ar_type_routes_to_the_field_guide(message, topic):
    assert _classify_mode(message) == "field"
    assert _field_topic(message) == topic


def test_an_unrelated_need_question_is_left_alone():
    assert _classify_mode("what do I need to bring to the meeting") == "answer"


@pytest.mark.parametrize("message, sic", [
    ("My next assessment is SIC 2431", "2431"),
    ("my next assessment is NAICS 332710, what ARs are common?", "3599"),
    ("most common ARs for SIC 3089", "3089"),
])
def test_next_assessment_routes_to_the_ranking(message, sic):
    assert _classify_mode(message) == "sic_rank"
    assert _parse_checklist_query(message)[0] == sic


def test_naics_prefers_the_manufacturing_sic():
    # 311611 spans livestock services (0751) and meat packing (2011).
    assert _parse_checklist_query("pre-assessment for NAICS 311611")[0] == "2011"


def test_the_header_names_the_industry_and_the_naics_translation():
    header = _industry_header("next assessment is NAICS 332710", "3599")
    assert header.startswith("Industry: SIC 3599")
    assert "NAICS 332710 (Machine Shops)" in header


# --- identity ----------------------------------------------------------------

def test_the_assistant_is_archie():
    assert "You are ARchie" in BASE_SYSTEM_PROMPT


def test_archie_knows_it_already_introduced_itself():
    prompt = " ".join(BASE_SYSTEM_PROMPT.split())
    assert "Hey, my name is ARchie, how can I help?" in prompt
    assert "Do not repeat your name" in prompt


def test_field_mode_names_no_particular_recommendation():
    system = assemble_system_prompt([], field_mode=True).split("OUTPUT MODE: SITE-VISIT")[1]
    for example in ("compressed", "LED", "VFD", "lighting", "leak"):
        assert example not in system


# --- OpenAI provider (fake transport) ----------------------------------------

def _patch_transport(monkeypatch, handler):
    real = httpx.AsyncClient

    def fake(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(openai_mod.httpx, "AsyncClient", fake)


def test_complete_reads_output_text_and_counts_usage(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        })

    _patch_transport(monkeypatch, handler)
    before = openai_mod.USAGE["calls"]
    text = asyncio.run(OpenAIProvider("k", "m").complete(
        [Message(role="user", content="hello")], "be brief", 50))
    assert text == "hi"
    assert seen["instructions"] == "be brief" and seen["store"] is False
    assert seen["max_output_tokens"] > 50  # room for reasoning
    assert openai_mod.USAGE["calls"] == before + 1


def test_json_mode_adds_the_word_json_when_missing(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"output": [{"type": "message", "content": [
            {"type": "output_text", "text": "{}"}]}]})

    _patch_transport(monkeypatch, handler)
    asyncio.run(OpenAIProvider("k", "m").complete(
        [Message(role="user", content="route this")], "", 50, json_mode=True))
    assert seen["text"]["format"]["type"] == "json_object"
    assert "json" in json.dumps(seen["input"]).lower()


def test_stream_yields_text_deltas(monkeypatch):
    events = [
        {"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
        {"type": "response.completed", "response": {"usage": {"input_tokens": 1}}},
    ]
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    _patch_transport(monkeypatch, lambda request: httpx.Response(200, text=body))

    async def collect():
        return [t async for t in OpenAIProvider("k", "m").stream(
            [Message(role="user", content="x")], "", 50)]

    assert "".join(asyncio.run(collect())) == "Hello"


def test_a_missing_key_says_how_to_fix_it():
    from app.llm.base import LLMError

    with pytest.raises(LLMError, match="LLM_PROVIDER=ollama"):
        asyncio.run(OpenAIProvider("", "m").complete([], "", 10))
