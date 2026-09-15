"""Tests for the Best Practice engine (Section B). Fully separate from ARC.

B1 (parsing + collection) runs against the real docx fixture. B3 (retrieval) and
B4 (synthesis + orchestrator wiring) are fully mocked -- FakeReportStore /
FakeBPStore / a one-hot FakeEmbedder -- so they exercise the precedent + semantic
fallback, candidate formatting, the router flag, and the orchestrator swap without
ChromaDB or torch. A separation test asserts nothing here references the ARC
collections.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.chat import orchestrator
from app.chat.prompts import (
    BP_CHECKLIST_MODE_INSTRUCTION,
    BP_WRITEUP_INSTRUCTION,
    format_bp_material,
    format_bp_writeup_material,
)
from app.rag.ingest_bp import parse_bp_docx
from app.rag.waterfall import (
    ARSection,
    BPCandidate,
    BPResult,
    ReportRef,
    bp_precedent_candidates,
    bp_semantic_candidates,
    build_bp_candidates,
    lookup_bp_item,
)
from app.skills.router import RouterDecision, build_router_prompt, parse_decision

from test_orchestrator import RecordingProvider, _collect, _patch


def _docx_path() -> Path:
    from app.config import repo_path

    return repo_path("data/reference/Best_Practice_List.docx")


@pytest.fixture(scope="module")
def bp_items():
    path = _docx_path()
    if not path.exists():
        pytest.skip(f"Best Practice docx not found: {path}")
    return parse_bp_docx(path)


def test_bp_parse_counts_18_categories_115_items(bp_items):
    categories = list(dict.fromkeys(i.category for i in bp_items))
    assert len(bp_items) == 115
    assert len(categories) == 18


def test_bp_fill_ins_stripped_from_embed_text_kept_in_metadata(bp_items):
    with_fill = [i for i in bp_items if i.fill_ins]
    assert with_fill, "expected some items to carry (Insert ...) fill-ins"
    item = with_fill[0]
    # The bracketed fill-in prompt is out of the embedded text but kept as metadata.
    assert "(Insert" not in item.embed_text
    assert "insert" not in item.embed_text.lower()
    assert item.metadata["fill_ins"]
    assert item.category and item.metadata["name"]


def test_bp_every_item_has_category_and_name(bp_items):
    assert all(i.category and i.name for i in bp_items)


def test_bp_fill_ins_catch_authoring_verbs_beyond_insert(bp_items):
    # The fill-in regex must catch writer-directed parentheticals that don't use the
    # literal word "insert" (placeholder blanks, "Include ...", "only include ..."),
    # while leaving descriptive acronym expansions in the embedded text.
    by_name = {i.name: i for i in bp_items}

    for name in ("Compressed Air Pressure", "Dust Collector System", "Sustainability Goal"):
        item = by_name[name]
        assert item.fill_ins, f"expected {name!r} to carry a fill-in prompt"
        assert "(" not in item.embed_text, f"fill-in leaked into {name!r} embed_text"

    # Acronym expansions / clarifiers are real content: kept in text, not flagged.
    for name in ("Maximum MCC Temperature Setpoints", "RO System", "Cogeneration"):
        item = by_name[name]
        assert not item.fill_ins
        assert "(" in item.embed_text


def test_bp_feature_does_not_touch_arc_collections():
    # Section B must share no collection with the ARC feature.
    from app.rag import ingest_bp
    from app.rag.store import ARC_COLLECTION_NAME, ARC_PRECEDENT_COLLECTION_NAME, BP_COLLECTION_NAME

    assert BP_COLLECTION_NAME == "bp_reference"
    assert BP_COLLECTION_NAME not in (ARC_COLLECTION_NAME, ARC_PRECEDENT_COLLECTION_NAME)
    # No ARC collection name appears anywhere in the BP ingest module source.
    src = Path(ingest_bp.__file__).read_text(encoding="utf-8")
    assert "arc_reference" not in src
    assert "arc_precedent" not in src


# --- B3/B4 fakes ------------------------------------------------------------

class FakeEmbedder:
    """One-hot theme vectors so same-theme text matches exactly (like test_waterfall)."""

    THEMES = ["air", "light", "solar", "motor", "waste"]

    def _vec(self, text):
        low = text.lower()
        v = [1.0 if k in low else 0.0 for k in self.THEMES]
        if not any(v):
            v[-1] = 0.5  # unknown -> weak shared bucket
        return v

    def encode_query(self, text):
        return self._vec(text)

    def encode_documents(self, texts):
        return [self._vec(t) for t in texts]


# BP metadata rows in the bp_reference storage shape (see ingest_bp.metadata):
# fill_ins joined with " | ". Each maps to one theme via its name+template.
BP_METAS = [
    {"name": "Fix Compressed Air Leaks", "category": "Compressed Air Practices",
     "template": "repair air leaks in the system", "fill_ins": "(Number of PSI)"},
    {"name": "LED Lighting Retrofit", "category": "Efficient Lighting Practices",
     "template": "replace light fixtures with LED", "fill_ins": ""},
    {"name": "Solar Panels", "category": "Renewable Energy Practices",
     "template": "install rooftop solar panels", "fill_ins": "(Insert location of solar panels)"},
    {"name": "High Efficiency Motors", "category": "Efficient Mechanical Practices",
     "template": "upgrade to efficient motor drives", "fill_ins": ""},
    {"name": "Recycle Waste Materials", "category": "Recycling and Reusing Practices",
     "template": "recycle waste materials", "fill_ins": ""},
    {"name": "Cardboard Recycling", "category": "Recycling and Reusing Practices",
     "template": "recycle cardboard waste", "fill_ins": ""},
    {"name": "Pallet Reuse", "category": "Recycling and Reusing Practices",
     "template": "reuse wood waste pallets", "fill_ins": ""},
]


class FakeBPStore:
    """bp_reference stand-in: cosine of the query vector against each item's vector."""

    def __init__(self, metas=BP_METAS):
        self.metas = metas
        self._emb = FakeEmbedder()

    def query(self, embedding, k):
        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(x * x for x in b)) or 1.0
            return dot / (na * nb)

        scored = [
            (m.get("template", ""), m, cos(embedding, self._emb._vec(m["name"] + " " + m.get("template", ""))))
            for m in self.metas
        ]
        scored.sort(key=lambda t: t[2], reverse=True)
        return scored[:k]


class FakeReportStore:
    """Serves canned SIC matches and AR sections; no profile_search needed for BP."""

    def __init__(self, sic_map=None, ar_map=None):
        self.sic_map = sic_map or {}   # (level, value) -> [ReportRef]
        self.ar_map = ar_map or {}     # report_id -> [ARSection]
        self.ar_called_with = None

    def reports_for_sic(self, level, value):
        return list(self.sic_map.get((level, value), []))

    def ar_sections(self, report_ids):
        self.ar_called_with = set(report_ids)
        out = []
        for rid in report_ids:
            out += self.ar_map.get(rid, [])
        return out


def _ar(rid, num, title, era="2022-2026"):
    return ARSection(report_id=rid, ar_number=num, title=title, era=era)


BP_DEFAULTS = dict(
    min_reports=3, precedent_k=2, precedent_threshold=0.5,
    semantic_k=40, semantic_threshold=0.4, per_category=5,
)


# --- B3: precedent path -----------------------------------------------------

def test_bp_precedent_maps_ar_titles_to_items_and_tags_source():
    reports = [ReportRef("R1", "2022-2026"), ReportRef("R2", "2022-2026")]
    store = FakeReportStore(ar_map={
        "R1": [_ar("R1", "1", "Repair compressed air leaks")],
        "R2": [_ar("R2", "1", "Install LED lighting")],
    })
    cands = bp_precedent_candidates(
        store, FakeBPStore(), FakeEmbedder(), reports, per_ar_k=2, threshold=0.5)
    by_name = {c.name: c for c in cands}
    assert set(by_name) == {"Fix Compressed Air Leaks", "LED Lighting Retrofit"}
    assert all(c.source == "precedent" for c in cands)
    assert by_name["Fix Compressed Air Leaks"].report_ids == ["R1"]


def test_bp_precedent_dedups_item_across_reports_and_unions_provenance():
    reports = [ReportRef("R1", "2022-2026"), ReportRef("R2", "2022-2026")]
    store = FakeReportStore(ar_map={
        "R1": [_ar("R1", "1", "Fix air leaks")],
        "R2": [_ar("R2", "1", "Repair compressed air leaks")],
    })
    cands = bp_precedent_candidates(
        store, FakeBPStore(), FakeEmbedder(), reports, per_ar_k=2, threshold=0.5)
    assert len(cands) == 1  # both ARs -> the same BP item
    assert sorted(cands[0].report_ids) == ["R1", "R2"]


def test_bp_precedent_ignores_old_era_reports():
    reports = [ReportRef("O1", "2000-2006")]
    store = FakeReportStore(ar_map={"O1": [_ar("O1", "1", "Install solar panels", era="2000-2006")]})
    cands = bp_precedent_candidates(
        store, FakeBPStore(), FakeEmbedder(), reports, per_ar_k=2, threshold=0.5)
    assert cands == []  # era policy: old-era reports never mined for AR titles
    assert store.ar_called_with is None  # not even queried


# --- B3: semantic fallback --------------------------------------------------

def test_bp_semantic_caps_per_category():
    # "waste" query matches all three Recycling items; cap keeps the first two.
    cands = bp_semantic_candidates(
        FakeBPStore(), FakeEmbedder(), "recycle waste", k=40, threshold=0.4, per_category=2)
    recycling = [c for c in cands if c.category == "Recycling and Reusing Practices"]
    assert len(recycling) == 2
    assert all(c.source == "semantic" for c in cands)


# --- B3: build_bp_candidates orchestration ----------------------------------

def _precedent_store():
    return FakeReportStore(
        sic_map={("sic4", "2599"): [ReportRef("R1", "2022-2026"), ReportRef("R2", "2022-2026"),
                                    ReportRef("R3", "2022-2026")]},
        ar_map={
            "R1": [_ar("R1", "1", "Repair compressed air leaks")],
            "R2": [_ar("R2", "1", "Install LED lighting")],
            "R3": [_ar("R3", "1", "Install solar panels")],
        },
    )


def test_build_bp_precedent_only_when_enough_reports():
    res = build_bp_candidates(
        sic="2599", description="anything", store=_precedent_store(),
        bp_store=FakeBPStore(), embedder=FakeEmbedder(), **BP_DEFAULTS)
    assert res.precedent_reports == 3 and res.thin is False
    assert res.used_semantic_fallback is False  # not thin -> no fallback
    assert {c.source for c in res.candidates} == {"precedent"}
    assert res.sic_pattern == "2599"


def test_build_bp_semantic_fallback_when_precedent_thin():
    store = FakeReportStore(
        sic_map={("sic4", "2599"): [ReportRef("R1", "2022-2026")],
                 ("sic3", "259"): [ReportRef("R1", "2022-2026")],
                 ("sic2", "25"): [ReportRef("R1", "2022-2026")]},
        ar_map={"R1": [_ar("R1", "1", "Repair compressed air leaks")]})
    res = build_bp_candidates(
        sic="2599", description="waste recycling and solar", store=store,
        bp_store=FakeBPStore(), embedder=FakeEmbedder(), **BP_DEFAULTS)
    assert res.thin is True and res.used_semantic_fallback is True
    sources = {c.source for c in res.candidates}
    assert "precedent" in sources and "semantic" in sources


def test_build_bp_semantic_only_when_no_sic():
    res = build_bp_candidates(
        sic=None, description="recycle waste", store=FakeReportStore(),
        bp_store=FakeBPStore(), embedder=FakeEmbedder(), **BP_DEFAULTS)
    assert res.precedent_reports == 0 and res.used_semantic_fallback is True
    assert res.candidates and all(c.source == "semantic" for c in res.candidates)


def test_build_bp_dedups_precedent_over_semantic():
    # R1 (thin, 1 report) surfaces Solar Panels via precedent; the solar description
    # would also surface it semantically -> it must appear once, tagged precedent.
    store = FakeReportStore(
        sic_map={("sic4", "2599"): [ReportRef("R1", "2022-2026")],
                 ("sic3", "259"): [ReportRef("R1", "2022-2026")],
                 ("sic2", "25"): [ReportRef("R1", "2022-2026")]},
        ar_map={"R1": [_ar("R1", "1", "Install solar panels")]})
    res = build_bp_candidates(
        sic="2599", description="solar panels", store=store,
        bp_store=FakeBPStore(), embedder=FakeEmbedder(), **BP_DEFAULTS)
    solar = [c for c in res.candidates if c.name == "Solar Panels"]
    assert len(solar) == 1 and solar[0].source == "precedent"


# --- B4: candidate formatting -----------------------------------------------

def _bp_result(candidates, **kw):
    base = dict(precedent_reports=3, precedent_report_ids=["R1"], sic_level="sic4",
                sic_pattern="2599", used_semantic_fallback=False, thin=False)
    base.update(kw)
    return BPResult(candidates=candidates, **base)


def test_format_bp_material_groups_by_category_without_fill_in_blanks():
    cands = [
        BPCandidate("Fix Compressed Air Leaks", "Compressed Air Practices",
                    "repair air leaks", ["(Number of PSI)"], "precedent", 0.8, ["R1"]),
        BPCandidate("Solar Panels", "Renewable Energy Practices",
                    "install rooftop solar", ["(Insert location of solar panels)"], "semantic", 0.6),
    ]
    block = format_bp_material(_bp_result(cands, used_semantic_fallback=True))
    assert block.startswith("<reference_material>")
    assert "Category: Compressed Air Practices" in block
    assert "Category: Renewable Energy Practices" in block
    # Fill-in record prompts are deliberately NOT injected into the checklist material.
    assert "record:" not in block
    assert "(Number of PSI)" not in block and "(Insert" not in block
    assert "repair air leaks" in block  # the (blank-free) description still feeds the check
    assert "3 comparable facilities matched by SIC 2599" in block
    assert "semantic match on the facility description" in block
    assert "0.8" not in block and "0.6" not in block  # no scores leak in


def test_format_bp_material_omits_thin_note_and_handles_empty():
    block = format_bp_material(_bp_result(
        [BPCandidate("Solar Panels", "Renewable Energy Practices", "solar", [], "semantic", 0.5)],
        thin=True))
    assert "thin" not in block.lower()  # the thin NOTE is no longer injected
    empty = format_bp_material(_bp_result([]))
    assert "no candidate best practices" in empty.lower()


def test_bp_instruction_is_trimmed_names_only_list():
    t = BP_CHECKLIST_MODE_INSTRUCTION
    assert "OUTPUT MODE: BEST-PRACTICE PRE-ASSESSMENT CHECKLIST" in t
    assert "category" in t.lower()
    # Trimmed: the verbose "already in place" section is dropped, and assumptions,
    # inferences, suggestions, and commentary are all forbidden.
    assert 'no "already in place" section' in t.lower()
    assert "assumptions or inferences" in t.lower()
    assert "commentary" in t.lower()
    # Still forbids the old noise: checkboxes, fill-in blanks, status parentheticals,
    # scores, thin note.
    assert "checkbox" in t.lower() and "(Insert" in t
    assert "(not mentioned)" in t and "(partial - verify)" in t
    assert "no scores" in t.lower()


# --- B4: router flag --------------------------------------------------------

def test_parse_decision_reads_wants_bp_checklist():
    d = parse_decision(
        '{"skills": [], "needs_rag": true, "wants_table": false, "wants_bp_checklist": true}', {})
    assert d.wants_bp_checklist is True


def test_parse_decision_defaults_wants_bp_checklist_false():
    d = parse_decision('{"skills": [], "needs_rag": true, "wants_table": false}', {})
    assert d.wants_bp_checklist is False


def test_router_prompt_documents_wants_bp_checklist():
    prompt = build_router_prompt("a wood cabinet shop, what should I look for", [], {})
    assert "wants_bp_checklist" in prompt


# --- B4: orchestrator wiring ------------------------------------------------

def _canned_bp_result():
    return BPResult(
        candidates=[BPCandidate("Fix Compressed Air Leaks", "Compressed Air Practices",
                                "repair air leaks", ["(Number of PSI)"], "precedent", 0.8, ["R1"])],
        precedent_reports=3, precedent_report_ids=["R1"], sic_level="sic4",
        sic_pattern="2599", used_semantic_fallback=False, thin=False)


@pytest.mark.asyncio
async def test_run_chat_bp_emits_sources_and_swaps_instruction(monkeypatch):
    provider = RecordingProvider(["Category: Compressed Air..."])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True, wants_bp_checklist=True), provider)

    async def fake_build(sic, description):
        return _canned_bp_result()

    monkeypatch.setattr(orchestrator, "_build_bp_candidates", fake_build)

    events = await _collect(orchestrator.run_chat(
        "a metal stamping shop with an air compressor, what practices should I check", []))

    src = [e for e in events if e.event == "sources"]
    assert len(src) == 1
    assert src[0].data[0] == {"report_id": "R1", "fiscal_year": "",
                              "section": "comparable facility (SIC 2599)", "score": None}
    assert "OUTPUT MODE: BEST-PRACTICE PRE-ASSESSMENT CHECKLIST" in provider.system
    assert "Fix Compressed Air Leaks" in provider.system
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_run_chat_bp_falls_back_when_no_candidates(monkeypatch):
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True, wants_bp_checklist=True), provider)

    async def empty_build(sic, description):
        return BPResult(candidates=[], precedent_reports=0, precedent_report_ids=[],
                        sic_level=None, sic_pattern=None, used_semantic_fallback=False, thin=True)

    monkeypatch.setattr(orchestrator, "_build_bp_candidates", empty_build)

    events = await _collect(orchestrator.run_chat("a small plastics shop overview", []))
    assert not any(e.event == "sources" for e in events)
    assert "OUTPUT MODE: BEST-PRACTICE PRE-ASSESSMENT CHECKLIST" not in provider.system


@pytest.mark.asyncio
async def test_run_chat_non_bp_does_not_run_bp(monkeypatch):
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True, wants_bp_checklist=False), provider)

    async def fail(sic, description):  # pragma: no cover - must not be called
        raise AssertionError("BP path ran for a non-BP request")

    monkeypatch.setattr(orchestrator, "_build_bp_candidates", fail)

    await _collect(orchestrator.run_chat("What did AS0494 recommend?", []))
    assert "OUTPUT MODE: BEST-PRACTICE PRE-ASSESSMENT CHECKLIST" not in provider.system


# --- B5: single-practice write-up -------------------------------------------

def test_lookup_bp_item_returns_best_over_threshold():
    item = lookup_bp_item(FakeBPStore(), FakeEmbedder(), "solar panels", threshold=0.5)
    assert item is not None and item.name == "Solar Panels"
    assert item.source == "writeup"
    assert item.fill_ins == ["(Insert location of solar panels)"]  # blanks kept


def test_lookup_bp_item_none_below_threshold():
    # A store whose only item shares no theme with the topic: top hit scores 0.
    solar_only = FakeBPStore(metas=[BP_METAS[2]])  # Solar Panels only
    assert lookup_bp_item(solar_only, FakeEmbedder(), "compressed air leaks", threshold=0.5) is None


def test_lookup_bp_item_none_for_empty_topic():
    assert lookup_bp_item(FakeBPStore(), FakeEmbedder(), "  ", threshold=0.5) is None


@pytest.mark.parametrize("msg,topic", [
    ("write me a best practice for LED lighting", "LED lighting"),
    ("draft a best practice for compressed air leaks", "compressed air leaks"),
    ("write up the solar panels best practice", "solar panels"),
    ("give me a best practice writeup for occupancy sensors", "occupancy sensors"),
])
def test_bp_writeup_topic_parses(msg, topic):
    assert orchestrator._bp_writeup_topic(msg).lower() == topic.lower()


@pytest.mark.parametrize("msg", [
    "what are the best practices for a wood shop",       # no drafting verb
    "my SIC is 2599, what should I look for",            # checklist
    "write me a checklist of best practices for my shop",  # checklist overlap guard
    "what did AS0494 recommend",                          # answer
])
def test_bp_writeup_topic_rejects_non_writeup(msg):
    assert orchestrator._bp_writeup_topic(msg) is None


def test_format_bp_writeup_material_keeps_fill_in_blanks():
    item = BPCandidate("Solar Panels", "Renewable Energy Practices",
                       "install rooftop solar panels", ["(Insert location of solar panels)"],
                       "writeup", 0.8)
    block = format_bp_writeup_material(item)
    assert 'template for "Solar Panels"' in block
    assert "install rooftop solar panels" in block
    assert "(Insert location of solar panels)" in block  # blanks belong in a write-up
    assert "Renewable Energy Practices" in block


def test_format_bp_writeup_material_handles_no_fill_ins():
    item = BPCandidate("LED Exit Signs", "Efficient Lighting Practices",
                       "all exit signs use LED", [], "writeup", 0.7)
    block = format_bp_writeup_material(item)
    assert "placeholders to keep in the write-up: none" in block


def test_bp_writeup_instruction_is_single_practice_drafting():
    t = BP_WRITEUP_INSTRUCTION
    assert "OUTPUT MODE: BEST-PRACTICE WRITE-UP" in t
    assert "(Insert" in t  # keep the blanks as placeholders
    assert "paragraph" in t.lower()


@pytest.mark.parametrize("msg,topics", [
    # A single direct topic is unchanged.
    ("write me a best practice for LED lighting", ["LED lighting"]),
    # A comma/"and" list of topics splits into several.
    ("write me best practices for LEDs, VFDs and sludge reusing",
     ["LEDs", "VFDs", "sludge reusing"]),
    # A back-reference ("... for these") is resolved to the items described earlier,
    # with the "my facility had ..." lead-in stripped off the first fragment.
    ("my facility had LEDs, VFDs, sludge reusing. Write me best practices for these",
     ["LEDs", "VFDs", "sludge reusing"]),
])
def test_bp_writeup_topics_multi_and_backref(msg, topics):
    assert orchestrator._bp_writeup_topics(msg) == topics


def test_bp_writeup_topics_rejects_checklist_and_non_writeup():
    assert orchestrator._bp_writeup_topics("write me a checklist of best practices") == []
    assert orchestrator._bp_writeup_topics("what are the best practices for a wood shop") == []


@pytest.mark.asyncio
async def test_lookup_bp_writeups_drops_misses_and_dedups(monkeypatch):
    # "VFDs" and "led" both resolve to a template; "unicorns" matches nothing; the
    # duplicate name is collapsed.
    catalog = {
        "vfds": BPCandidate("Variable Frequency Drives", "Efficient Mechanical Practices",
                            "install VFDs on motors", [], "writeup", 0.8),
        "led": BPCandidate("LED Installation", "Efficient Lighting Practices",
                           "install LED lighting", [], "writeup", 0.8),
        "leds": BPCandidate("LED Installation", "Efficient Lighting Practices",
                            "install LED lighting", [], "writeup", 0.8),
    }

    async def fake_lookup(topic):
        return catalog.get(topic.lower())

    monkeypatch.setattr(orchestrator, "_lookup_bp_writeup", fake_lookup)
    items = await orchestrator._lookup_bp_writeups(["VFDs", "led", "unicorns", "leds"])
    assert [i.name for i in items] == ["Variable Frequency Drives", "LED Installation"]


def test_format_bp_writeup_material_renders_multiple():
    items = [
        BPCandidate("LED Installation", "Efficient Lighting Practices",
                    "install LED lighting", [], "writeup", 0.8),
        BPCandidate("Variable Frequency Drives", "Efficient Mechanical Practices",
                    "install VFDs on motors", ["(Insert motor HP)"], "writeup", 0.8),
    ]
    block = format_bp_writeup_material(items)
    assert 'template for "LED Installation"' in block
    assert 'template for "Variable Frequency Drives"' in block
    assert "install LED lighting" in block and "install VFDs on motors" in block
    assert "(Insert motor HP)" in block


@pytest.mark.asyncio
async def test_run_chat_bp_writeup_drafts_all_named_topics(monkeypatch):
    # The reported bug: "... best practices for these" must ground every named topic in
    # the best-practice doc instead of falling through to a generic essay.
    provider = RecordingProvider(["drafted"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    templates = {
        "leds": ("LED Installation", "install LED lighting"),
        "vfds": ("Variable Frequency Drives", "install VFDs on motors"),
        "sludge reusing": ("Sludge Reuse", "reuse treated sludge"),
    }

    async def fake_lookup(topic):
        hit = templates.get(topic.lower())
        if hit is None:
            return None
        name, template = hit
        return BPCandidate(name, "cat", template, [], "writeup", 0.8)

    monkeypatch.setattr(orchestrator, "_lookup_bp_writeup", fake_lookup)

    events = await _collect(orchestrator.run_chat(
        "my facility had LEDs, VFDs, sludge reusing. Write me best practices for these", []))
    answer = "".join(e.data["t"] for e in events if e.event == "token")
    # The write-up is rendered deterministically (LLM bypassed): each practice's
    # literature appears verbatim under its name heading, and the provider is never
    # streamed, so nothing else can be added.
    assert provider.system is None
    assert "## LED Installation" in answer
    assert "install LED lighting" in answer
    assert "install VFDs on motors" in answer
    assert "reuse treated sludge" in answer
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_run_chat_bp_writeup_swaps_instruction_no_sources(monkeypatch):
    provider = RecordingProvider(["This facility has installed..."])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def fake_lookup(topic):
        assert topic.lower() == "led lighting"
        return BPCandidate("Outdoor LED Lighting", "Efficient Lighting Practices",
                           "replaced outdoor lighting with LED", ["(Insert the older type of bulbs)"],
                           "writeup", 0.8)

    monkeypatch.setattr(orchestrator, "_lookup_bp_writeup", fake_lookup)

    events = await _collect(orchestrator.run_chat("write me a best practice for LED lighting", []))
    answer = "".join(e.data["t"] for e in events if e.event == "token")
    assert not any(e.event == "sources" for e in events)
    # Deterministic clean literature: heading + verbatim template + the on-site blank,
    # and nothing else (the model is never invoked).
    assert provider.system is None
    assert "## Outdoor LED Lighting" in answer
    assert "replaced outdoor lighting with LED" in answer
    assert "(Insert the older type of bulbs)" in answer
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_run_chat_bp_writeup_falls_through_when_no_match(monkeypatch):
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def no_match(topic):
        return None

    monkeypatch.setattr(orchestrator, "_lookup_bp_writeup", no_match)

    events = await _collect(orchestrator.run_chat("write me a best practice for teleportation", []))
    assert "OUTPUT MODE: BEST-PRACTICE WRITE-UP" not in provider.system  # fell through
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_run_chat_bp_writeup_falls_through_when_topics_are_not_practices(monkeypatch):
    # An explicit write verb wins over the checklist flag, but only if the named topics
    # actually resolve to BP items. Here they do not ("my wood shop" is a facility, not a
    # practice), so no write-up is emitted and it falls through to the normal path.
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True, wants_bp_checklist=True), provider)

    async def no_match(topic):
        return None

    async def empty_build(sic, description):
        return BPResult(candidates=[], precedent_reports=0, precedent_report_ids=[],
                        sic_level=None, sic_pattern=None, used_semantic_fallback=False, thin=True)

    monkeypatch.setattr(orchestrator, "_lookup_bp_writeup", no_match)
    monkeypatch.setattr(orchestrator, "_build_bp_candidates", empty_build)

    events = await _collect(orchestrator.run_chat(
        "write me a best practice for my wood shop, what should I check", []))
    answer = "".join(e.data["t"] for e in events if e.event == "token")
    # No deterministic write-up short-circuit: it fell through to the LLM path.
    assert provider.system is not None
    assert "## " not in answer
