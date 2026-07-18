"""Tests for ARC-code lookup (Feature 1).

Two layers:
- Fast unit tests (no model): intent detection, terse output formatting, the
  sources payload shape, and orchestrator wiring with a mocked retrieval seam.
- Integration tests against the real `arc_reference` collection for the semantic
  ranking assertions, skipped automatically when the collection is not built.
"""
from __future__ import annotations

import pytest

from app.chat import orchestrator
from app.chat.orchestrator import _arc_sources, _classify_mode
from app.chat.prompts import ARC_LOOKUP_INSTRUCTION, format_arc_material
from app.rag.retrieve import Retrieval
from app.skills.router import RouterDecision

from test_orchestrator import RecordingProvider, _collect, _patch


def _arc_hit(code, description, score, chapter="", subsection=""):
    meta = {"code": code, "description": description, "chapter": chapter, "subsection": subsection}
    return (f"{code} {description}", meta, score)


# --- intent detection -------------------------------------------------------

@pytest.mark.parametrize(
    "message",
    [
        "What's the ARC code for replacing our trash compactor?",
        "which arc code covers compressed air leaks",
        "what arc applies to occupancy sensors",
        "code this AR: install higher efficiency lamps",
        "my AR name is replace current trash haul system with in-house baler",
        "ARC for lighting occupancy controls",
        "give me the arc code for dust collection scheduling",
    ],
)
def test_classify_arc_positive(message):
    assert _classify_mode(message) == "arc"


@pytest.mark.parametrize(
    "message",
    [
        "Which reports mention solar panel cleaning?",   # enumerate, not arc
        "What did AS0494 recommend?",                    # answer
        "How do I format the AR section?",               # answer (no 'arc code')
        "Summarize the compressed air recommendation",   # answer
    ],
)
def test_classify_arc_negative(message):
    assert _classify_mode(message) != "arc"


# --- terse output formatting ------------------------------------------------

def test_arc_instruction_is_terse_code_list():
    text = ARC_LOOKUP_INSTRUCTION
    assert "OUTPUT MODE: ARC LOOKUP" in text
    assert "One line per candidate" in text
    assert "No confident ARC match. Closest candidates (verify manually):" in text
    assert "no scores" in text.lower()


def test_format_arc_material_confident_lists_codes_without_scores():
    r = Retrieval(
        query="occupancy sensors",
        hits=[
            _arc_hit("2.7135", "INSTALL OCCUPANCY SENSORS", 0.819, "2.7 Building and Grounds"),
            _arc_hit("2.7446", "UTILIZE SENSORS CONTROLLING ROOF AND WALL OPENINGS", 0.655),
            _arc_hit("4.126", "INSTALL SENSORS TO DETECT DEFECTS", 0.626),
        ],
        min_relevance=0.65,
    )
    block = format_arc_material(r)

    assert block.startswith("<reference_material>")
    assert "Confident ARC match" in block
    assert "- 2.7135 INSTALL OCCUPANCY SENSORS" in block
    # exactly top-3, best first
    assert "- 4.126 INSTALL SENSORS TO DETECT DEFECTS" in block
    # no scores leak into the prompt text
    assert "0.819" not in block
    assert "0.65" not in block


def test_format_arc_material_weak_marks_guesses():
    # Best hit below the 0.65 floor -> not confident.
    r = Retrieval(
        query="the color of the sky",
        hits=[
            _arc_hit("2.7425", "CLEAN OR COLOR ROOF TO REDUCE SOLAR LOAD", 0.632),
            _arc_hit("2.7312", "MINIMIZE USE OF OUTSIDE MAKE-UP AIR", 0.541),
        ],
        min_relevance=0.65,
    )
    block = format_arc_material(r)
    assert "No confident ARC match" in block
    assert "guesses" in block.lower()
    assert "- 2.7425 CLEAN OR COLOR ROOF TO REDUCE SOLAR LOAD" in block


def test_format_arc_material_empty():
    r = Retrieval(query="q", hits=[], min_relevance=0.65)
    block = format_arc_material(r)
    assert "no matching arc code" in block.lower()


def test_arc_sources_shape_carries_score_for_ui():
    r = Retrieval(
        query="occupancy sensors",
        hits=[
            _arc_hit("2.7135", "INSTALL OCCUPANCY SENSORS", 0.8187, "2.7 Building and Grounds"),
            _arc_hit("2.7446", "UTILIZE SENSORS CONTROLLING ROOF AND WALL OPENINGS", 0.6557, "2.7 Building and Grounds"),
        ],
        min_relevance=0.65,
    )
    assert _arc_sources(r) == [
        {
            "report_id": "2.7135",
            "fiscal_year": "2.7 Building and Grounds",
            "section": "INSTALL OCCUPANCY SENSORS",
            "score": 0.819,
        },
        {
            "report_id": "2.7446",
            "fiscal_year": "2.7 Building and Grounds",
            "section": "UTILIZE SENSORS CONTROLLING ROOF AND WALL OPENINGS",
            "score": 0.656,
        },
    ]


# --- orchestrator wiring ----------------------------------------------------

def _arc_retrieval() -> Retrieval:
    return Retrieval(
        query="occupancy sensors",
        hits=[
            _arc_hit("2.7135", "INSTALL OCCUPANCY SENSORS", 0.819, "2.7 Building and Grounds"),
            _arc_hit("2.7446", "UTILIZE SENSORS CONTROLLING ROOF AND WALL OPENINGS", 0.655),
            _arc_hit("4.126", "INSTALL SENSORS TO DETECT DEFECTS", 0.626),
        ],
        min_relevance=0.65,
    )


@pytest.mark.asyncio
async def test_run_chat_arc_emits_code_sources_and_swaps_instruction(monkeypatch):
    provider = RecordingProvider(["2.7135 INSTALL OCCUPANCY SENSORS"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def fake_arc(message):
        return _arc_retrieval()

    monkeypatch.setattr(orchestrator, "_retrieve_arc", fake_arc)

    events = await _collect(orchestrator.run_chat("what arc code for occupancy sensors", []))

    source_events = [e for e in events if e.event == "sources"]
    assert len(source_events) == 1
    assert source_events[0].data[0]["report_id"] == "2.7135"
    assert source_events[0].data[0]["score"] == 0.819

    assert provider.system is not None
    assert "OUTPUT MODE: ARC LOOKUP" in provider.system
    assert "- 2.7135 INSTALL OCCUPANCY SENSORS" in provider.system
    assert events[-1].event == "done"


@pytest.mark.asyncio
async def test_run_chat_arc_falls_back_to_answer_when_store_down(monkeypatch):
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def broken(message):
        return None

    monkeypatch.setattr(orchestrator, "_retrieve_arc", broken)

    events = await _collect(orchestrator.run_chat("what arc code for occupancy sensors", []))

    assert not any(e.event == "sources" for e in events)
    assert provider.system is not None
    assert "OUTPUT MODE: ARC LOOKUP" not in provider.system


@pytest.mark.asyncio
async def test_run_chat_arc_does_not_run_for_answer_question(monkeypatch):
    provider = RecordingProvider(["x"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def fail_arc(message):  # pragma: no cover - must not be called
        raise AssertionError("ARC retrieval ran for a non-ARC question")

    monkeypatch.setattr(orchestrator, "_retrieve_arc", fail_arc)

    await _collect(orchestrator.run_chat("What did AS0494 recommend?", []))

    assert provider.system is not None
    assert "OUTPUT MODE: ARC LOOKUP" not in provider.system


# --- verb-alignment tie-break (unit) ----------------------------------------

def test_leading_verb_reads_past_code_prefix():
    from app.chat.orchestrator import _leading_verb

    assert _leading_verb("replace current trash haul system") == "replace"
    assert _leading_verb("3.6193 INSTALL EQUIPMENT (E.G. COMPACTOR)") == "install"
    assert _leading_verb("occupancy sensors") is None  # not an action verb


def test_verb_alignment_breaks_near_tie_by_intent():
    from app.chat.orchestrator import _apply_verb_alignment

    # 3.5315 wins on raw score (literal "baler"); 3.6193's precedent title opens
    # with "Replace", matching the query verb, so the tie-break lifts it to #1.
    r = Retrieval(
        query="replace current trash haul system with in-house baler",
        hits=[
            ("3.5315 LEASE / PURCHASE BALER; SELL CARDBOARD TO RECYCLER",
             {"code": "3.5315", "description": "LEASE / PURCHASE BALER"}, 0.662),
            ("Replace Smash My Trash Service with an In-House Stationary Trash Compactor",
             {"code": "3.6193", "description": "INSTALL EQUIPMENT (E.G. COMPACTOR)",
              "ar_title": "Replace Smash My Trash Service with an In-House Stationary Trash Compactor"},
             0.654),
        ],
        min_relevance=0.65,
    )
    _apply_verb_alignment(r.query, r, boost=0.03)
    assert r.hits[0][1]["code"] == "3.6193"
    # Scores are untouched: the floor still sees the honest 0.654.
    assert r.hits[0][2] == 0.654


def test_verb_alignment_noop_without_leading_verb():
    from app.chat.orchestrator import _apply_verb_alignment

    r = Retrieval(
        query="occupancy sensors",
        hits=[
            ("2.7135 INSTALL OCCUPANCY SENSORS", {"code": "2.7135", "description": "x"}, 0.83),
            ("2.7446 UTILIZE SENSORS", {"code": "2.7446", "description": "y"}, 0.66),
        ],
        min_relevance=0.65,
    )
    _apply_verb_alignment(r.query, r, boost=0.03)
    assert [h[1]["code"] for h in r.hits] == ["2.7135", "2.7446"]  # order unchanged


# --- precedent extraction (unit) --------------------------------------------

def test_precedent_clean_title_strips_lead_and_citation():
    from app.rag.ingest_arc_precedent import _clean_title

    assert _clean_title("AR-1 Replace Smash My Trash Service with an In-House Stationary Trash Compactor (ARC 3.6193)") == \
        "Replace Smash My Trash Service with an In-House Stationary Trash Compactor"
    assert _clean_title("AR 6: Optimize RO Water Pump Performance, ARC # 2.4323.1") == \
        "Optimize RO Water Pump Performance"
    assert _clean_title("4.5\tInstall High Efficiency Motors\tARC 2.4133.1") == \
        "Install High Efficiency Motors"


def test_precedent_paragraph_texts_reads_text_box_runs():
    from app.rag.ingest_arc_precedent import _paragraph_texts

    # A text-box paragraph is still a <w:p>; joined runs reconstruct the title.
    xml = (
        "<w:p><w:r><w:t>AR-1 Replace </w:t></w:r><w:r><w:t>Compactor (ARC 3.6193)</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>   </w:t></w:r></w:p>"
    )
    assert _paragraph_texts(xml) == ["AR-1 Replace Compactor (ARC 3.6193)"]


# --- composite store dedup-by-code (unit) -----------------------------------

class _FakeArcSub:
    def __init__(self, hits):
        self._hits = hits

    def query(self, embedding, k):
        return self._hits[:k]

    def count(self):
        return len(self._hits)


def test_arc_lookup_store_dedups_by_code_keeping_best(monkeypatch):
    from app.rag import store as store_mod

    book = _FakeArcSub([
        ("3.6193 INSTALL EQUIPMENT", {"code": "3.6193", "description": "INSTALL EQUIPMENT"}, 0.55),
        ("3.5315 LEASE BALER", {"code": "3.5315", "description": "LEASE BALER"}, 0.66),
    ])
    precedent = _FakeArcSub([
        ("Replace service with in-house compactor",
         {"code": "3.6193", "description": "INSTALL EQUIPMENT", "source": "precedent"}, 0.72),
    ])
    monkeypatch.setattr(store_mod.ArcStore, "__init__", lambda self, name=None, path=None: None)
    lookup = store_mod.ArcLookupStore.__new__(store_mod.ArcLookupStore)
    lookup.book, lookup.precedent = book, precedent

    hits = lookup.query([0.0], k=5)
    codes = [m["code"] for _t, m, _s in hits]
    assert codes == ["3.6193", "3.5315"]  # 3.6193 promoted by its 0.72 precedent
    # the kept 3.6193 hit is the higher-scoring precedent one
    assert hits[0][2] == 0.72
    assert hits[0][1].get("source") == "precedent"


# --- integration: real ranking through the composite ARC path ---------------

@pytest.fixture(scope="module")
def arc_lookup():
    """Real embedder + composite ARC store + verb tie-break, or skip if not built."""
    try:
        from app.chat.orchestrator import _apply_verb_alignment
        from app.config import get_settings
        from app.rag.embeddings import Embedder
        from app.rag.retrieve import retrieve
        from app.rag.store import ArcLookupStore

        store = ArcLookupStore()
        if store.book.count() == 0:
            pytest.skip("arc_reference is empty; run python -m app.rag.ingest_arc")
        embedder = Embedder()
        s = get_settings()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ARC store/embedder unavailable: {e}")

    def _search(query, k=5):
        r = retrieve(query, embedder=embedder, store=store, k=k, min_relevance=s.rag_arc_min_relevance)
        _apply_verb_alignment(query, r, s.rag_arc_verb_boost)
        return r

    return _search


@pytest.mark.parametrize(
    "query,expected",
    [
        ("replace current trash haul system with in-house baler", "3.6193"),
        ("compressed air leak repair", "2.4236"),
        ("higher efficiency lamps", "2.7142"),
        ("occupancy sensors", "2.7135"),
    ],
)
def test_arc_semantic_top_match(arc_lookup, query, expected):
    r = arc_lookup(query)
    assert r.hits, "expected at least one ARC hit"
    assert r.hits[0][1]["code"] == expected
    assert r.has_match, "a real lookup should clear the weakness floor"


def test_arc_nonsense_query_is_not_confident(arc_lookup):
    # A nonsense query must fall below the floor so it is flagged as a guess.
    r = arc_lookup("schedule a meeting with the client next tuesday")
    assert not r.has_match
