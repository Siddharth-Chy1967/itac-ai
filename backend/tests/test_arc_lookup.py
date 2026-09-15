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
from app.chat.prompts import render_arc_answer
from app.rag.retrieve import Retrieval
from app.skills.router import RouterDecision

from test_orchestrator import RecordingProvider, _collect, _patch


def _arc_hit(code, description, score, chapter="", subsection="", source="", ar_title=""):
    meta = {"code": code, "description": description, "chapter": chapter, "subsection": subsection}
    if source:
        meta["source"] = source
    if ar_title:
        meta["ar_title"] = ar_title
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


# --- deterministic answer rendering -----------------------------------------

def test_render_arc_answer_wide_gap_is_single_answer():
    # Top leads #2 by >= 0.15 -> one confident answer, only that line shown.
    r = Retrieval(
        query="replace with compactor",
        hits=[
            _arc_hit("3.6193", "INSTALL EQUIPMENT (E.G. COMPACTOR) TO REDUCE DISPOSAL COSTS", 0.872),
            _arc_hit("2.4113", "UTILIZE ENERGY-EFFICIENT BELTS", 0.572),
            _arc_hit("2.4111", "UTILIZE ENERGY-EFFICIENT BELTS AND OTHER MECH", 0.567),
        ],
        min_relevance=0.65,
    )
    ans = render_arc_answer(r, gap_threshold=0.15)
    assert "Closest ARC match:" in ans
    assert "- 3.6193 INSTALL EQUIPMENT (E.G. COMPACTOR) TO REDUCE DISPOSAL COSTS" in ans
    assert "2.4113" not in ans  # only the single answer, no runners-up
    assert "0.872" not in ans   # no scores


def test_render_arc_answer_small_gap_is_three_equal_options():
    # Top leads #2 by < 0.15 -> present all three as equal options, no ranking.
    r = Retrieval(
        query="solar panel cleaning",
        hits=[
            _arc_hit("2.6125", "CLEAN SOLAR PANELS", 0.747),
            _arc_hit("2.7425", "CLEAN OR COLOR ROOF TO REDUCE SOLAR LOAD", 0.666),
            _arc_hit("2.7123", "KEEP LAMPS AND REFLECTORS CLEAN", 0.626),
        ],
        min_relevance=0.65,
    )
    ans = render_arc_answer(r, gap_threshold=0.15)
    assert "fit equally well" in ans
    assert "- 2.6125 CLEAN SOLAR PANELS" in ans
    assert "- 2.7425 CLEAN OR COLOR ROOF TO REDUCE SOLAR LOAD" in ans
    assert "- 2.7123 KEEP LAMPS AND REFLECTORS CLEAN" in ans
    assert "0.747" not in ans


def test_render_arc_answer_weak_marks_guesses_with_real_codes():
    # Best hit below the 0.65 floor -> not confident, real closest codes shown.
    r = Retrieval(
        query="the color of the sky",
        hits=[
            _arc_hit("2.7425", "CLEAN OR COLOR ROOF TO REDUCE SOLAR LOAD", 0.632),
            _arc_hit("2.7312", "MINIMIZE USE OF OUTSIDE MAKE-UP AIR", 0.541),
        ],
        min_relevance=0.65,
    )
    ans = render_arc_answer(r)
    assert "No confident ARC match" in ans
    assert "verify manually" in ans.lower()
    assert "- 2.7425 CLEAN OR COLOR ROOF TO REDUCE SOLAR LOAD" in ans


def test_render_arc_answer_empty_refuses_to_guess():
    r = Retrieval(query="q", hits=[], min_relevance=0.65)
    ans = render_arc_answer(r)
    assert "will not guess" in ans.lower()


def test_render_arc_answer_labels_precedent_distinctly_from_book():
    # A precedent-sourced hit shows the authoritative book title AND the real report
    # phrasing on a labeled sub-line, so the two are never conflated.
    r = Retrieval(
        query="replace trash service with in-house compactor",
        hits=[
            _arc_hit("3.6193", "INSTALL EQUIPMENT (E.G. COMPACTOR) TO REDUCE DISPOSAL COSTS", 0.83,
                     source="precedent",
                     ar_title="Replace Smash My Trash Service with an In-House Stationary Trash Compactor"),
        ],
        min_relevance=0.65,
    )
    ans = render_arc_answer(r, gap_threshold=0.15)
    assert "- 3.6193 INSTALL EQUIPMENT (E.G. COMPACTOR) TO REDUCE DISPOSAL COSTS" in ans
    assert "used in past reports as: Replace Smash My Trash Service" in ans


def test_render_arc_answer_cannot_emit_a_code_not_retrieved():
    # Determinism guarantee: only retrieved codes can appear; nothing else can.
    r = Retrieval(
        query="find arc code for ev charging stations",
        hits=[
            _arc_hit("4.621", "USE FIXTURES TO REDUCE MACHINE CHANGEOUT TIMES", 0.642),
            _arc_hit("2.8114", "CHANGE RATE SCHEDULES OR OTHER CHANGES IN UTILITY SERVICE", 0.634),
        ],
        min_relevance=0.65,
    )
    ans = render_arc_answer(r, gap_threshold=0.15)
    assert "2.1421" not in ans  # the fabricated code from the incident cannot appear
    assert "No confident ARC match" in ans and "4.621" in ans


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
async def test_run_chat_arc_emits_deterministic_answer_bypassing_llm(monkeypatch):
    provider = RecordingProvider(["SHOULD NOT BE USED"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def fake_arc(message):
        return _arc_retrieval()

    monkeypatch.setattr(orchestrator, "_retrieve_arc", fake_arc)

    events = await _collect(orchestrator.run_chat("what arc code for occupancy sensors", []))

    source_events = [e for e in events if e.event == "sources"]
    assert len(source_events) == 1
    assert source_events[0].data[0]["report_id"] == "2.7135"
    assert source_events[0].data[0]["score"] == 0.819

    # ARC bypasses the LLM entirely: provider.stream is never called.
    assert provider.system is None
    tokens = "".join(e.data["t"] for e in events if e.event == "token")
    assert "- 2.7135 INSTALL OCCUPANCY SENSORS" in tokens
    assert "SHOULD NOT BE USED" not in tokens  # the model's output is not used
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


@pytest.fixture(scope="module")
def arc_lookup_store():
    """Real composite ARC store (no embedder needed for exact code lookup)."""
    try:
        from app.rag.store import ArcLookupStore

        store = ArcLookupStore()
        if store.book.count() == 0:
            pytest.skip("arc_reference is empty; run python -m app.rag.ingest_arc")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ARC store unavailable: {e}")
    return store


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


# --- A1: direct (reverse) code lookup ---------------------------------------

@pytest.mark.parametrize("message", [
    "what is the name for ARC code 3.4113",
    "what does 2.7135 mean",
    "ARC 3.6193",
])
def test_literal_code_routes_to_arc(message):
    assert _classify_mode(message) == "arc"


@pytest.mark.parametrize("code,expected_desc", [
    ("3.4113", "TREAT AND REUSE RINSE WATERS"),
    ("2.7135", "INSTALL OCCUPANCY SENSORS"),
])
def test_direct_code_lookup_exact(arc_lookup_store, code, expected_desc):
    hit = arc_lookup_store.get_by_code(code)
    assert hit is not None
    assert hit[1]["code"] == code
    assert hit[1]["description"] == expected_desc


def test_direct_code_lookup_unknown_code(arc_lookup_store):
    assert arc_lookup_store.get_by_code("9.9999") is None


@pytest.mark.asyncio
async def test_run_chat_direct_code_lookup_single_answer(monkeypatch):
    provider = RecordingProvider(["3.4113 TREAT AND REUSE RINSE WATERS"])
    _patch(monkeypatch, RouterDecision(skills=[], needs_rag=True), provider)

    async def fake_direct(message):
        return Retrieval(
            query=message,
            hits=[("3.4113 TREAT AND REUSE RINSE WATERS",
                   {"code": "3.4113", "description": "TREAT AND REUSE RINSE WATERS", "chapter": ""}, 1.0)],
            min_relevance=0.65,
        )

    monkeypatch.setattr(orchestrator, "_direct_arc_lookup", fake_direct)

    events = await _collect(orchestrator.run_chat("what is the name for ARC code 3.4113", []))
    # Deterministic: a single confident hit renders as the one closest-match line,
    # verbatim, without the LLM.
    assert provider.system is None
    tokens = "".join(e.data["t"] for e in events if e.event == "token")
    assert "Closest ARC match:" in tokens
    assert "- 3.4113 TREAT AND REUSE RINSE WATERS" in tokens
    src = [e for e in events if e.event == "sources"]
    assert src and src[0].data[0]["report_id"] == "3.4113"


# --- A2: forward lookup, water-reuse rephrasings stay correct ----------------

@pytest.mark.parametrize("query", [
    "reuse treated paint wash water",
    "reuse treated ink wash water",
    "reuse dye wash water",
    "reuse degreasing rinse water",
])
def test_arc_water_reuse_top_match_is_3_4113(arc_lookup, query):
    # Regression guard: the water/action concept must beat the modifier noun.
    r = arc_lookup(query)
    assert r.hits[0][1]["code"] == "3.4113"


# --- A3: dynamic display against real scores --------------------------------

def test_arc_display_compactor_is_single(arc_lookup):
    from app.config import get_settings
    r = arc_lookup("replace smash my trash service with an in-house trash compactor")
    ans = render_arc_answer(r, gap_threshold=get_settings().rag_arc_gap_threshold)
    assert "Closest ARC match:" in ans
    assert "3.6193" in ans


def test_arc_display_solar_cleaning_is_tied(arc_lookup):
    from app.config import get_settings
    r = arc_lookup("solar panel cleaning")
    ans = render_arc_answer(r, gap_threshold=get_settings().rag_arc_gap_threshold)
    assert "fit equally well" in ans
