"""Every mode that lists results to the user lists at most MAX_LISTED_RESULTS.

The cap is presentation-only: these tests hand the renderers and the sources-event
helper oversized inputs (as a wide SIC family or a common topic produces) and check
the reply never lists more than the ceiling, while retrieval inputs stay untouched.
"""
import re

from app.chat.orchestrator import _sources_event
from app.chat.prompts import (
    format_bp_material,
    format_checklist_material,
    format_enumerate_material,
    render_arc_answer,
    render_bp_writeup_answer,
    render_enumerate_answer,
    render_field_guide_answer,
    render_sic_ranking_answer,
)
from app.config import MAX_LISTED_RESULTS
from app.rag.fielddata import ARInstance, FieldGuide, FieldParameter
from app.rag.retrieve import Retrieval
from app.rag.waterfall import BPCandidate, BPResult, ChecklistResult, SicARRanking, ThemeGroup

N = 40  # well past the cap


def _hits(n: int, *, distinct_reports: bool = True):
    return [
        (
            f"chunk {i}",
            {
                "report_id": f"AS{i:04d}" if distinct_reports else "AS0001",
                "fiscal_year": f"FY{2000 + i % 26}",
                "section": f"AR-{i} Repair Compressed Air Leaks",
                "code": f"2.{4000 + i}",
                "description": "ELIMINATE LEAKS",
            },
            0.9 - i * 0.001,
        )
        for i in range(n)
    ]


def _bullets(text: str, prefix: str = "- ") -> int:
    return sum(1 for line in text.splitlines() if line.startswith(prefix))


def _numbered(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^\s*\d+\.\s", line))


def _themes(n: int) -> list[ThemeGroup]:
    return [
        ThemeGroup(
            representative=f"Theme {i}",
            examples=[f"Theme {i}"],
            report_ids=[f"AS{j:04d}" for j in range(n - i)],
            modern_report_ids=[],
        )
        for i in range(n)
    ]


def test_cap_is_ten():
    assert MAX_LISTED_RESULTS == 10


def test_answer_mode_sources_capped():
    r = Retrieval(query="q", hits=_hits(N), min_relevance=0.0)
    assert len(r.matches()) == N  # retrieval untouched
    assert len(r.sources()) == MAX_LISTED_RESULTS


def test_enumerate_uncapped_retrieval_still_capped():
    r = Retrieval(query="q", hits=_hits(N), min_relevance=0.0, max_reports=None)
    assert len(r.enumerated()) == MAX_LISTED_RESULTS
    assert len(r.enumerated_sources()) == MAX_LISTED_RESULTS
    assert r.truncated
    assert _bullets(render_enumerate_answer(r)) == MAX_LISTED_RESULTS
    assert _bullets(format_enumerate_material(r)) == MAX_LISTED_RESULTS


def test_enumerate_setting_above_cap_cannot_lift_it():
    r = Retrieval(query="q", hits=_hits(N), min_relevance=0.0, max_reports=500)
    assert len(r.enumerated()) == MAX_LISTED_RESULTS


def test_enumerate_smaller_setting_still_wins():
    r = Retrieval(query="q", hits=_hits(N), min_relevance=0.0, max_reports=5)
    assert len(r.enumerated()) == 5


def test_enumerate_under_cap_not_truncated():
    r = Retrieval(query="q", hits=_hits(4), min_relevance=0.0)
    assert len(r.enumerated()) == 4
    assert not r.truncated


def test_sources_event_capped_for_every_mode():
    payload = [{"report_id": f"AS{i:04d}"} for i in range(N)]
    event = _sources_event(payload)
    assert event.event == "sources"
    assert len(event.data) == MAX_LISTED_RESULTS
    assert event.data[0]["report_id"] == "AS0000"  # keeps the leading (best) entries


def test_arc_answer_capped_even_with_large_top_n():
    r = Retrieval(query="q", hits=_hits(N), min_relevance=0.99)  # no match -> list view
    assert _bullets(render_arc_answer(r, top_n=N)) == MAX_LISTED_RESULTS


def test_sic_ranking_capped():
    result = SicARRanking(
        themes=_themes(N), total_reports=N, modern_reports=0, total_ars=N,
        sic_level="sic4", sic_pattern="3089", requested_pattern="3089",
        widened=False, thin=False,
    )
    assert _numbered(render_sic_ranking_answer(result)) == MAX_LISTED_RESULTS


def test_checklist_material_capped():
    result = ChecklistResult(
        themes=_themes(N), total_reports=N, n_industry=N, n_process=0,
        sic_level="sic2", sic_pattern="34xx", thin=False,
    )
    assert _bullets(format_checklist_material(result), "- theme:") == MAX_LISTED_RESULTS


def test_field_guide_capped():
    instances = [ARInstance(f"AS{i:04d}", "1", "Repair Leaks", score=0.8) for i in range(12)]
    params = [
        FieldParameter(name=f"Parameter {i}", units="", report_ids=["AS0001", "AS0002"])
        for i in range(N)
    ]
    guide = FieldGuide(topic="leaks", instances=instances, parameters=params)
    assert _numbered(render_field_guide_answer(guide)) == MAX_LISTED_RESULTS


def _bp(i: int, source: str = "precedent", score: float = 0.5) -> BPCandidate:
    return BPCandidate(
        name=f"Practice {i}", category=f"Cat {i % 3}", template="t", fill_ins=[],
        source=source, score=score,
    )


def test_bp_material_capped_keeps_strongest():
    cands = [_bp(i, score=0.5 + i / 1000) for i in range(N)]
    result = BPResult(
        candidates=cands, precedent_reports=5, precedent_report_ids=[],
        sic_level="sic4", sic_pattern="3089", used_semantic_fallback=False, thin=False,
    )
    text = format_bp_material(result)
    assert _bullets(text) == MAX_LISTED_RESULTS
    assert f"Practice {N - 1}" in text  # highest score kept
    assert "Practice 0 " not in text  # lowest dropped


def test_bp_material_precedent_ahead_of_semantic():
    cands = [_bp(i, "semantic", 0.99) for i in range(N)] + [_bp(100, "precedent", 0.1)]
    result = BPResult(
        candidates=cands, precedent_reports=5, precedent_report_ids=[],
        sic_level=None, sic_pattern=None, used_semantic_fallback=True, thin=True,
    )
    assert "Practice 100" in format_bp_material(result)


def test_bp_writeup_capped():
    text = render_bp_writeup_answer([_bp(i) for i in range(N)])
    assert sum(1 for line in text.splitlines() if line.startswith("## ")) == MAX_LISTED_RESULTS
