"""Tests for RAG retrieval + reference-material formatting (spec §5 step 2, §6).

Uses a fake embedder and a fake store holding dummy ChromaDB-shaped hits so the
threshold logic (match vs no confident match) is exercised without loading the
embedding model or opening a real vector store.
"""
from __future__ import annotations

from app.chat.prompts import format_reference_material
from app.rag.retrieve import Retrieval, retrieve


class FakeEmbedder:
    def encode_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeStore:
    """Returns preset (text, metadata, similarity) hits, best-first, like ReportStore."""

    def __init__(self, hits):
        self._hits = hits
        self.last_k = None

    def query(self, query_embedding, k):
        self.last_k = k
        return self._hits[:k]


def _hit(report_id, fy, section, score, text="chunk text"):
    return (text, {"report_id": report_id, "fiscal_year": fy, "section": section}, score)


# --- retrieve() + Retrieval threshold logic -------------------------------

def test_retrieve_passes_k_and_returns_hits():
    store = FakeStore([_hit("AS0599", "FY2025", "AR #1", 0.9)])
    r = retrieve("q", embedder=FakeEmbedder(), store=store, k=6, min_relevance=0.6)
    assert store.last_k == 6
    assert r.top_score == 0.9


def test_match_when_top_hit_clears_threshold():
    store = FakeStore([_hit("AS0599", "FY2025", "AR #1", 0.85), _hit("AS0600", "FY2025", "2. Bkgd", 0.5)])
    r = retrieve("q", embedder=FakeEmbedder(), store=store, k=6, min_relevance=0.6)

    assert r.has_match is True
    # Only the above-threshold hit is a match; the 0.5 hit is filtered out.
    assert len(r.matches()) == 1
    assert r.sources() == [
        {"report_id": "AS0599", "fiscal_year": "FY2025", "section": "AR #1", "score": 0.85}
    ]


def test_no_match_when_top_hit_below_threshold():
    store = FakeStore([_hit("AS0123", "FY2019", "2. Background", 0.41)])
    r = retrieve("q", embedder=FakeEmbedder(), store=store, k=6, min_relevance=0.6)

    assert r.has_match is False
    assert r.matches() == []
    assert r.sources() == []
    assert r.best is not None and r.best[1]["report_id"] == "AS0123"


def test_no_match_when_store_empty():
    r = retrieve("q", embedder=FakeEmbedder(), store=FakeStore([]), k=6, min_relevance=0.6)
    assert r.has_match is False
    assert r.best is None
    assert r.top_score == 0.0


# --- format_reference_material: the two prompt paths -----------------------

def test_format_match_requires_citation():
    r = Retrieval(
        query="q",
        hits=[_hit("AS0599", "FY2025", "AR #1: Repair leaks", 0.85, text="Anticipated savings 45,000 kWh/yr.")],
        min_relevance=0.6,
    )
    block = format_reference_material(r)

    assert block.startswith("<reference_material>")
    assert block.endswith("</reference_material>")
    assert "report_id=AS0599" in block
    assert "fiscal_year=FY2025" in block
    assert "cite" in block.lower()
    assert "45,000 kWh/yr" in block  # the actual chunk content is included


def test_format_no_match_offers_closest_not_precedent():
    r = Retrieval(
        query="q",
        hits=[_hit("AS0123", "FY2019", "2. Background", 0.41)],
        min_relevance=0.6,
    )
    block = format_reference_material(r)

    assert "confident match" in block.lower()
    assert "closest" in block.lower()
    assert "AS0123" in block
    assert "FY2019" in block
    assert "0.41" in block  # the weak similarity is disclosed
    assert "0.60" in block  # against the threshold
    assert "not present it as an established precedent" in block.lower()


def test_format_no_match_empty_store():
    r = Retrieval(query="q", hits=[], min_relevance=0.6)
    block = format_reference_material(r)

    assert "no matching report" in block.lower()
    assert "returned no results" in block.lower()
