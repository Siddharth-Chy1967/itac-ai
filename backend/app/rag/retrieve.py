"""Query-time retrieval over the report vector store (spec §5 step 2, §6).

Embeds the user's message, queries ChromaDB, and wraps the hits in a `Retrieval`
that knows whether the top result clears the configured relevance threshold. The
orchestrator uses that distinction to tell the model either "here is a real
precedent, cite it" or "nothing matched, say so plainly."

`retrieve` takes the embedder and store as arguments so it can be unit-tested
with fakes; the orchestrator wires in the real, cached singletons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

# (text, metadata, similarity), best first -- matches ReportStore.query.
Hit = tuple[str, dict, float]


class _Embedder(Protocol):
    def encode_query(self, text: str) -> list[float]: ...


class _Store(Protocol):
    def query(self, query_embedding: list[float], k: int) -> list[Hit]: ...


@dataclass
class Retrieval:
    """The outcome of retrieving for one message."""

    query: str
    hits: list[Hit]
    min_relevance: float

    @property
    def best(self) -> Optional[Hit]:
        return self.hits[0] if self.hits else None

    @property
    def top_score(self) -> float:
        return self.hits[0][2] if self.hits else 0.0

    @property
    def has_match(self) -> bool:
        """True when the closest hit clears the relevance threshold.

        Scores are cosine similarities (`ReportStore.query` returns
        1 - cosine_distance), so higher means more relevant and a hit matches
        when its score is at or above the floor.
        """
        return self.top_score >= self.min_relevance

    def matches(self) -> list[Hit]:
        """Hits at or above the threshold, best first (empty when no confident match)."""
        return [h for h in self.hits if h[2] >= self.min_relevance]

    def sources(self) -> list[dict]:
        """Payload for the `sources` SSE event: one entry per confident match."""
        return [
            {
                "report_id": meta.get("report_id"),
                "fiscal_year": meta.get("fiscal_year"),
                "section": meta.get("section"),
                "score": round(score, 3),
            }
            for _text, meta, score in self.matches()
        ]

    def enumerated(self) -> list[dict]:
        """Matches deduplicated by report, best report first.

        Enumerate mode ("which reports mention X") wants each report listed once,
        not once per matching chunk. One entry per report_id keeps the report's
        best score and every distinct matching section (best-scoring first).
        """
        by_report: dict[str, dict] = {}
        for _text, meta, score in self.matches():
            rid = meta.get("report_id")
            if not rid:
                continue
            entry = by_report.get(rid)
            if entry is None:
                entry = by_report[rid] = {
                    "report_id": rid,
                    "fiscal_year": meta.get("fiscal_year"),
                    "score": score,
                    "sections": [],
                }
            section = meta.get("section")
            if section and section not in entry["sections"]:
                entry["sections"].append(section)
        return sorted(by_report.values(), key=lambda e: e["score"], reverse=True)

    def enumerated_sources(self) -> list[dict]:
        """Payload for the `sources` SSE event in enumerate mode: one entry per report.

        The score stays in this payload so the UI source cards can show it; the
        enumerate chat text itself never states scores.
        """
        return [
            {
                "report_id": e["report_id"],
                "fiscal_year": e["fiscal_year"],
                "section": "; ".join(e["sections"]),
                "score": round(e["score"], 3),
            }
            for e in self.enumerated()
        ]


def retrieve(
    message: str,
    *,
    embedder: _Embedder,
    store: _Store,
    k: int,
    min_relevance: float,
) -> Retrieval:
    """Embed `message`, query the store, and return a threshold-aware `Retrieval`."""
    query_vec = embedder.encode_query(message)
    hits = store.query(query_vec, k=k)
    return Retrieval(query=message, hits=hits, min_relevance=min_relevance)
