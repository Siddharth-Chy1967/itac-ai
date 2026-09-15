"""Query-time retrieval over the report vector store (spec §5 step 2, §6).

Embeds the user's message, queries ChromaDB, and wraps the hits in a `Retrieval`
that knows whether the top result clears the configured relevance threshold. The
orchestrator uses that distinction to tell the model either "here is a real
precedent, cite it" or "nothing matched, say so plainly."

`retrieve` takes the embedder and store as arguments so it can be unit-tested
with fakes; the orchestrator wires in the real, cached singletons.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol

from ..config import MAX_LISTED_RESULTS

# (text, metadata, similarity), best first -- matches ReportStore.query.
Hit = tuple[str, dict, float]


def _report_year(fiscal_year: Optional[str]) -> int:
    """The 4-digit year in a fiscal_year label ("FY2025" -> 2025), or 0 if none.

    Older-era reports have no fiscal year ("" on disk), so they sort to the bottom
    of a newest-first ordering, which is what "latest descending" should do.
    """
    m = re.search(r"(\d{4})", fiscal_year or "")
    return int(m.group(1)) if m else 0


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
    # Enumerate mode only: keep at most this many reports after the newest-first
    # sort. None means the global MAX_LISTED_RESULTS ceiling, which also bounds any
    # larger value. See `enumerated`.
    max_reports: Optional[int] = None

    @property
    def report_cap(self) -> int:
        """Effective enumerate cap: `max_reports`, never above MAX_LISTED_RESULTS."""
        return min(self.max_reports or MAX_LISTED_RESULTS, MAX_LISTED_RESULTS)

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
        """Payload for the `sources` SSE event: one entry per confident match (capped)."""
        return [
            {
                "report_id": meta.get("report_id"),
                "fiscal_year": meta.get("fiscal_year"),
                "section": meta.get("section"),
                "score": round(score, 3),
            }
            for _text, meta, score in self.matches()[:MAX_LISTED_RESULTS]
        ]

    def enumerated(self) -> list[dict]:
        """Matches deduplicated by report, newest report first, capped at `max_reports`.

        Enumerate mode ("which reports mention X") wants each report listed once,
        not once per matching chunk. One entry per report_id keeps the report's
        best score and every distinct matching section (best-scoring first).

        Ordered latest-first by fiscal year, breaking ties (and placing older-era
        reports with no fiscal year) by relevance score, so the list reads as a
        recency-ordered index of what the corpus has on the topic.

        The relevance floor alone does not make a long tail trustworthy: on a common
        topic it admits 100+ reports, and past the newest handful the entries are
        near-ties on score with no fiscal year to rank them, so their ordering is
        effectively arbitrary and they read as noise. `max_reports` keeps only the
        newest few, where recency ordering is real. Truncation happens AFTER the sort,
        so the cap always yields the genuinely newest matches.
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
        ordered = sorted(
            by_report.values(),
            key=lambda e: (_report_year(e["fiscal_year"]), e["score"]),
            reverse=True,
        )
        return ordered[: self.report_cap]

    @property
    def truncated(self) -> bool:
        """True when the report cap actually cut reports off the enumerate list."""
        return len({m.get("report_id") for _t, m, _s in self.matches() if m.get("report_id")}) > (
            self.report_cap
        )

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
