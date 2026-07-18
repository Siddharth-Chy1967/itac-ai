"""ChromaDB vector store for report chunks (spec §6.3).

One persistent collection, `itac_reports`, on cosine distance. We pass our own
embeddings (from `embeddings.Embedder`) rather than letting Chroma pick a default
embedding function, so ingest and query always use the configured model.

Chunk IDs are deterministic (`{report_id}#{chunk_index}`) and writes use `upsert`,
so re-ingesting a report replaces its chunks instead of erroring on duplicates.
`has_report` / `ingested_report_ids` back the ingest CLI's idempotency: a report
already present is skipped on the next run.

The chromadb import is deferred to construction so importing this module is cheap.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import get_settings, repo_path

if TYPE_CHECKING:
    from .chunker import Chunk

COLLECTION_NAME = "itac_reports"
ARC_COLLECTION_NAME = "arc_reference"
# Real (AR title -> ARC code) precedent pairs mined from past reports, kept
# separate from the book collection so each can be rebuilt independently.
ARC_PRECEDENT_COLLECTION_NAME = "arc_precedent"


class ReportStore:
    def __init__(self, path: str | None = None):
        import chromadb  # lazy: heavy import

        db_path = repo_path(path or get_settings().chroma_path)
        db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def has_report(self, report_id: str) -> bool:
        """True if any chunk for this report is already stored (idempotency check)."""
        res = self.collection.get(where={"report_id": report_id}, limit=1)
        return bool(res.get("ids"))

    def ingested_report_ids(self) -> set[str]:
        """The set of report_ids already in the store."""
        res = self.collection.get(include=["metadatas"])
        return {m["report_id"] for m in res.get("metadatas", []) if m.get("report_id")}

    def add_chunks(self, chunks: list["Chunk"], embeddings: list[list[float]]) -> None:
        """Upsert a batch of chunks with their precomputed embeddings."""
        if not chunks:
            return
        ids = [f"{c.metadata['report_id']}#{c.metadata['chunk_index']}" for c in chunks]
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def delete_report(self, report_id: str) -> None:
        self.collection.delete(where={"report_id": report_id})

    def query(self, query_embedding: list[float], k: int = 6) -> list[tuple[str, dict, float]]:
        """Return up to k (document, metadata, similarity) tuples for a query vector.

        Similarity is 1 - cosine_distance, so higher is closer.
        """
        res = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return [(doc, meta, 1.0 - dist) for doc, meta, dist in zip(docs, metas, dists)]

    def list_docs(self) -> list[dict]:
        """Distinct reports in the store with their chunk counts (backs GET /api/documents)."""
        res = self.collection.get(include=["metadatas"])
        by_report: dict[str, dict] = {}
        for m in res.get("metadatas", []):
            rid = m.get("report_id")
            if not rid:
                continue
            entry = by_report.setdefault(
                rid,
                {
                    "report_id": rid,
                    "fiscal_year": m.get("fiscal_year"),
                    "era": m.get("era"),
                    "n_chunks": 0,
                },
            )
            entry["n_chunks"] += 1
        return list(by_report.values())

    def count(self) -> int:
        return self.collection.count()


class ArcStore:
    """Vector store for an ARC entry collection (spec: ARC lookup).

    Backs two collections with the same shape: `arc_reference` (the official
    leaf-code book, from the PDF) and `arc_precedent` (real AR-title -> code pairs
    mined from past reports). Both are kept distinct from the report corpus so an
    ARC lookup never returns report chunks and vice versa. Same Chroma client and
    same nomic embeddings (passed in) as ReportStore, so `retrieve()` drives them
    unchanged. Each is small and fully derived from source, so ingest rebuilds the
    whole collection (`reset`) rather than upserting incrementally.
    """

    def __init__(self, collection_name: str = ARC_COLLECTION_NAME, path: str | None = None):
        import chromadb  # lazy: heavy import

        self.collection_name = collection_name
        db_path = repo_path(path or get_settings().chroma_path)
        db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Drop and recreate the collection so a re-ingest starts from empty."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_entries(
        self, ids: list[str], texts: list[str], metadatas: list[dict], embeddings: list[list[float]]
    ) -> None:
        """Add ARC entries under the given Chroma IDs (unique per entry)."""
        if not ids:
            return
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], k: int = 5) -> list[tuple[str, dict, float]]:
        """Return up to k (document, metadata, similarity) tuples, best first.

        Same shape and convention as ReportStore.query: similarity is
        1 - cosine_distance, so higher is closer. This is what lets `retrieve()`
        drive ARC lookup without a parallel search path.
        """
        res = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return [(doc, meta, 1.0 - dist) for doc, meta, dist in zip(docs, metas, dists)]

    def count(self) -> int:
        return self.collection.count()


class ArcLookupStore:
    """Composite ARC store: the official book plus mined report precedents.

    Queries both collections and merges the hits, then deduplicates by ARC code
    keeping the best-scoring hit per code. This is what lets a lookup match real
    assessor phrasing ("Replace Smash My Trash Service with an In-House Stationary
    Trash Compactor" -> 3.6193) that the terse official description alone misses,
    while still guaranteeing every code is representable via the book collection.

    Exposes the same `query(embedding, k)` shape as the other stores, so
    `retrieve()` drives it with no parallel search path. The precedent collection
    is optional: if it is empty (not yet mined), lookup degrades to the book only.
    """

    def __init__(self, path: str | None = None):
        self.book = ArcStore(ARC_COLLECTION_NAME, path)
        self.precedent = ArcStore(ARC_PRECEDENT_COLLECTION_NAME, path)

    def query(self, query_embedding: list[float], k: int = 5) -> list[tuple[str, dict, float]]:
        # Over-fetch from each collection so that after dedup-by-code we still have
        # at least k distinct codes to return.
        fetch = max(k * 3, 15)
        merged = self.book.query(query_embedding, fetch)
        merged += self.precedent.query(query_embedding, fetch)

        best: dict[str, tuple[str, dict, float]] = {}
        for text, meta, score in merged:
            code = meta.get("code")
            if not code:
                continue
            if code not in best or score > best[code][2]:
                best[code] = (text, meta, score)
        ranked = sorted(best.values(), key=lambda h: h[2], reverse=True)
        return ranked[:k]

    def count(self) -> int:
        return self.book.count() + self.precedent.count()
