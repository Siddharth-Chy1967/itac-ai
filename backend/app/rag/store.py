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
