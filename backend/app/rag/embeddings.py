"""Sentence-transformers embedding wrapper (spec §6.3).

Loads the model named in config once (lazily, on construction) and encodes text
to L2-normalized vectors so cosine similarity in the vector store reduces to a
dot product. Uses CUDA when a GPU is available (falling back to CPU otherwise),
which matters for ingesting the full corpus. The heavy import is kept inside
`__init__` so importing this module (e.g. from the ingest CLI before it knows
there is work to do) does not drag in torch.

The default model, nomic-embed-text-v1.5, is an asymmetric retrieval model: it
expects a task prefix on every input ("search_document: " for stored chunks,
"search_query: " for queries) and covers up to 8192 tokens, so full ~700-token
chunks embed without truncation. Callers must therefore encode via
`encode_documents` (ingest) vs `encode_query` (retrieval) rather than a single
generic method; the prefixes are blank for models that do not use them.
"""
from __future__ import annotations

import logging

from ..config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 64

# Task prefixes nomic-embed-text expects for asymmetric retrieval.
_NOMIC_DOC_PREFIX = "search_document: "
_NOMIC_QUERY_PREFIX = "search_query: "


class Embedder:
    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ):
        import torch  # lazy: heavy import
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or get_settings().embedding_model
        self.batch_size = batch_size
        # Prefer the GPU when present; caller can force a device for testing.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading embedding model %s on %s", self.model_name, self.device)
        # trust_remote_code lets nomic load its custom model class; harmless for
        # models (e.g. all-MiniLM) that ship no remote code.
        self.model = SentenceTransformer(
            self.model_name, device=self.device, trust_remote_code=True
        )

        # Run in half precision on CUDA. On this hardware (RTX 4070, WDDM, GPU shared
        # with the ollama chat model) the GPU stays pinned at its idle clock, where
        # fp32 embedding is memory-bandwidth-bound and painfully slow (~5 texts/sec).
        # fp16 uses the tensor cores and is ~20x faster even at the idle clock, with
        # negligible effect on normalized-cosine ranking. CPU keeps fp32 (fp16 ops
        # are not well supported there).
        if self.device.startswith("cuda"):
            self.model = self.model.half()

        uses_nomic_prefixes = "nomic" in self.model_name.lower()
        self._doc_prefix = _NOMIC_DOC_PREFIX if uses_nomic_prefixes else ""
        self._query_prefix = _NOMIC_QUERY_PREFIX if uses_nomic_prefixes else ""

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode chunk texts for storage (applies the document task prefix)."""
        return self._encode([self._doc_prefix + t for t in texts])

    def encode_query(self, text: str) -> list[float]:
        """Encode a single search query (applies the query task prefix)."""
        return self._encode([self._query_prefix + text])[0]

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        """Encode several short query-side texts in one batch (query task prefix).

        For comparing a query against other short phrases on the same side of the
        asymmetric model, e.g. a field-data topic against candidate AR titles, where
        a document prefix on the titles would skew the scores.
        """
        return self._encode([self._query_prefix + t for t in texts])

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())
