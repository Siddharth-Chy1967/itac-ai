"""Section-aware chunking (spec §6.2).

Reports have a consistent skeleton, so we chunk by section rather than with a
blind sliding window: each section (an AR block, an Executive Summary, etc.) is
one chunk when it fits the token budget, and only oversized sections are split
on paragraph boundaries with overlap. The section heading is prepended to every
chunk so a retrieved fragment still says what it is.

Token counts are the config's `chunk_size_tokens` / `chunk_overlap_tokens`;
`_approx_tokens` estimates tokens as chars/4 (a standard heuristic) to avoid
pulling in a tokenizer here.

Every chunk carries the metadata a retrieval answer needs to cite its source:
report_id, fiscal_year, era, section heading, ar_number, and chunk_index.
"""
from __future__ import annotations

from dataclasses import dataclass

from .extract import ExtractedDoc

_CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    text: str
    metadata: dict


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _pack(paragraphs: list[str], size_tokens: int, overlap_tokens: int) -> list[str]:
    """Pack paragraphs into windows of ~size_tokens with ~overlap_tokens carried over.

    A single paragraph larger than the budget becomes its own (oversized) window
    rather than being dropped or mid-word split.
    """
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        tokens = _approx_tokens(para)
        if current and current_tokens + tokens > size_tokens:
            windows.append("\n".join(current))
            # Seed the next window with a trailing overlap of whole paragraphs.
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(current):
                if tail and tail_tokens + _approx_tokens(prev) > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_tokens += _approx_tokens(prev)
            current = list(tail)
            current_tokens = tail_tokens
        current.append(para)
        current_tokens += tokens

    if current:
        windows.append("\n".join(current))
    return windows


def chunk_document(
    doc: ExtractedDoc,
    base_metadata: dict,
    *,
    size_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Chunk an ExtractedDoc, stamping `base_metadata` (report_id/fiscal_year/era) on each chunk."""
    chunks: list[Chunk] = []
    index = 0

    for section in doc.sections:
        if not section.text:
            continue
        heading = section.heading
        has_heading = bool(heading) and heading != "(preamble)"
        for body in _pack(section.paragraphs, size_tokens, overlap_tokens):
            text = f"{heading}\n{body}".strip() if has_heading else body
            metadata = {
                **base_metadata,
                "section": heading,
                "ar_number": section.ar_number or "",
                "chunk_index": index,
            }
            chunks.append(Chunk(text=text, metadata=metadata))
            index += 1

    return chunks
