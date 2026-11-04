"""Local, offline inference backend: the only module that knows how pages reach a model.

Replaces the OpenAI Files API + Responses API. Nothing is uploaded and no API key exists.

    PDF page ──PyMuPDF──► text layer (digital PDF)  ─┐
                     └──► 300 DPI render ─► RapidOCR ┴─► transcript (exact characters)
    PDF page ──PyMuPDF──► image sized to a VRAM budget ─► layout, labels, redactions
    instructions + images + transcripts ─► Ollama /api/chat (Qwen2.5-VL, JSON schema)
    ─► json ─► the caller's existing Pydantic model (validated exactly as before)

Why a hybrid: a 4-bit 7B vision model reads layout well but can misread or invent digits
(account numbers, kWh, dollar amounts), while OCR reads characters accurately but has no
notion of which amount belongs to which label. The model gets both and is told to take
structure from the image and characters from the transcript. OCR runs on the CPU so the
whole GPU is available to the vision model.

Callers use `LocalInferenceClient.parse_pdf(...)` / `parse_text(...)`; the prompts and the
Pydantic models stay in the modules that own them. Settings come from environment
variables or the project's `.env` (see `.env.example`).
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence, TypeVar

from pydantic import BaseModel, ValidationError

DEFAULT_MODEL = "qwen2.5vl:latest"
# Same threshold `bill_extraction.inspect_pdf_pages` uses to call a page text-based.
TEXT_LAYER_MIN_CHARS = 50
# Qwen2.5-VL: 14 px patches merged 2x2, so one visual token per 28x28 px.
_PIXELS_PER_VISUAL_TOKEN = 28 * 28
_MIN_IMAGE_SIDE = 640
_CHARS_PER_TEXT_TOKEN = 3.2  # conservative for numbers-heavy bill text

Model = TypeVar("Model", bound=BaseModel)
logger = logging.getLogger(__name__)

# Appended to every existing instruction block. The original prompts were written for
# uploaded PDFs; this explains the local input format without changing their rules.
LOCAL_INPUT_RULES = """
Input format for this run:
- Each supplied PDF page is provided as a rendered page image, in order.
- The user message also contains a transcript for each page, taken from the PDF's text
  layer or from OCR of that same page image.
- Use the images for layout: headings, which value belongs to which label, table rows
  and columns, and whether content is present, obscured, or redacted.
- Use the transcripts for exact characters. When a value visible in an image also
  appears in its transcript, copy it character-for-character from the transcript.
- OCR can misread characters, merge neighbouring columns, or produce text for redacted
  or blacked-out areas. If a transcript and its image disagree, trust what is clearly
  legible in the image; if neither is clearly legible, treat the value as unreadable.
- Respond with one JSON object that follows the required JSON schema exactly. Do not
  add markdown, code fences, or any text outside the JSON object.
"""


# --- configuration -------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {raw!r}") from None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


def normalize_host(host: str) -> str:
    """Accept Ollama's own OLLAMA_HOST forms ("127.0.0.1:11434", "0.0.0.0")."""
    host = (host or "").strip().rstrip("/") or "http://localhost:11434"
    if not re.match(r"^https?://", host):
        host = "http://" + host
    scheme, rest = host.split("://", 1)
    if rest.startswith("0.0.0.0"):  # a bind address, not a destination
        rest = "127.0.0.1" + rest[len("0.0.0.0"):]
    if not re.search(r":\d+$", rest):
        rest += ":11434"
    return f"{scheme}://{rest}"


def cache_label(model: str) -> str:
    """Folder name for this model's caches: "qwen2.5vl:latest" -> "ollama-qwen2.5vl-latest".

    Must satisfy `bill_value_extraction.cache_path_for` ([A-Za-z0-9][A-Za-z0-9._-]*). The
    "ollama-" prefix keeps local results visibly separate from the existing OpenAI
    caches (gpt-5.6-luna / gpt-5.6-terra), which are still read first.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-._")
    if not slug:
        raise ValueError(f"Invalid model name: {model!r}")
    return f"ollama-{slug}"


@dataclass(frozen=True)
class LocalInferenceConfig:
    host: str = "http://localhost:11434"
    model: str = DEFAULT_MODEL
    text_model: str = ""  # "" -> use `model`, so only one model occupies VRAM
    num_ctx: int = 16384
    max_num_ctx: int = 32768
    timeout_seconds: float = 900.0
    keep_alive: str = "15m"
    ocr_enabled: bool = True
    ocr_dpi: int = 300
    ocr_min_score: float = 0.5
    image_max_side: int = 1344
    image_token_budget: int = 6000
    max_attempts: int = 2

    @classmethod
    def from_env(cls, project_folder: str | Path | None = None) -> "LocalInferenceConfig":
        """Read settings from the environment, after loading `<project>/.env` if present."""
        if project_folder is not None:
            env_file = Path(project_folder) / ".env"
            if env_file.is_file():
                from dotenv import load_dotenv

                load_dotenv(env_file)
        base = cls()
        config = cls(
            # OLLAMA_HOST is the extractor's own name for it; OLLAMA_BASE_URL is
            # itac-ai's. Accept either so one .env drives both.
            host=normalize_host(
                os.environ.get("OLLAMA_HOST")
                or os.environ.get("OLLAMA_BASE_URL")
                or base.host
            ),
            model=os.environ.get("LOCAL_VLM_MODEL", "").strip() or base.model,
            text_model=os.environ.get("LOCAL_TEXT_MODEL", "").strip(),
            num_ctx=_env_int("OLLAMA_NUM_CTX", base.num_ctx),
            max_num_ctx=_env_int("OLLAMA_MAX_NUM_CTX", base.max_num_ctx),
            timeout_seconds=_env_float("OLLAMA_TIMEOUT_SECONDS", base.timeout_seconds),
            keep_alive=os.environ.get("OLLAMA_KEEP_ALIVE", "").strip() or base.keep_alive,
            ocr_enabled=_env_bool("LOCAL_OCR", base.ocr_enabled),
            ocr_dpi=_env_int("OCR_DPI", base.ocr_dpi),
            ocr_min_score=_env_float("OCR_MIN_SCORE", base.ocr_min_score),
            image_max_side=_env_int("VLM_IMAGE_MAX_SIDE", base.image_max_side),
            image_token_budget=_env_int("VLM_IMAGE_TOKEN_BUDGET", base.image_token_budget),
            max_attempts=max(1, _env_int("LOCAL_MAX_ATTEMPTS", base.max_attempts)),
        )
        cache_label(config.model)  # fail early on an unusable model name
        return config

    @classmethod
    def from_settings(cls, settings) -> "LocalInferenceConfig":
        """Build from itac-ai's `Settings`, the app's single source of truth.

        `from_env` stays for standalone/CLI use; inside the app nothing reads
        os.environ directly (config.py contract).
        """
        config = cls(
            host=normalize_host(settings.ollama_base_url),
            model=settings.bills_vlm_model,
            text_model=settings.bills_text_model,
            num_ctx=settings.bills_num_ctx,
            max_num_ctx=settings.bills_max_num_ctx,
            timeout_seconds=settings.bills_timeout_seconds,
            keep_alive=settings.bills_keep_alive,
            ocr_enabled=settings.bills_ocr,
            ocr_dpi=settings.bills_ocr_dpi,
            ocr_min_score=settings.bills_ocr_min_score,
            image_max_side=settings.bills_image_max_side,
            image_token_budget=settings.bills_image_token_budget,
            max_attempts=max(1, settings.bills_max_attempts),
        )
        cache_label(config.model)  # fail early on an unusable model name
        return config

    @property
    def cache_label(self) -> str:
        return cache_label(self.model)

    def describe(self) -> str:
        ocr = f"RapidOCR at {self.ocr_dpi} DPI" if self.ocr_enabled else "no OCR"
        return f"{self.model} via Ollama at {self.host} ({ocr})"


# --- rendering and OCR ------------------------------------------------------------

@dataclass
class PageMaterial:
    """One page as the model sees it."""

    label: str
    image_png: bytes
    image_tokens: int
    transcript: str
    transcript_source: str  # "PDF text layer" | "OCR" | "none"


def image_side_for(page_count: int, config: LocalInferenceConfig, aspect: float = 11 / 8.5) -> int:
    """Longest image side (px) so all pages together stay near the visual-token budget."""
    per_page = config.image_token_budget / max(page_count, 1)
    side = int(math.sqrt(per_page * _PIXELS_PER_VISUAL_TOKEN * max(aspect, 1.0)))
    return max(_MIN_IMAGE_SIDE, min(config.image_max_side, side))


# One recognised text box: (top, bottom, left, right, text), in page pixels.
OcrItem = tuple[float, float, float, float, str]


def ocr_items(
    boxes: Sequence, texts: Sequence[str], scores: Sequence[float], min_score: float
) -> list[OcrItem]:
    """The confident boxes, reduced to their extent and text."""
    items = []
    for box, text, score in zip(boxes, texts, scores):
        text = (text or "").strip()
        if not text or float(score) < min_score:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        items.append((min(ys), max(ys), min(xs), max(xs), text))
    return items


def ocr_lines(items: Sequence[OcrItem]) -> list[list[OcrItem]]:
    """Group boxes into printed lines, each ordered left to right.

    Boxes share a line when their vertical centres are within half a typical text
    height; the line's centre is re-averaged as boxes join so a slightly skewed
    scan still holds together.
    """
    if not items:
        return []
    heights = sorted(bottom - top for top, bottom, *_ in items)
    line_height = max(heights[len(heights) // 2], 1.0)
    lines: list[dict] = []
    for item in sorted(items, key=lambda i: (i[0] + i[1]) / 2):
        centre = (item[0] + item[1]) / 2
        if lines and abs(centre - lines[-1]["centre"]) <= 0.5 * line_height:
            line = lines[-1]
            line["items"].append(item)
            line["centre"] = sum((i[0] + i[1]) / 2 for i in line["items"]) / len(line["items"])
        else:
            lines.append({"centre": centre, "items": [item]})
    return [sorted(line["items"], key=lambda i: i[2]) for line in lines]


def assemble_ocr_text(
    boxes: Sequence, texts: Sequence[str], scores: Sequence[float], min_score: float
) -> str:
    """Rebuild reading-order lines from OCR boxes.

    A wide horizontal gap becomes a run of spaces so label/value columns
    ("Current Charges      $1,234.56") stay apart.
    """
    return text_from_items(ocr_items(boxes, texts, scores, min_score))


def text_from_items(items: Sequence[OcrItem]) -> str:
    if not items:
        return ""
    char_widths = sorted((x1 - x0) / max(len(t), 1) for _t0, _t1, x0, x1, t in items)
    char_width = max(char_widths[len(char_widths) // 2], 1.0)
    out = []
    for line in ocr_lines(items):
        parts, previous_end = [], None
        for _top, _bottom, x0, x1, text in line:
            if previous_end is not None:
                gap = (x0 - previous_end) / char_width
                parts.append("    " if gap >= 2.5 else " ")
            parts.append(text)
            previous_end = x1
        out.append("".join(parts))
    return "\n".join(out)


_ocr_engine = None
_OCR_CACHE: "OrderedDict[str, list[OcrItem]]" = OrderedDict()
_OCR_CACHE_SIZE = 256


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        for name in ("RapidOCR", "rapidocr"):
            logging.getLogger(name).setLevel(logging.WARNING)
        try:
            from rapidocr import RapidOCR
        except ImportError as error:
            raise RuntimeError(
                "RapidOCR is not installed. Run: python -m pip install -r requirements.txt "
                "(or set LOCAL_OCR=false to rely on the vision model alone)."
            ) from error
        _ocr_engine = RapidOCR()
        for name in ("RapidOCR", "rapidocr"):
            logging.getLogger(name).setLevel(logging.WARNING)
    return _ocr_engine


def ocr_page(page, config: LocalInferenceConfig) -> str:
    """OCR one PyMuPDF page at `ocr_dpi`, as reading-order text, chart axes removed."""
    from .page_text import split_axis_labels

    return text_from_items(split_axis_labels(ocr_page_items(page, config))[0])


def ocr_page_items(page, config: LocalInferenceConfig) -> list[OcrItem]:
    """OCR one page, keeping where every box sits. Identical page images are OCR'd once.

    The text transcript and the table reader (`grid`) both work from these, so a
    page read for extraction is not read again to rebuild its tables.
    """
    import numpy as np
    import pymupdf

    pixmap = page.get_pixmap(dpi=config.ocr_dpi, colorspace=pymupdf.csRGB, alpha=False)
    key = sha256(pixmap.samples).hexdigest() + f":{config.ocr_min_score}"
    if key in _OCR_CACHE:
        _OCR_CACHE.move_to_end(key)
        return _OCR_CACHE[key]
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, 3
    )[:, :, ::-1]  # RGB -> BGR, the OpenCV order RapidOCR expects
    result = _get_ocr_engine()(np.ascontiguousarray(image))
    boxes = getattr(result, "boxes", None)
    items = [] if boxes is None else ocr_items(
        boxes, result.txts, result.scores, config.ocr_min_score
    )
    _OCR_CACHE[key] = items
    if len(_OCR_CACHE) > _OCR_CACHE_SIZE:
        _OCR_CACHE.popitem(last=False)
    return items


def load_pdf_pages(
    pdf_paths: Sequence[Path], config: LocalInferenceConfig, labels: Sequence[str] | None = None
) -> list[PageMaterial]:
    """Render every page of the given PDFs into image + transcript, in order."""
    import pymupdf

    paths = [Path(p) for p in pdf_paths]
    if labels is not None and len(labels) != len(paths):
        raise ValueError("labels must match pdf_paths one-to-one")
    documents = [pymupdf.open(path) for path in paths]
    try:
        total = sum(doc.page_count for doc in documents)
        if total == 0:
            raise ValueError("The supplied PDF has no pages.")
        side = image_side_for(total, config)
        pages: list[PageMaterial] = []
        for doc_index, doc in enumerate(documents):
            for page in doc:
                if labels is not None:
                    label = labels[doc_index]
                    if doc.page_count > 1:
                        label += f", page {page.number + 1}"
                else:
                    label = f"Page {len(pages) + 1}"
                text = page.get_text("text", sort=True).strip()
                if len(text) >= TEXT_LAYER_MIN_CHARS:
                    # Chart axis labels are printed but are not bill figures; the
                    # model never sees them (see page_text).
                    from .page_text import split_axis_labels, text_layer_items

                    transcript = text_from_items(split_axis_labels(text_layer_items(page))[0])
                    source = "PDF text layer"
                elif config.ocr_enabled:
                    transcript, source = ocr_page(page, config), "OCR"
                else:
                    transcript, source = text, "PDF text layer" if text else "none"
                rect = page.rect
                scale = side / max(rect.width, rect.height, 1)
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csRGB, alpha=False
                )
                tokens = math.ceil(pixmap.width / 28) * math.ceil(pixmap.height / 28)
                pages.append(
                    PageMaterial(label, pixmap.tobytes("png"), tokens, transcript, source)
                )
        return pages
    finally:
        for doc in documents:
            doc.close()


# --- Ollama transport -----------------------------------------------------------

def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_error_text(error: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(error.read().decode("utf-8"))
        return str(body.get("error") or body)
    except Exception:  # noqa: BLE001 - best-effort diagnostics only
        return str(error)


# --- client ----------------------------------------------------------------------

class LocalInferenceClient:
    """Structured inference against a local Ollama server. Safe to create; no I/O until used."""

    def __init__(
        self,
        config: LocalInferenceConfig | None = None,
        *,
        project_folder: str | Path | None = None,
    ):
        self.config = config or LocalInferenceConfig.from_env(project_folder)
        self._ready_models: set[str] = set()

    @property
    def cache_label(self) -> str:
        return self.config.cache_label

    def describe(self) -> str:
        return self.config.describe()

    def close(self) -> None:
        """Nothing to release: Ollama unloads the model after `keep_alive`."""

    # -- readiness --

    def ensure_ready(self, model: str | None = None, *, vision: bool = True) -> None:
        """Fail fast, with the fix, when Ollama is down or the model is missing."""
        model = model or self.config.model
        if model in self._ready_models:
            return
        host = self.config.host
        try:
            tags = _get_json(f"{host}/api/tags", timeout=10)
        except (urllib.error.URLError, OSError) as error:
            raise RuntimeError(
                f"Cannot reach Ollama at {host} ({error}). Start Ollama (open the Ollama app "
                "or run `ollama serve`), or point OLLAMA_BASE_URL in .env at it."
            ) from error
        installed = {m.get("name", "") for m in tags.get("models", [])}
        wanted = model if ":" in model else f"{model}:latest"
        if model not in installed and wanted not in installed:
            raise RuntimeError(
                f"Ollama model {model!r} is not installed. Run: ollama pull {model}"
            )
        if vision:
            try:
                info = _post_json(f"{host}/api/show", {"model": model}, timeout=30)
            except (urllib.error.URLError, OSError):
                info = {}
            capabilities = info.get("capabilities")
            if capabilities is not None and "vision" not in capabilities:
                raise RuntimeError(
                    f"Ollama model {model!r} cannot read images (capabilities: "
                    f"{capabilities}). Set LOCAL_VLM_MODEL to a vision model such as "
                    f"{DEFAULT_MODEL}."
                )
        self._ready_models.add(model)

    # -- public parsing API --

    def parse_pdf(
        self,
        output_model: type[Model],
        instructions: str,
        prompt: str,
        pdf_paths: Sequence[Path],
        *,
        labels: Sequence[str] | None = None,
        max_output_tokens: int = 2048,
    ) -> Model:
        """Every page of `pdf_paths` as image + transcript -> validated `output_model`."""
        self.ensure_ready(self.config.model, vision=True)
        pages = load_pdf_pages(pdf_paths, self.config, labels)
        sections = [prompt.strip()]
        for number, page in enumerate(pages, start=1):
            transcript = page.transcript.strip() or "(no text could be read from this page)"
            sections.append(
                f"=== {page.label}: image {number} of {len(pages)}; "
                f"transcript from {page.transcript_source} ===\n{transcript}"
            )
        user_text = "\n\n".join(sections)
        images = [base64.b64encode(page.image_png).decode("ascii") for page in pages]
        visual_tokens = sum(page.image_tokens for page in pages)
        return self._chat(
            self.config.model, output_model, instructions, user_text, images,
            visual_tokens=visual_tokens, max_output_tokens=max_output_tokens,
        )

    def parse_text(
        self,
        output_model: type[Model],
        instructions: str,
        text: str,
        *,
        max_output_tokens: int = 2048,
    ) -> Model:
        """Text-only request (a page's PDF text layer) -> validated `output_model`."""
        model = self.config.text_model or self.config.model
        self.ensure_ready(model, vision=False)
        return self._chat(
            model, output_model, instructions, text, [],
            visual_tokens=0, max_output_tokens=max_output_tokens,
        )

    # -- internals --

    def _context_size(self, prompt_chars: int, visual_tokens: int, max_output_tokens: int) -> int:
        needed = int(prompt_chars / _CHARS_PER_TEXT_TOKEN) + visual_tokens + max_output_tokens + 256
        if needed <= self.config.num_ctx:
            return self.config.num_ctx
        rounded = int(math.ceil(needed / 4096) * 4096)
        if rounded > self.config.max_num_ctx:
            raise RuntimeError(
                f"Request needs about {needed} tokens of context, above OLLAMA_MAX_NUM_CTX="
                f"{self.config.max_num_ctx}. Raise OLLAMA_MAX_NUM_CTX or lower "
                "VLM_IMAGE_TOKEN_BUDGET in .env."
            )
        return rounded

    def _chat(
        self,
        model: str,
        output_model: type[Model],
        instructions: str,
        user_text: str,
        images: list[str],
        *,
        visual_tokens: int,
        max_output_tokens: int,
    ) -> Model:
        schema = output_model.model_json_schema()
        system = (
            instructions.strip()
            + "\n\n" + LOCAL_INPUT_RULES.strip()
            + "\n\nRequired JSON schema:\n" + json.dumps(schema, separators=(",", ":"))
        )
        user_message: dict = {"role": "user", "content": user_text}
        if images:
            user_message["images"] = images
        messages = [{"role": "system", "content": system}, user_message]
        num_ctx = self._context_size(len(system) + len(user_text), visual_tokens, max_output_tokens)

        last_error = "no attempt made"
        for attempt in range(1, self.config.max_attempts + 1):
            payload = {
                "model": model,
                "messages": messages,
                "format": schema,
                "stream": False,
                "keep_alive": self.config.keep_alive,
                "options": {
                    "temperature": 0,
                    "seed": 0,
                    "num_ctx": num_ctx,
                    "num_predict": max_output_tokens,
                },
            }
            try:
                response = _post_json(
                    f"{self.config.host}/api/chat", payload, self.config.timeout_seconds
                )
            except urllib.error.HTTPError as error:
                detail = _http_error_text(error)
                if error.code == 404 or "not found" in detail.lower():
                    raise RuntimeError(
                        f"Ollama model {model!r} is not installed. Run: ollama pull {model}"
                    ) from error
                raise RuntimeError(f"Ollama request failed ({error.code}): {detail}") from error
            except (urllib.error.URLError, OSError) as error:
                raise RuntimeError(
                    f"Ollama at {self.config.host} did not answer ({error}). Check that "
                    "Ollama is running; a large page on a busy GPU may need a higher "
                    "OLLAMA_TIMEOUT_SECONDS."
                ) from error

            content = (response.get("message") or {}).get("content", "")
            try:
                if response.get("done_reason") == "length":
                    raise ValueError(
                        f"output was cut off at {max_output_tokens} tokens before the JSON ended"
                    )
                return output_model.model_validate(json.loads(_strip_fences(content)))
            except (ValueError, ValidationError) as error:  # JSONDecodeError is a ValueError
                last_error = _short_error(error)
                logger.warning("Local model attempt %s returned invalid output: %s", attempt, last_error)
                messages = messages[:2] + [
                    {"role": "assistant", "content": content[:4000]},
                    {
                        "role": "user",
                        "content": (
                            "That reply was not a valid JSON object for the required schema: "
                            f"{last_error}. Return the complete corrected JSON object only."
                        ),
                    },
                ]
                if "cut off" in last_error:
                    max_output_tokens = min(max_output_tokens * 2, 16384)
                    num_ctx = self._context_size(
                        len(system) + len(user_text), visual_tokens, max_output_tokens
                    )
        raise RuntimeError(
            f"The local model did not return valid {output_model.__name__} JSON after "
            f"{self.config.max_attempts} attempts: {last_error}"
        )


def _strip_fences(content: str) -> str:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return fenced.group(1) if fenced else text


def _short_error(error: Exception) -> str:
    text = str(error).replace("\n", " ")
    return text if len(text) <= 600 else text[:597] + "..."
