# ITAC AI backend — FastAPI + RAG (CPU inference).
#
# Build context is the REPO ROOT (see docker-compose.yml), not backend/, because
# the app resolves its repo root two/three levels above the `app` package and
# loads skills from `<repo_root>/skills`. We therefore recreate that layout inside
# the image under /srv:
#   /srv/backend/app/...   (config.py -> parents[2] == /srv)
#   /srv/skills/...        (loader.py -> parents[3] == /srv)
#   /srv/data/...          (mounted at runtime: chroma DB, reports, uploads, exports)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Embedding model cache (nomic-embed) lands here; backed by a named volume so
    # the ~500MB download survives container rebuilds.
    HF_HOME=/srv/hf-cache

# build-essential covers the rare dependency that has no manylinux wheel for 3.11;
# curl is used by the compose healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Install the CPU-only PyTorch wheel FIRST, from PyTorch's CPU index, so the later
# `pip install -e` (which pulls sentence-transformers) does not drag in the multi-
# gigabyte CUDA build. This is the biggest layer and rarely changes, so it caches
# well across code edits. Embedding at query time is a single short string, fast on CPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# App code + manifest + committed skills (skills contain no client data). The editable
# install must see the `app` package at build time for `import app` to resolve, so the
# code is copied before pip runs.
COPY backend/pyproject.toml /srv/backend/pyproject.toml
COPY backend/app /srv/backend/app
COPY skills /srv/skills
RUN pip install -e /srv/backend

EXPOSE 8000

# Editable install keeps `import app` pointing at /srv/backend/app, so __file__-based
# repo-root resolution still yields /srv. Run from there.
WORKDIR /srv/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
