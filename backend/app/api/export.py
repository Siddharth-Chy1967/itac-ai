"""Export download API (spec §7.3).

`GET /api/export/download/{token}` serves a previously rendered .xlsx from
`export_path`. Chat tables are saved as `{token}.xlsx` (see the orchestrator's
`file` event, whose URL points here); this endpoint hands that file back and
404s if the token has no file on disk.

`POST /api/export/xlsx` (the frontend "re-download" button, spec §7.3) is not
built yet; it will accept a raw TableSpec and render on demand.
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import get_settings

router = APIRouter()

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Download tokens are `secrets.token_urlsafe` output: URL-safe base64, so no dots
# or slashes. Rejecting anything else stops crafted values (path traversal,
# absolute paths) before they ever reach the filesystem.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@router.get("/export/download/{token}")
async def download_export(token: str) -> FileResponse:
    if not _TOKEN_RE.match(token):
        raise HTTPException(status_code=404, detail="Unknown export token")

    export_dir = Path(get_settings().export_path).resolve()
    path = (export_dir / f"{token}.xlsx").resolve()

    # Defense in depth: the resolved path must still sit inside export_dir.
    if export_dir != path.parent or not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown export token")

    return FileResponse(path, media_type=XLSX_MEDIA_TYPE, filename=f"{token}.xlsx")
