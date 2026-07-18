"""Tests for the export download endpoint (spec §7.3).

`GET /api/export/download/{token}` serves `export_path/{token}.xlsx`, 404s for an
unknown token, and refuses tokens that could escape the export directory.
"""
from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import export
from app.main import create_app


def _client(monkeypatch, tmp_path) -> TestClient:
    monkeypatch.setattr(export, "get_settings", lambda: SimpleNamespace(export_path=str(tmp_path)))
    return TestClient(create_app())


def test_download_serves_existing_file(monkeypatch, tmp_path):
    (tmp_path / "abc123.xlsx").write_bytes(b"PK\x03\x04fake-xlsx-bytes")
    client = _client(monkeypatch, tmp_path)

    resp = client.get("/api/export/download/abc123")

    assert resp.status_code == 200
    assert resp.content == b"PK\x03\x04fake-xlsx-bytes"
    assert "spreadsheetml" in resp.headers["content-type"]


def test_download_unknown_token_returns_404(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)

    resp = client.get("/api/export/download/doesnotexist")

    assert resp.status_code == 404


def test_download_rejects_traversal_token(monkeypatch, tmp_path):
    # A token with dots/slashes is rejected outright, never touching the filesystem.
    client = _client(monkeypatch, tmp_path)

    resp = client.get("/api/export/download/..%2f..%2fsecret")

    assert resp.status_code == 404
