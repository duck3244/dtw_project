"""End-to-end tests for the FastAPI surface, with isolated per-test storage."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


def _wav_bytes(seconds: float = 1.0, sr: int = 16000) -> bytes:
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(int(seconds * sr)).astype(np.float32) * 0.05
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate persistence to tmp_path before importing app
    from app.core import config as cfg

    monkeypatch.setattr(cfg.settings, "store_db_path", tmp_path / "store.db")
    monkeypatch.setattr(cfg.settings, "templates_dir", tmp_path / "templates")
    monkeypatch.setattr(cfg.settings, "backup_dir", tmp_path / "backups")
    monkeypatch.setattr(cfg.settings, "templates_path", tmp_path / "templates.pkl")  # absent → no migration

    # Reset singleton so the service binds to the patched paths
    from app.services import recognizer_service as svc_mod

    monkeypatch.setattr(svc_mod, "_service", None)

    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["accel_available"], bool)
    assert body["sample_rate"] == 16000


def test_empty_listing(client):
    r = client.get("/api/templates")
    assert r.status_code == 200
    assert r.json() == {"labels": [], "total": 0}


def test_recognize_without_templates_returns_409(client):
    r = client.post(
        "/api/recognize",
        files={"file": ("z.wav", _wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 409


def test_register_recognize_delete_label(client):
    payload = _wav_bytes()
    # register
    r = client.post(
        "/api/templates",
        files={"file": ("noise.wav", payload, "audio/wav")},
        data={"label": "noise"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["label"] == "noise"
    assert body["count"] == 1
    assert isinstance(body["template_id"], str) and len(body["template_id"]) == 32

    # listing reflects it
    r = client.get("/api/templates")
    assert r.json() == {"labels": [{"label": "noise", "count": 1}], "total": 1}

    # detail
    r = client.get("/api/templates/noise")
    assert r.status_code == 200
    assert r.json()["template_ids"] == [body["template_id"]]

    # recognize finds it
    r = client.post("/api/recognize", files={"file": ("noise.wav", payload, "audio/wav")})
    assert r.status_code == 200
    assert r.json()["label"] == "noise"

    # delete by label
    r = client.delete("/api/templates/noise")
    assert r.status_code == 204

    r = client.get("/api/templates")
    assert r.json() == {"labels": [], "total": 0}


def test_delete_by_template_id(client):
    payload = _wav_bytes()
    a = client.post(
        "/api/templates",
        files={"file": ("a.wav", payload, "audio/wav")},
        data={"label": "x"},
    ).json()
    b = client.post(
        "/api/templates",
        files={"file": ("b.wav", payload, "audio/wav")},
        data={"label": "x"},
    ).json()
    assert client.get("/api/templates").json()["total"] == 2

    r = client.delete(f"/api/templates/x/{a['template_id']}")
    assert r.status_code == 204

    detail = client.get("/api/templates/x").json()
    assert detail["template_ids"] == [b["template_id"]]

    # missing id -> 404
    r = client.delete(f"/api/templates/x/{a['template_id']}")
    assert r.status_code == 404


def test_unsupported_extension_returns_415(client):
    r = client.post(
        "/api/templates",
        files={"file": ("evil.exe", b"MZ\x00\x00", "application/octet-stream")},
        data={"label": "bad"},
    )
    assert r.status_code == 415


def test_oversize_returns_413(client):
    # ~12 MB > default 10 MB cap
    payload = b"\x00" * (12 * 1024 * 1024)
    r = client.post(
        "/api/templates",
        files={"file": ("big.wav", payload, "audio/wav")},
        data={"label": "big"},
    )
    assert r.status_code == 413


def test_label_detail_404_for_unknown(client):
    r = client.get("/api/templates/does-not-exist")
    assert r.status_code == 404


def test_evaluate_409_when_no_templates(client):
    r = client.post(
        "/api/evaluate",
        files=[("files", ("a.wav", _wav_bytes(), "audio/wav"))],
        data={"expected": ["x"]},
    )
    assert r.status_code == 409


def test_evaluate_metrics(client):
    # register two distinct templates
    rng = np.random.default_rng(42)
    sig_a = rng.standard_normal(16000).astype(np.float32) * 0.05
    sig_b = rng.standard_normal(16000).astype(np.float32) * 0.05 + 0.5

    def to_wav(sig):
        buf = io.BytesIO()
        sf.write(buf, sig, 16000, format="WAV", subtype="FLOAT")
        return buf.getvalue()

    client.post(
        "/api/templates",
        files={"file": ("a.wav", to_wav(sig_a), "audio/wav")},
        data={"label": "a"},
    )
    client.post(
        "/api/templates",
        files={"file": ("b.wav", to_wav(sig_b), "audio/wav")},
        data={"label": "b"},
    )

    # evaluation set: 2 a's, 2 b's, plus one "unknown" that uses sig_a (will predict 'a')
    payload = [
        ("files", ("a1.wav", to_wav(sig_a), "audio/wav")),
        ("files", ("a2.wav", to_wav(sig_a), "audio/wav")),
        ("files", ("b1.wav", to_wav(sig_b), "audio/wav")),
        ("files", ("b2.wav", to_wav(sig_b), "audio/wav")),
        ("files", ("unknown.wav", to_wav(sig_a), "audio/wav")),
    ]
    r = client.post(
        "/api/evaluate",
        files=payload,
        data={"expected": ["a", "a", "b", "b", ""]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["n_total"] == 5
    assert body["n_scored"] == 4  # the empty-expected is excluded from accuracy
    assert body["n_correct"] == 4
    assert body["accuracy"] == 1.0
    assert body["avg_latency_ms"] > 0
    labels = {s["label"]: s for s in body["per_label"]}
    assert labels["a"]["tp"] == 2 and labels["a"]["precision"] == 1.0 and labels["a"]["recall"] == 1.0
    assert labels["b"]["tp"] == 2
    assert all(c["latency_ms"] > 0 for c in body["cases"])


def test_evaluate_length_mismatch_400(client):
    client.post(
        "/api/templates",
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"label": "a"},
    )
    r = client.post(
        "/api/evaluate",
        files=[
            ("files", ("a1.wav", _wav_bytes(), "audio/wav")),
            ("files", ("a2.wav", _wav_bytes(), "audio/wav")),
        ],
        data={"expected": ["a"]},  # only one label for two files
    )
    assert r.status_code == 400


def test_snapshot_bundles_db_and_blobs(client):
    # need at least one row to make the snapshot non-trivial
    client.post(
        "/api/templates",
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
        data={"label": "x"},
    )
    r = client.post("/api/admin/snapshot")
    assert r.status_code == 200
    path = r.json()["path"]
    assert path.endswith(".tar.gz")

    import tarfile
    from pathlib import Path

    archive = Path(path)
    assert archive.exists() and archive.stat().st_size > 0
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    assert "store.db" in names
    blobs = [n for n in names if n.startswith("templates/") and n.endswith(".npy")]
    assert len(blobs) >= 1, f"expected at least one .npy blob in snapshot, got {names}"
