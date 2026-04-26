"""Singleton wrapper around SpeechRecognizer with a SQLite-backed store."""

from __future__ import annotations

import io
import logging
import pickle
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

BACKEND_ROOT = Path(__file__).resolve().parents[2]
SRC = BACKEND_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from feature_extraction import MFCCExtractor  # noqa: E402
from dtw_algorithm import DTWAlgorithm  # noqa: E402
from speech_recognizer import SpeechRecognizer  # noqa: E402
from backends import is_accel_available  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.services.template_store import TemplateStore  # noqa: E402

log = logging.getLogger("dtw.service")


class TemplateNotFound(LookupError):
    pass


class RecognizerService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._extractor = MFCCExtractor(sr=settings.sample_rate, n_mfcc=settings.n_mfcc)
        self._dtw = DTWAlgorithm(backend=settings.dtw_backend, band=settings.dtw_band)
        self._recognizer = SpeechRecognizer(
            self._extractor,
            self._dtw,
            normalize=settings.normalize,
            score_aggregation=settings.score_aggregation,
            knn_k=settings.knn_k,
        )
        self._store = TemplateStore(settings.store_db_path, settings.templates_dir)
        # Per-label list of template ids, parallel to _recognizer.templates[label]
        self._ids: dict[str, list[str]] = {}

        self._maybe_migrate_pickle()
        self._reload_all()

    # ---------- bootstrap ----------

    def _maybe_migrate_pickle(self) -> None:
        pkl = settings.templates_path
        if not pkl.exists() or self._store.has_any():
            return
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            log.warning("pickle migration skipped (%s): %s", pkl, exc)
            return
        templates = data.get("templates", {}) if isinstance(data, dict) else {}
        moved = 0
        for label, feats_list in templates.items():
            for feats in feats_list:
                arr = np.asarray(feats, dtype=np.float32)
                if arr.ndim != 2:
                    continue
                self._store.add(str(label), arr)
                moved += 1
        bak = pkl.with_suffix(pkl.suffix + ".bak")
        try:
            pkl.rename(bak)
        except Exception:
            log.warning("could not rename %s -> %s", pkl, bak)
        log.info("migrated %d templates from pickle to store", moved)

    def _reload_all(self) -> None:
        self._recognizer.templates.clear()
        self._recognizer.template_metadata.clear()
        self._ids.clear()
        for tid, label, feats in self._store.iter_all():
            self._recognizer.templates.setdefault(label, []).append(feats)
            self._recognizer.template_metadata.setdefault(label, []).append({})
            self._ids.setdefault(label, []).append(tid)

    # ---------- meta ----------

    @property
    def backend_name(self) -> str:
        return self._dtw.backend_name if hasattr(self._dtw, "backend_name") else settings.dtw_backend

    # ---------- mutations ----------

    def add_template(self, label: str, audio_bytes: bytes) -> tuple[str, int]:
        signal = self._decode(audio_bytes)
        features = self._extractor.extract_from_array(signal)
        features = np.asarray(features, dtype=np.float32)
        with self._lock:
            tid = self._store.add(label, features)
            self._recognizer.templates.setdefault(label, []).append(features)
            self._recognizer.template_metadata.setdefault(label, []).append({})
            self._ids.setdefault(label, []).append(tid)
            return tid, len(self._recognizer.templates[label])

    def remove_label(self, label: str) -> int:
        with self._lock:
            n = self._store.delete_label(label)
            self._recognizer.templates.pop(label, None)
            self._recognizer.template_metadata.pop(label, None)
            self._ids.pop(label, None)
            return n

    def remove_template(self, label: str, template_id: str) -> None:
        with self._lock:
            ids = self._ids.get(label, [])
            try:
                idx = ids.index(template_id)
            except ValueError:
                raise TemplateNotFound(f"{label}/{template_id}")
            ok = self._store.delete_id(template_id)
            if not ok:
                raise TemplateNotFound(f"{label}/{template_id}")
            ids.pop(idx)
            self._recognizer.templates[label].pop(idx)
            self._recognizer.template_metadata[label].pop(idx)
            if not ids:
                self._recognizer.templates.pop(label, None)
                self._recognizer.template_metadata.pop(label, None)
                self._ids.pop(label, None)

    # ---------- reads ----------

    def list_templates(self) -> dict[str, int]:
        with self._lock:
            return self._store.list_counts()

    def list_label_ids(self, label: str) -> list[str]:
        with self._lock:
            return list(self._ids.get(label, []))

    def recognize(self, audio_bytes: bytes, top_k: int = 3) -> tuple[str, float, list[tuple[str, float]]]:
        signal = self._decode(audio_bytes)
        with self._lock:
            label, distance, _scores, top = self._recognizer.recognize_from_array(
                signal, return_scores=True, top_k=top_k
            )
        return str(label), float(distance), [(str(l), float(d)) for l, d in top]

    def has_templates(self) -> bool:
        with self._lock:
            return any(self._recognizer.templates.values())

    def snapshot(self) -> Path:
        """Bundle the SQLite store *and* the .npy blob directory into a single
        tar.gz. The DB alone is incomplete because feature blobs live on the
        filesystem; restoring from a DB-only snapshot leaves orphan rows."""
        ts = time.strftime("%Y%m%dT%H%M%S")
        dest = settings.backup_dir / f"snapshot-{ts}.tar.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with tempfile.TemporaryDirectory() as tmpdir:
                db_temp = Path(tmpdir) / "store.db"
                self._store.snapshot(db_temp)
                with tarfile.open(dest, "w:gz") as tar:
                    tar.add(db_temp, arcname="store.db")
                    if settings.templates_dir.exists():
                        for npy in sorted(settings.templates_dir.glob("*.npy")):
                            tar.add(npy, arcname=f"templates/{npy.name}")
        return dest

    # ---------- internals ----------

    def _decode(self, audio_bytes: bytes) -> np.ndarray:
        try:
            signal, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        except sf.LibsndfileError:
            # m4a/mp4/aac etc. — librosa.load needs a path to engage audioread+ffmpeg
            import librosa
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                signal, sr = librosa.load(tmp.name, sr=settings.sample_rate, mono=True)
            return signal.astype(np.float32, copy=False)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        if sr != settings.sample_rate:
            import librosa
            signal = librosa.resample(signal, orig_sr=sr, target_sr=settings.sample_rate)
        return signal.astype(np.float32, copy=False)


_service: RecognizerService | None = None


def get_recognizer_service() -> RecognizerService:
    global _service
    if _service is None:
        _service = RecognizerService()
    return _service


def accel_available() -> bool:
    return is_accel_available()
