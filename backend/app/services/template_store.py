"""SQLite-backed template store. Features stored as individual .npy files.

Schema:
    templates(
        id TEXT PRIMARY KEY,            # uuid4 hex
        label TEXT NOT NULL,
        blob_path TEXT NOT NULL,        # relative to templates_dir
        frames INTEGER NOT NULL,
        n_mfcc INTEGER NOT NULL,
        created_at REAL NOT NULL
    )

WAL mode for concurrent readers; writes serialized at the service layer.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

log = logging.getLogger("dtw.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS templates (
    id          TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    blob_path   TEXT NOT NULL,
    frames      INTEGER NOT NULL,
    n_mfcc      INTEGER NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_templates_label ON templates(label);
"""


class TemplateStore:
    def __init__(self, db_path: Path, blob_dir: Path) -> None:
        self.db_path = Path(db_path)
        self.blob_dir = Path(blob_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    # ---------- writes ----------

    def add(self, label: str, features: np.ndarray) -> str:
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D (frames, n_mfcc); got shape {features.shape}")
        tid = uuid.uuid4().hex
        rel = f"{tid}.npy"
        path = self.blob_dir / rel
        np.save(path, features.astype(np.float32, copy=False))
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO templates(id,label,blob_path,frames,n_mfcc,created_at) VALUES (?,?,?,?,?,?)",
                    (tid, label, rel, int(features.shape[0]), int(features.shape[1]), time.time()),
                )
        except Exception:
            # roll back the orphan blob if INSERT failed
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            raise
        return tid

    def delete_id(self, template_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT blob_path FROM templates WHERE id=?", (template_id,)
            ).fetchone()
            if row is None:
                return False
            self._conn.execute("DELETE FROM templates WHERE id=?", (template_id,))
        try:
            (self.blob_dir / row[0]).unlink(missing_ok=True)
        except Exception:
            log.warning("orphan blob not removed: %s", row[0])
        return True

    def delete_label(self, label: str) -> int:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, blob_path FROM templates WHERE label=?", (label,)
            ).fetchall()
            self._conn.execute("DELETE FROM templates WHERE label=?", (label,))
        for _id, rel in rows:
            try:
                (self.blob_dir / rel).unlink(missing_ok=True)
            except Exception:
                log.warning("orphan blob not removed: %s", rel)
        return len(rows)

    # ---------- reads ----------

    def has_any(self) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM templates LIMIT 1").fetchone() is not None

    def list_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT label, COUNT(*) FROM templates GROUP BY label ORDER BY label"
            ).fetchall()
        return {label: count for label, count in rows}

    def list_label_ids(self, label: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM templates WHERE label=? ORDER BY created_at, id",
                (label,),
            ).fetchall()
        return [r[0] for r in rows]

    def iter_label(self, label: str) -> Iterator[tuple[str, np.ndarray]]:
        for tid in self.list_label_ids(label):
            yield tid, self._load_blob_for(tid)

    def iter_all(self) -> Iterator[tuple[str, str, np.ndarray]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, label, blob_path FROM templates ORDER BY label, created_at, id"
            ).fetchall()
        for tid, label, rel in rows:
            try:
                yield tid, label, np.load(self.blob_dir / rel)
            except Exception as exc:
                log.warning("skip unreadable blob %s for %s: %s", rel, tid, exc)

    def _load_blob_for(self, template_id: str) -> np.ndarray:
        with self._lock:
            row = self._conn.execute(
                "SELECT blob_path FROM templates WHERE id=?", (template_id,)
            ).fetchone()
        if row is None:
            raise KeyError(template_id)
        return np.load(self.blob_dir / row[0])

    # ---------- ops ----------

    def snapshot(self, dest: Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            target = sqlite3.connect(dest)
            try:
                self._conn.backup(target)
            finally:
                target.close()
        return dest

    def import_pairs(self, pairs: Iterable[tuple[str, np.ndarray]]) -> int:
        n = 0
        for label, feats in pairs:
            self.add(label, feats)
            n += 1
        return n

    def close(self) -> None:
        with self._lock:
            self._conn.close()
