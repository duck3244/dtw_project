#!/usr/bin/env bash
# Single-worker invariant: the recognizer holds in-memory templates and a
# SQLite store. Multiple workers would have inconsistent in-memory views and
# would race on writes. For horizontal scaling, move the store out of process
# (e.g. shared Postgres) and re-evaluate. For now: one worker.

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-dev}"

case "$MODE" in
  dev)
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ;;
  prod)
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
    ;;
  *)
    echo "usage: $0 [dev|prod]" >&2
    exit 2
    ;;
esac
