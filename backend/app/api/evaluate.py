"""Batch accuracy evaluation: upload labeled audio files, get metrics back."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.responses import EvaluateResponse, EvaluationCase, LabelStats
from app.services.recognizer_service import RecognizerService, get_recognizer_service

router = APIRouter(tags=["evaluate"])


def _suffix_ok(filename: str) -> bool:
    suffix = "." + (filename or "").rsplit(".", 1)[-1].lower()
    return suffix in settings.allowed_audio_suffixes


def _summarize(cases: list[EvaluationCase]) -> tuple[int, int, list[LabelStats]]:
    """Compute n_correct + per-label precision/recall/F1 over scored cases."""
    scored = [c for c in cases if c.expected is not None and c.predicted is not None]
    n_correct = sum(1 for c in scored if c.correct)

    labels: set[str] = set()
    for c in scored:
        if c.expected:
            labels.add(c.expected)
        if c.predicted:
            labels.add(c.predicted)

    stats: list[LabelStats] = []
    for label in sorted(labels):
        tp = sum(1 for c in scored if c.expected == label and c.predicted == label)
        fp = sum(1 for c in scored if c.expected != label and c.predicted == label)
        fn = sum(1 for c in scored if c.expected == label and c.predicted != label)
        support = sum(1 for c in scored if c.expected == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        stats.append(
            LabelStats(
                label=label,
                support=support,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return n_correct, len(scored), stats


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    files: list[UploadFile] = File(...),
    expected: list[str] | None = Form(None),
    svc: RecognizerService = Depends(get_recognizer_service),
) -> EvaluateResponse:
    expected = expected or []
    if not svc.has_templates():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "No templates registered — register at least one before evaluating.",
        )
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "files is empty")
    if len(files) > settings.max_evaluate_files:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"too many files (max {settings.max_evaluate_files})",
        )
    if expected and len(expected) != len(files):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"expected length {len(expected)} != files length {len(files)}",
        )

    cases: list[EvaluationCase] = []
    for i, f in enumerate(files):
        exp = (expected[i].strip() if expected else "") or None
        filename = f.filename or f"file-{i}"
        t0 = time.perf_counter()

        if not _suffix_ok(filename):
            cases.append(
                EvaluationCase(
                    filename=filename,
                    expected=exp,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    error=f"unsupported suffix",
                )
            )
            continue

        try:
            payload = await f.read()
            if len(payload) > settings.max_audio_bytes:
                raise ValueError(f"file exceeds {settings.max_audio_bytes} bytes")
            predicted, distance, _top = svc.recognize(payload, top_k=1)
            elapsed = (time.perf_counter() - t0) * 1000
            cases.append(
                EvaluationCase(
                    filename=filename,
                    expected=exp,
                    predicted=predicted,
                    distance=distance,
                    latency_ms=elapsed,
                    correct=(predicted == exp) if exp is not None else None,
                )
            )
        except Exception as exc:
            cases.append(
                EvaluationCase(
                    filename=filename,
                    expected=exp,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    error=str(exc),
                )
            )

    n_correct, n_scored, per_label = _summarize(cases)
    accuracy = n_correct / n_scored if n_scored else 0.0
    avg_latency = sum(c.latency_ms for c in cases) / len(cases) if cases else 0.0

    return EvaluateResponse(
        n_total=len(cases),
        n_scored=n_scored,
        n_correct=n_correct,
        accuracy=accuracy,
        avg_latency_ms=avg_latency,
        per_label=per_label,
        cases=cases,
    )
