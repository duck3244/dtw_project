"""
core vs accel 백엔드 정확도 동등성 및 속도 벤치마크.

실행:
    python examples/benchmark_backends.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backends import get_backend, is_accel_available  # noqa: E402

REPEAT = 30
WARMUP = 2
SHAPES: List[Tuple[int, int]] = [
    (50, 60),
    (100, 120),
    (200, 250),
    (400, 500),
    (800, 1000),
]
METRICS = ["euclidean", "manhattan", "cosine"]
BANDS: List[Optional[int]] = [None, 50, 100]
DIM = 13


def _time_call(fn, *args, **kwargs) -> float:
    for _ in range(WARMUP):
        fn(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        fn(*args, **kwargs)
    return (time.perf_counter() - t0) / REPEAT * 1000.0  # ms


def synthetic_benchmark() -> None:
    if not is_accel_available():
        print("accel backend not available — install requirements-accel.txt")
        return
    core = get_backend("core")
    accel = get_backend("accel")

    print("=" * 90)
    print("DTW Backend Benchmark — synthetic sequences (D=13)")
    print("=" * 90)
    header = (
        f"{'shape':<14s} {'metric':<10s} {'band':<6s} "
        f"{'core(ms)':>10s} {'accel(ms)':>10s} {'speedup':>9s} {'eq':>4s}"
    )
    print(header)
    print("-" * 90)

    rng = np.random.default_rng(0)
    n_total = 0
    n_eq = 0
    for N, M in SHAPES:
        x = rng.standard_normal((N, DIM)).astype(np.float32)
        y = rng.standard_normal((M, DIM)).astype(np.float32)
        for metric in METRICS:
            for band in BANDS:
                # band가 시퀀스 길이 차보다 작으면 양쪽 모두 inf — 가독성 위해 표시만
                d_c, _, _ = core.dtw(x, y, metric=metric, band=band)
                d_a, _, _ = accel.dtw(x, y, metric=metric, band=band)
                eq = (
                    np.isclose(d_c, d_a, rtol=1e-4, atol=1e-3)
                    or (np.isinf(d_c) and np.isinf(d_a))
                )
                n_total += 1
                n_eq += int(eq)

                t_c = _time_call(core.dtw, x, y, metric=metric, band=band)
                t_a = _time_call(accel.dtw, x, y, metric=metric, band=band)
                speedup = t_c / t_a if t_a > 0 else float("inf")
                tag = "OK" if eq else "FAIL"
                print(
                    f"({N:>3d},{M:>4d})  {metric:<10s} {str(band):<6s} "
                    f"{t_c:>9.3f}  {t_a:>9.3f}  {speedup:>8.1f}x  {tag:>4s}"
                )
    print("-" * 90)
    print(f"equivalence: {n_eq}/{n_total} cases match\n")


def real_data_throughput() -> None:
    """Mini Speech Commands에서 backend별 인식 throughput 비교."""
    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", "mini_speech_commands"
    )
    if not os.path.isdir(data_dir):
        print("[skip] Mini Speech Commands data not present at", data_dir)
        return

    import glob
    import random

    from dtw_algorithm import DTWAlgorithm
    from feature_extraction import MFCCExtractor
    from speech_recognizer import SpeechRecognizer

    # 균일한 split (eval_mini_speech_commands.py와 동일 시드)
    classes = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    rnd = random.Random(0)
    n_tpl, n_test = 10, 50
    train_files, test_files = {}, {}
    for cls in classes:
        files = sorted(glob.glob(os.path.join(data_dir, cls, "*.wav")))
        rnd.shuffle(files)
        train_files[cls] = files[:n_tpl]
        test_files[cls] = files[n_tpl : n_tpl + n_test]

    extractor = MFCCExtractor(sr=16000, n_mfcc=13, include_deltas=True)
    print("=" * 90)
    print("Mini Speech Commands — backend throughput (policy: min + norm)")
    print("=" * 90)

    # MFCC 한 번만 캐시
    train_feats = {
        c: [extractor.extract(f) for f in fs] for c, fs in train_files.items()
    }
    test_feats = [
        (cls, extractor.extract(f))
        for cls, fs in test_files.items()
        for f in fs
    ]

    print(f"{'backend':<8s} {'accuracy':>10s} {'eval_time':>12s} {'per_call(ms)':>14s}")
    print("-" * 50)
    for backend in ("core", "accel"):
        dtw = DTWAlgorithm(backend=backend)
        rec = SpeechRecognizer(
            extractor, dtw, normalize=True, score_aggregation="min"
        )
        for cls, fl in train_feats.items():
            for feats in fl:
                rec.templates.setdefault(cls, []).append(feats)
                rec.template_metadata.setdefault(cls, []).append({})
        # warmup (numba JIT compile)
        rec._recognize_from_features(test_feats[0][1], False, 1)

        correct = 0
        t0 = time.perf_counter()
        for cls, feats in test_feats:
            pred, _ = rec._recognize_from_features(feats, False, 1)
            correct += int(pred == cls)
        elapsed = time.perf_counter() - t0
        n_calls = len(test_feats) * sum(len(v) for v in rec.templates.values())
        print(
            f"{backend:<8s} {correct / len(test_feats) * 100:>9.2f}% "
            f"{elapsed:>10.2f}s  {elapsed / n_calls * 1000:>12.4f}"
        )


def main() -> None:
    synthetic_benchmark()
    real_data_throughput()


if __name__ == "__main__":
    main()
