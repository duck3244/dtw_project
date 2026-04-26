"""
Mini Speech Commands 데이터셋으로 인식기 스코어 정책을 비교 평가합니다.

데이터: data/mini_speech_commands/{down,go,left,no,right,stop,up,yes}/*.wav
        (한 클래스당 wav 1000개, 16kHz, 약 1초)

실행:
    python examples/eval_mini_speech_commands.py
"""

from __future__ import annotations

import glob
import os
import random
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from dtw_algorithm import DTWAlgorithm  # noqa: E402
from feature_extraction import MFCCExtractor  # noqa: E402
from speech_recognizer import SpeechRecognizer  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "mini_speech_commands")
N_TEMPLATE = 10
N_TEST = 50
SEED = 0


def split_files(seed: int = SEED) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    classes = sorted(
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    rnd = random.Random(seed)
    train_files: Dict[str, List[str]] = {}
    test_files: Dict[str, List[str]] = {}
    for cls in classes:
        files = sorted(glob.glob(os.path.join(DATA_DIR, cls, "*.wav")))
        rnd.shuffle(files)
        train_files[cls] = files[:N_TEMPLATE]
        test_files[cls] = files[N_TEMPLATE : N_TEMPLATE + N_TEST]
    return train_files, test_files


def precompute_features(
    extractor: MFCCExtractor, files_by_class: Dict[str, List[str]]
) -> Dict[str, List[np.ndarray]]:
    """파일을 한 번씩만 읽어 MFCC를 캐시 — 정책별 평가에서 재사용."""
    feats: Dict[str, List[np.ndarray]] = {}
    for cls, files in files_by_class.items():
        feats[cls] = [extractor.extract(f) for f in files]
    return feats


def evaluate_policy(
    train_feats: Dict[str, List[np.ndarray]],
    test_feats: Dict[str, List[np.ndarray]],
    extractor: MFCCExtractor,
    dtw: DTWAlgorithm,
    **policy_kwargs,
) -> tuple[float, float, float]:
    rec = SpeechRecognizer(extractor, dtw, **policy_kwargs)

    t = time.perf_counter()
    for cls, feats_list in train_feats.items():
        for feats in feats_list:
            # MFCC를 직접 주입 (재추출 회피)
            rec.templates.setdefault(cls, []).append(feats)
            rec.template_metadata.setdefault(cls, []).append({})
    train_t = time.perf_counter() - t

    correct = total = 0
    t = time.perf_counter()
    for cls, feats_list in test_feats.items():
        for feats in feats_list:
            pred, _ = rec._recognize_from_features(feats, return_scores=False, top_k=1)
            correct += int(pred == cls)
            total += 1
    test_t = time.perf_counter() - t
    return correct / total, train_t, test_t


def main() -> None:
    train_files, test_files = split_files()
    classes = list(train_files.keys())
    print(f"classes ({len(classes)}): {classes}")
    n_tpl = sum(len(v) for v in train_files.values())
    n_test = sum(len(v) for v in test_files.values())
    print(f"templates: {n_tpl} ({N_TEMPLATE}/class), tests: {n_test} ({N_TEST}/class)")

    extractor = MFCCExtractor(sr=16000, n_mfcc=13, include_deltas=True)
    dtw = DTWAlgorithm()  # auto -> accel when numba available
    print(f"DTW backend: {dtw.backend_name}\n")

    print("[MFCC 사전 추출]")
    t = time.perf_counter()
    train_feats = precompute_features(extractor, train_files)
    test_feats = precompute_features(extractor, test_files)
    # 평균 프레임 길이 — band 권장값 산정에 참고
    sample_lengths = [f.shape[0] for fl in train_feats.values() for f in fl]
    print(
        f"  cached {n_tpl + n_test} files in {time.perf_counter() - t:.2f}s "
        f"(MFCC frames: mean={np.mean(sample_lengths):.1f}, "
        f"max={np.max(sample_lengths)})\n"
    )

    # ===== 1) 정책 비교 (band=None) =====
    print("[A] 스코어 정책 비교 (band=None, 무제약 DTW)")
    print(f"{'policy':<25s} {'accuracy':>10s} {'eval_time':>12s}")
    print("-" * 52)
    configs = [
        ("legacy(mean, unnorm)", dict(normalize=False, score_aggregation="mean")),
        ("mean + norm",          dict(normalize=True,  score_aggregation="mean")),
        ("min + norm (default)", dict(normalize=True,  score_aggregation="min")),
        ("knn k=3 + norm",       dict(normalize=True,  score_aggregation="knn", knn_k=3)),
        ("knn k=5 + norm",       dict(normalize=True,  score_aggregation="knn", knn_k=5)),
    ]
    for name, kw in configs:
        acc, _train_t, test_t = evaluate_policy(
            train_feats, test_feats, extractor, dtw, **kw
        )
        print(f"{name:<25s} {acc * 100:>9.2f}% {test_t:>10.2f}s")

    # ===== 2) band 비교 (best 정책 = min+norm 기본값) =====
    print("\n[B] Sakoe-Chiba band 효과 (best 정책: min + norm)")
    print(f"{'band':<10s} {'accuracy':>10s} {'eval_time':>12s} {'note':<20s}")
    print("-" * 58)
    for band in [None, 30, 20, 15, 10, 5]:
        dtw_b = DTWAlgorithm(band=band)
        acc, _train_t, test_t = evaluate_policy(
            train_feats, test_feats, extractor, dtw_b,
            normalize=True, score_aggregation="min",
        )
        note = "unconstrained" if band is None else ""
        if band is not None and band <= 5:
            note = "may produce inf"
        print(f"{str(band):<10s} {acc * 100:>9.2f}% {test_t:>10.2f}s {note:<20s}")


if __name__ == "__main__":
    main()
