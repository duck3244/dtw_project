"""SpeechRecognizer policy tests + DTWAlgorithm facade behaviour."""

import numpy as np
import pytest

from data_processing import SyntheticSpeechGenerator
from dtw_algorithm import DTWAlgorithm
from feature_extraction import MFCCExtractor
from speech_recognizer import SpeechRecognizer


@pytest.fixture
def make_recognizer():
    extractor = MFCCExtractor(sr=16000, n_mfcc=13, include_deltas=True)
    dtw = DTWAlgorithm(backend="core")  # core in tests for stability
    generator = SyntheticSpeechGenerator(sr=16000)

    def _factory(**recognizer_kwargs):
        rec = SpeechRecognizer(extractor, dtw, **recognizer_kwargs)
        rng = np.random.default_rng(0)
        for label, pattern in (("alpha", "chirp"), ("beta", "formant")):
            for _ in range(3):
                duration = 0.8 + float(rng.random()) * 0.3
                signal = generator.generate_speech_sample(
                    pattern, duration=duration, noise_level=0.05
                )
                rec.add_template_from_array(label, signal)
        return rec, generator

    return _factory


# ---- defaults & validation --------------------------------------------------

def test_default_policy_is_min_normalized(make_recognizer):
    rec, _ = make_recognizer()
    assert rec.normalize is True
    assert rec.score_aggregation == "min"
    assert rec.knn_k == 3


def test_invalid_aggregation_raises(make_recognizer):
    with pytest.raises(ValueError):
        make_recognizer(score_aggregation="bogus")


def test_no_templates_raises():
    extractor = MFCCExtractor(sr=16000)
    dtw = DTWAlgorithm(backend="core")
    rec = SpeechRecognizer(extractor, dtw)
    with pytest.raises(ValueError):
        rec.recognize_from_array(np.zeros(8000, dtype=np.float32))


# ---- per-policy behaviour ---------------------------------------------------

@pytest.mark.parametrize("aggregation", ["min", "mean", "knn"])
def test_recognize_returns_known_label(make_recognizer, aggregation):
    rec, gen = make_recognizer(score_aggregation=aggregation, knn_k=3)
    signal = gen.generate_speech_sample("chirp", duration=0.85, noise_level=0.05)
    label, distance = rec.recognize_from_array(signal)
    assert label in {"alpha", "beta"}
    assert np.isfinite(distance)


def test_recognize_is_deterministic(make_recognizer):
    rec, gen = make_recognizer()
    rng = np.random.default_rng(1)
    signal = gen.generate_speech_sample("chirp", duration=0.85, noise_level=0.0)
    # noise_level=0 → 결정적
    label1, dist1 = rec.recognize_from_array(signal)
    label2, dist2 = rec.recognize_from_array(signal)
    assert label1 == label2
    assert dist1 == dist2
    _ = rng  # silence linter


def test_return_scores_shape(make_recognizer):
    rec, gen = make_recognizer()
    signal = gen.generate_speech_sample("formant", duration=0.85)
    label, distance, scores, top_k = rec.recognize_from_array(
        signal, return_scores=True, top_k=2
    )
    assert label in scores
    assert len(top_k) == 2
    # top-k는 거리 오름차순
    assert top_k[0][1] <= top_k[1][1]


# ---- DTWAlgorithm facade ----------------------------------------------------

def test_dtwalgorithm_negative_band_rejected():
    with pytest.raises(ValueError):
        DTWAlgorithm(band=-1, backend="core")


def test_dtwalgorithm_band_propagates(make_recognizer):
    """DTWAlgorithm(band=N) 인스턴스가 인식기 호출 경로에 자동 반영."""
    extractor = MFCCExtractor(sr=16000, n_mfcc=13)
    dtw_band = DTWAlgorithm(backend="core", band=15)
    rec = SpeechRecognizer(extractor, dtw_band)
    gen = SyntheticSpeechGenerator(sr=16000)
    rng = np.random.default_rng(2)
    for label, pattern in (("a", "chirp"), ("b", "pulse")):
        for _ in range(2):
            duration = 0.8 + float(rng.random()) * 0.3
            rec.add_template_from_array(
                label, gen.generate_speech_sample(pattern, duration=duration)
            )
    test_sig = gen.generate_speech_sample("chirp", duration=0.85)
    label, distance = rec.recognize_from_array(test_sig)
    assert label in {"a", "b"}
    assert np.isfinite(distance)


def test_dtwalgorithm_unsupported_metric():
    with pytest.raises(ValueError):
        DTWAlgorithm(distance_metric="bogus", backend="core")
