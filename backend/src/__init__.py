"""
DTW Speech Recognition Package (NumPy only)
DTW 기반 음성 인식 패키지 - PyTorch 의존성 제거
"""

__version__ = "1.0.0-numpy"
__author__ = "DTW Project Team"

# 주요 클래스 임포트
from .feature_extraction import MFCCExtractor, SpectrogramExtractor, FeatureAugmenter
from .dtw_algorithm import DTWAlgorithm, FastDTW, ConstrainedDTW
from .speech_recognizer import SpeechRecognizer, EnsembleRecognizer, OnlineRecognizer
from .data_processing import SyntheticSpeechGenerator, AudioPreprocessor, DatasetManager
from .visualization import DTWVisualizer, FeatureVisualizer, ResultVisualizer
from .evaluation import Evaluator, Benchmarker, MetricCalculator

__all__ = [
    # Feature Extraction
    'MFCCExtractor',
    'SpectrogramExtractor',
    'FeatureAugmenter',
    
    # DTW Algorithm
    'DTWAlgorithm',
    'FastDTW',
    'ConstrainedDTW',
    
    # Speech Recognition
    'SpeechRecognizer',
    'EnsembleRecognizer',
    'OnlineRecognizer',
    
    # Data Processing
    'SyntheticSpeechGenerator',
    'AudioPreprocessor',
    'DatasetManager',
    
    # Visualization
    'DTWVisualizer',
    'FeatureVisualizer',
    'ResultVisualizer',
    
    # Evaluation
    'Evaluator',
    'Benchmarker',
    'MetricCalculator',
]
