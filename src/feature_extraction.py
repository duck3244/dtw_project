"""
Feature Extraction Module
음성 신호에서 특징을 추출하는 모듈
"""

import numpy as np
import librosa
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class MFCCExtractor:
    """
    MFCC (Mel-Frequency Cepstral Coefficients) 특징 추출기
    
    음성 신호를 MFCC 특징으로 변환합니다.
    Delta 및 Delta-Delta 특징도 함께 추출할 수 있습니다.
    """
    
    def __init__(
        self,
        sr: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 40,
        include_deltas: bool = True
    ):
        """
        Args:
            sr: 샘플링 레이트 (Hz)
            n_mfcc: MFCC 계수 개수
            n_fft: FFT 윈도우 크기
            hop_length: 홉 길이 (프레임 간격)
            n_mels: Mel 필터뱅크 개수
            include_deltas: Delta 및 Delta-Delta 특징 포함 여부
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.include_deltas = include_deltas
    
    def extract(self, audio_path: str) -> np.ndarray:
        """
        오디오 파일에서 MFCC 특징 추출
        
        Args:
            audio_path: 오디오 파일 경로
        
        Returns:
            features: MFCC 특징 배열 (T x D)
                T: 시간 프레임 수
                D: 특징 차원 (n_mfcc * 3 if include_deltas else n_mfcc)
        """
        # 오디오 로드
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        return self.extract_from_array(y)
    
    def extract_from_array(self, y: np.ndarray) -> np.ndarray:
        """
        numpy 배열에서 MFCC 특징 추출
        
        Args:
            y: 오디오 신호 배열
        
        Returns:
            features: MFCC 특징 배열 (T x D)
        """
        # MFCC 추출
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        if self.include_deltas:
            # Delta (1차 미분) 특징 추가
            delta_mfcc = librosa.feature.delta(mfcc)
            # Delta-Delta (2차 미분) 특징 추가
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            # 모든 특징 결합
            features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
        else:
            features = mfcc
        
        # NumPy 배열로 변환 (T x D)
        features = features.T.astype(np.float32)
        
        return features
    
    def extract_batch(self, audio_paths: list) -> List[np.ndarray]:
        """
        여러 오디오 파일에서 배치로 특징 추출
        
        Args:
            audio_paths: 오디오 파일 경로 리스트
        
        Returns:
            features_list: 특징 배열 리스트
        """
        features_list = []
        for path in audio_paths:
            features = self.extract(path)
            features_list.append(features)
        
        return features_list
    
    def normalize(self, features: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        특징 정규화
        
        Args:
            features: 입력 특징 (T x D)
            method: 정규화 방법 ('zscore', 'minmax')
        
        Returns:
            normalized_features: 정규화된 특징
        """
        if method == 'zscore':
            # Z-score 정규화
            mean = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True)
            normalized = (features - mean) / (std + 1e-8)
        
        elif method == 'minmax':
            # Min-Max 정규화
            min_val = features.min(axis=0, keepdims=True)
            max_val = features.max(axis=0, keepdims=True)
            normalized = (features - min_val) / (max_val - min_val + 1e-8)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def get_feature_dimension(self) -> int:
        """특징 벡터의 차원 반환"""
        if self.include_deltas:
            return self.n_mfcc * 3
        else:
            return self.n_mfcc


class SpectrogramExtractor:
    """
    스펙트로그램 특징 추출기
    
    멜 스펙트로그램을 추출합니다.
    """
    
    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128
    ):
        """
        Args:
            sr: 샘플링 레이트
            n_fft: FFT 윈도우 크기
            hop_length: 홉 길이
            n_mels: Mel 필터뱅크 개수
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def extract(self, audio_path: str) -> np.ndarray:
        """오디오 파일에서 멜 스펙트로그램 추출"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        return self.extract_from_array(y)
    
    def extract_from_array(self, y: np.ndarray) -> np.ndarray:
        """numpy 배열에서 멜 스펙트로그램 추출"""
        # 멜 스펙트로그램 계산
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # dB 스케일로 변환
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # NumPy 배열로 변환 (T x D)
        features = mel_spec_db.T.astype(np.float32)
        
        return features


class FeatureAugmenter:
    """
    특징 증강기
    
    데이터 증강을 통해 학습 데이터를 확장합니다.
    """
    
    @staticmethod
    def add_noise(features: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
        """
        특징에 가우시안 노이즈 추가
        
        Args:
            features: 입력 특징
            noise_level: 노이즈 레벨 (표준편차)
        
        Returns:
            noisy_features: 노이즈가 추가된 특징
        """
        noise = np.random.randn(*features.shape) * noise_level
        return features + noise
    
    @staticmethod
    def time_stretch(features: np.ndarray, rate: float = 1.1) -> np.ndarray:
        """
        시간 축 늘이기/줄이기
        
        Args:
            features: 입력 특징 (T x D)
            rate: 늘림/줄임 비율 (>1: 늘림, <1: 줄임)
        
        Returns:
            stretched_features: 변환된 특징
        """
        # 선형 보간을 사용한 리샘플링
        original_length = features.shape[0]
        new_length = int(original_length * rate)
        
        # 각 차원에 대해 보간
        indices = np.linspace(0, original_length - 1, new_length)
        stretched = np.zeros((new_length, features.shape[1]))
        
        for d in range(features.shape[1]):
            stretched[:, d] = np.interp(
                indices,
                np.arange(original_length),
                features[:, d]
            )
        
        return stretched.astype(np.float32)
    
    @staticmethod
    def frequency_mask(features: np.ndarray, num_masks: int = 1, 
                      mask_size: int = 5) -> np.ndarray:
        """
        주파수 마스킹 (SpecAugment)
        
        Args:
            features: 입력 특징 (T x D)
            num_masks: 마스크 개수
            mask_size: 마스크 크기
        
        Returns:
            masked_features: 마스킹된 특징
        """
        features = features.copy()
        D = features.shape[1]
        
        for _ in range(num_masks):
            f = np.random.randint(0, mask_size)
            f0 = np.random.randint(0, D - f)
            features[:, f0:f0+f] = 0
        
        return features


if __name__ == "__main__":
    print("Feature Extraction Module (NumPy only)")
    print("=" * 60)
    
    # 사용 예제
    print("\n[사용 예제]")
    print("""
    # MFCC 추출
    extractor = MFCCExtractor(sr=16000, n_mfcc=13)
    features = extractor.extract('audio.wav')
    
    # 정규화
    normalized = extractor.normalize(features, method='zscore')
    
    # 데이터 증강
    augmenter = FeatureAugmenter()
    noisy = augmenter.add_noise(features, noise_level=0.01)
    stretched = augmenter.time_stretch(features, rate=1.1)
    """)
