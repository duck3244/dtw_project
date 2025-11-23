"""
Data Processing Module
음성 데이터 생성, 전처리 및 관리
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List, Optional, Dict
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')


class SyntheticSpeechGenerator:
    """
    합성 음성 생성기
    
    테스트 및 데모를 위한 합성 음성 신호를 생성합니다.
    """
    
    def __init__(self, sr: int = 16000):
        """
        Args:
            sr: 샘플링 레이트
        """
        self.sr = sr
    
    def generate_sine_wave(
        self,
        frequency: float = 440.0,
        duration: float = 1.0,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        사인파 생성
        
        Args:
            frequency: 주파수 (Hz)
            duration: 지속 시간 (초)
            amplitude: 진폭
        
        Returns:
            signal: 오디오 신호
        """
        t = np.linspace(0, duration, int(self.sr * duration))
        signal = amplitude * np.sin(2 * np.pi * frequency * t)
        return signal.astype(np.float32)
    
    def generate_chirp(
        self,
        f0: float = 200.0,
        f1: float = 800.0,
        duration: float = 1.0,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Chirp 신호 생성 (주파수 변화)
        
        Args:
            f0: 시작 주파수 (Hz)
            f1: 종료 주파수 (Hz)
            duration: 지속 시간
            amplitude: 진폭
        
        Returns:
            signal: Chirp 신호
        """
        t = np.linspace(0, duration, int(self.sr * duration))
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
        signal = amplitude * np.sin(phase)
        return signal.astype(np.float32)
    
    def generate_pulse_train(
        self,
        pulse_freq: float = 100.0,
        duration: float = 1.0,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        펄스 트레인 생성
        
        Args:
            pulse_freq: 펄스 주파수 (Hz)
            duration: 지속 시간
            amplitude: 진폭
        
        Returns:
            signal: 펄스 트레인
        """
        t = np.linspace(0, duration, int(self.sr * duration))
        signal = np.zeros_like(t)
        
        # 펄스 위치
        pulse_indices = np.arange(0, len(t), int(self.sr / pulse_freq))
        signal[pulse_indices] = amplitude
        
        # 저역 통과 필터로 부드럽게
        window_size = int(self.sr * 0.01)
        window = np.ones(window_size) / window_size
        signal = np.convolve(signal, window, mode='same')
        
        return signal.astype(np.float32)
    
    def generate_formant(
        self,
        f1: float = 700.0,
        f2: float = 1220.0,
        f3: float = 2600.0,
        duration: float = 1.0,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        포먼트 유사 신호 생성 (모음 근사)
        
        Args:
            f1, f2, f3: 포먼트 주파수들 (Hz)
            duration: 지속 시간
            amplitude: 진폭
        
        Returns:
            signal: 포먼트 신호
        """
        t = np.linspace(0, duration, int(self.sr * duration))
        
        # 여러 포먼트 결합
        signal = (
            amplitude * np.sin(2 * np.pi * f1 * t) +
            amplitude * 0.5 * np.sin(2 * np.pi * f2 * t) +
            amplitude * 0.25 * np.sin(2 * np.pi * f3 * t)
        )
        
        return signal.astype(np.float32)
    
    def apply_envelope(
        self,
        signal: np.ndarray,
        fade_ratio: float = 0.05
    ) -> np.ndarray:
        """
        엔벨로프 적용 (페이드 인/아웃)
        
        Args:
            signal: 입력 신호
            fade_ratio: 페이드 비율 (0~1)
        
        Returns:
            enveloped_signal: 엔벨로프가 적용된 신호
        """
        envelope = np.ones_like(signal)
        fade_len = int(len(signal) * fade_ratio)
        
        # 페이드 인
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        # 페이드 아웃
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        
        return signal * envelope
    
    def add_noise(
        self,
        signal: np.ndarray,
        noise_level: float = 0.05
    ) -> np.ndarray:
        """
        가우시안 노이즈 추가
        
        Args:
            signal: 입력 신호
            noise_level: 노이즈 레벨 (표준편차)
        
        Returns:
            noisy_signal: 노이즈가 추가된 신호
        """
        noise = np.random.randn(len(signal)) * noise_level
        return signal + noise
    
    def normalize(self, signal: np.ndarray) -> np.ndarray:
        """신호 정규화"""
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            return signal / max_val
        return signal
    
    def generate_speech_sample(
        self,
        pattern: str = 'chirp',
        duration: float = 1.0,
        noise_level: float = 0.05,
        normalize: bool = True
    ) -> np.ndarray:
        """
        완전한 음성 샘플 생성
        
        Args:
            pattern: 패턴 ('sine', 'chirp', 'pulse', 'formant')
            duration: 지속 시간
            noise_level: 노이즈 레벨
            normalize: 정규화 여부
        
        Returns:
            signal: 생성된 음성 신호
        """
        # 기본 신호 생성
        if pattern == 'sine':
            signal = self.generate_sine_wave(duration=duration)
        elif pattern == 'chirp':
            signal = self.generate_chirp(duration=duration)
        elif pattern == 'pulse':
            signal = self.generate_pulse_train(duration=duration)
        elif pattern == 'formant':
            signal = self.generate_formant(duration=duration)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        # 엔벨로프 적용
        signal = self.apply_envelope(signal)
        
        # 노이즈 추가
        if noise_level > 0:
            signal = self.add_noise(signal, noise_level)
        
        # 정규화
        if normalize:
            signal = self.normalize(signal)
        
        return signal


class AudioPreprocessor:
    """
    오디오 전처리기
    
    음성 신호를 전처리합니다.
    """
    
    def __init__(self, sr: int = 16000):
        """
        Args:
            sr: 샘플링 레이트
        """
        self.sr = sr
    
    def load_audio(
        self,
        audio_path: str,
        duration: Optional[float] = None,
        offset: float = 0.0
    ) -> np.ndarray:
        """
        오디오 파일 로드
        
        Args:
            audio_path: 오디오 파일 경로
            duration: 로드할 지속 시간 (None: 전체)
            offset: 시작 오프셋
        
        Returns:
            audio: 오디오 신호
        """
        audio, _ = librosa.load(
            audio_path,
            sr=self.sr,
            duration=duration,
            offset=offset
        )
        return audio
    
    def trim_silence(
        self,
        audio: np.ndarray,
        top_db: int = 20
    ) -> np.ndarray:
        """
        무음 구간 제거
        
        Args:
            audio: 오디오 신호
            top_db: 무음 판단 임계값 (dB)
        
        Returns:
            trimmed_audio: 무음이 제거된 오디오
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    
    def normalize_loudness(
        self,
        audio: np.ndarray,
        target_db: float = -20.0
    ) -> np.ndarray:
        """
        음량 정규화
        
        Args:
            audio: 오디오 신호
            target_db: 목표 dB 레벨
        
        Returns:
            normalized_audio: 정규화된 오디오
        """
        # RMS 계산
        rms = np.sqrt(np.mean(audio**2))
        
        # 목표 RMS 계산
        target_rms = 10**(target_db / 20)
        
        # 스케일링
        if rms > 0:
            normalized = audio * (target_rms / rms)
        else:
            normalized = audio
        
        return normalized
    
    def apply_preemphasis(
        self,
        audio: np.ndarray,
        coef: float = 0.97
    ) -> np.ndarray:
        """
        프리엠퍼시스 필터 적용
        
        고주파 성분을 강조합니다.
        
        Args:
            audio: 오디오 신호
            coef: 프리엠퍼시스 계수
        
        Returns:
            emphasized_audio: 필터 적용된 오디오
        """
        emphasized = np.append(audio[0], audio[1:] - coef * audio[:-1])
        return emphasized
    
    def resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        리샘플링
        
        Args:
            audio: 오디오 신호
            orig_sr: 원본 샘플링 레이트
            target_sr: 목표 샘플링 레이트
        
        Returns:
            resampled_audio: 리샘플링된 오디오
        """
        resampled = librosa.resample(
            audio,
            orig_sr=orig_sr,
            target_sr=target_sr
        )
        return resampled
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sr: Optional[int] = None
    ):
        """
        오디오 저장
        
        Args:
            audio: 오디오 신호
            output_path: 출력 파일 경로
            sr: 샘플링 레이트 (None: self.sr 사용)
        """
        if sr is None:
            sr = self.sr
        
        sf.write(output_path, audio, sr)
        print(f"✓ Audio saved to {output_path}")


class DatasetManager:
    """
    데이터셋 관리자
    
    음성 데이터셋을 관리하고 로드합니다.
    """
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: 데이터 디렉토리 경로
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def create_synthetic_dataset(
        self,
        classes: List[Tuple[str, str]],
        samples_per_class: int = 5,
        sr: int = 16000
    ):
        """
        합성 데이터셋 생성
        
        Args:
            classes: [(class_name, pattern), ...]
            samples_per_class: 클래스당 샘플 수
            sr: 샘플링 레이트
        """
        generator = SyntheticSpeechGenerator(sr=sr)
        
        for class_name, pattern in classes:
            class_dir = self.data_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            for i in range(samples_per_class):
                # 다양성을 위해 약간씩 다른 파라미터
                duration = 0.8 + np.random.rand() * 0.4  # 0.8~1.2초
                noise_level = 0.02 + np.random.rand() * 0.03  # 0.02~0.05
                
                signal = generator.generate_speech_sample(
                    pattern=pattern,
                    duration=duration,
                    noise_level=noise_level
                )
                
                # 저장
                output_path = class_dir / f"{class_name}_{i:03d}.wav"
                sf.write(output_path, signal, sr)
            
            print(f"✓ Created {samples_per_class} samples for '{class_name}'")
    
    def load_dataset(self) -> Dict[str, List[str]]:
        """
        데이터셋 로드
        
        Returns:
            dataset: {class_name: [file_paths, ...]}
        """
        dataset = {}
        
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                audio_files = list(class_dir.glob('*.wav'))
                dataset[class_name] = [str(f) for f in audio_files]
        
        return dataset
    
    def split_dataset(
        self,
        dataset: Dict[str, List[str]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[Dict, Dict, Dict]:
        """
        데이터셋 분할
        
        Args:
            dataset: 전체 데이터셋
            train_ratio: 훈련 세트 비율
            val_ratio: 검증 세트 비율
            test_ratio: 테스트 세트 비율
        
        Returns:
            train_set, val_set, test_set
        """
        train_set, val_set, test_set = {}, {}, {}
        
        for class_name, files in dataset.items():
            n_samples = len(files)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            # 셔플
            shuffled = np.random.permutation(files).tolist()
            
            train_set[class_name] = shuffled[:n_train]
            val_set[class_name] = shuffled[n_train:n_train+n_val]
            test_set[class_name] = shuffled[n_train+n_val:]
        
        return train_set, val_set, test_set


if __name__ == "__main__":
    print("Data Processing Module")
    print("=" * 60)
    
    print("\n[사용 예제]")
    print("""
    # 합성 음성 생성
    generator = SyntheticSpeechGenerator(sr=16000)
    signal = generator.generate_speech_sample('chirp', duration=1.0)
    
    # 오디오 전처리
    preprocessor = AudioPreprocessor(sr=16000)
    audio = preprocessor.load_audio('audio.wav')
    trimmed = preprocessor.trim_silence(audio)
    normalized = preprocessor.normalize_loudness(trimmed)
    
    # 데이터셋 관리
    manager = DatasetManager('data/speech')
    manager.create_synthetic_dataset([('hello', 'chirp'), ('bye', 'formant')])
    dataset = manager.load_dataset()
    """)
