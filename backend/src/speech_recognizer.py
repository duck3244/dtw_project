"""
Speech Recognizer Module
DTW 기반 음성 인식기
"""

import logging
import numpy as np
from typing import Tuple, List, Dict, Optional
import pickle
from pathlib import Path

log = logging.getLogger(__name__)


class SpeechRecognizer:
    """
    DTW 기반 음성 인식기.

    스코어 정책 (기본값은 권장 모드):
        normalize=True
            DTW 거리를 정렬 경로 길이로 나누어 비교 (시퀀스 길이 차의 영향 제거).
        score_aggregation='min'
            레이블 점수를 그 레이블 템플릿들과의 거리 중 최소값으로 결정.
            노이즈 템플릿이 있어도 평균이 끌려가지 않아 강건.
        score_aggregation='mean'
            레이블별 평균 거리. 호환 모드 (이전 기본 동작).
        score_aggregation='knn'
            레이블 무관 전체 템플릿에서 가장 가까운 ``knn_k`` 개로 다수결.
            동률은 평균 거리가 작은 쪽 우선.
    """

    _VALID_AGG = ("min", "mean", "knn")

    def __init__(
        self,
        mfcc_extractor,
        dtw_algorithm,
        normalize: bool = True,
        score_aggregation: str = "min",
        knn_k: int = 3,
    ):
        if score_aggregation not in self._VALID_AGG:
            raise ValueError(
                f"Unknown score_aggregation: {score_aggregation!r}. "
                f"Choose one of {self._VALID_AGG}"
            )
        self.mfcc_extractor = mfcc_extractor
        self.dtw_algorithm = dtw_algorithm
        self.normalize = bool(normalize)
        self.score_aggregation = score_aggregation
        self.knn_k = max(1, int(knn_k))
        self.templates = {}  # {label: [features, ...]}
        self.template_metadata = {}  # {label: [metadata, ...]}

    def _pairwise_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the (optionally normalized) DTW distance between two feature seqs."""
        if self.normalize:
            return float(self.dtw_algorithm.compute_dtw_normalized(x, y))
        distance, _, _ = self.dtw_algorithm.compute_dtw(x, y)
        return float(distance)
    
    def add_template(
        self,
        label: str,
        audio_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        템플릿 추가 (파일에서)
        
        Args:
            label: 단어/클래스 레이블
            audio_path: 오디오 파일 경로
            metadata: 추가 메타데이터 (선택)
        """
        features = self.mfcc_extractor.extract(audio_path)
        
        if label not in self.templates:
            self.templates[label] = []
            self.template_metadata[label] = []
        
        self.templates[label].append(features)
        self.template_metadata[label].append(metadata or {})
        
        log.info("template added: %r shape=%s", label, features.shape)
    
    def add_template_from_array(
        self,
        label: str,
        audio_array: np.ndarray,
        metadata: Optional[Dict] = None
    ):
        """
        템플릿 추가 (배열에서)
        
        Args:
            label: 단어/클래스 레이블
            audio_array: 오디오 신호 배열
            metadata: 추가 메타데이터
        """
        features = self.mfcc_extractor.extract_from_array(audio_array)
        
        if label not in self.templates:
            self.templates[label] = []
            self.template_metadata[label] = []
        
        self.templates[label].append(features)
        self.template_metadata[label].append(metadata or {})
    
    def add_templates_batch(
        self,
        label: str,
        audio_paths: List[str]
    ):
        """
        여러 템플릿을 배치로 추가
        
        Args:
            label: 단어/클래스 레이블
            audio_paths: 오디오 파일 경로 리스트
        """
        for path in audio_paths:
            self.add_template(label, path)
    
    def remove_template(self, label: str, index: int = -1):
        """
        템플릿 제거
        
        Args:
            label: 레이블
            index: 제거할 템플릿 인덱스 (-1: 마지막)
        """
        if label in self.templates and self.templates[label]:
            self.templates[label].pop(index)
            self.template_metadata[label].pop(index)
            log.info("template removed: %r at index %d", label, index)
    
    def recognize(
        self,
        audio_path: str,
        return_scores: bool = False,
        top_k: int = 1
    ) -> Tuple:
        """
        음성 인식 (파일에서)
        
        Args:
            audio_path: 인식할 오디오 파일 경로
            return_scores: 모든 클래스의 점수 반환 여부
            top_k: 상위 k개 결과 반환
        
        Returns:
            predicted_label: 예측된 레이블
            distance: DTW 거리
            (선택) scores: 모든 클래스의 점수
            (선택) top_k_results: 상위 k개 결과
        """
        # 테스트 특징 추출
        test_features = self.mfcc_extractor.extract(audio_path)
        
        return self._recognize_from_features(
            test_features,
            return_scores,
            top_k
        )
    
    def recognize_from_array(
        self,
        audio_array: np.ndarray,
        return_scores: bool = False,
        top_k: int = 1
    ) -> Tuple:
        """
        음성 인식 (배열에서)
        
        Args:
            audio_array: 오디오 신호 배열
            return_scores: 모든 클래스의 점수 반환 여부
            top_k: 상위 k개 결과 반환
        
        Returns:
            결과 튜플
        """
        test_features = self.mfcc_extractor.extract_from_array(audio_array)
        
        return self._recognize_from_features(
            test_features,
            return_scores,
            top_k
        )
    
    def _recognize_from_features(
        self,
        test_features: np.ndarray,
        return_scores: bool,
        top_k: int
    ) -> Tuple:
        """
        특징에서 음성 인식 (내부 메서드)
        """
        if not self.templates:
            raise ValueError("No templates available. Add templates first.")

        if self.score_aggregation == "knn":
            return self._recognize_knn(test_features, return_scores, top_k)

        # min / mean: 레이블별로 템플릿 거리 집계
        scores: Dict[str, float] = {}
        for label, template_list in self.templates.items():
            distances = [
                self._pairwise_distance(test_features, t) for t in template_list
            ]
            if self.score_aggregation == "min":
                scores[label] = float(np.min(distances))
            else:  # mean
                scores[label] = float(np.mean(distances))

        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        top_k_results = sorted_scores[:top_k]
        predicted_label = top_k_results[0][0]
        min_distance = top_k_results[0][1]

        if return_scores:
            return predicted_label, min_distance, scores, top_k_results
        return predicted_label, min_distance

    def _recognize_knn(
        self,
        test_features: np.ndarray,
        return_scores: bool,
        top_k: int,
    ) -> Tuple:
        """KNN 집계: 전체 템플릿 중 가장 가까운 k개로 다수결."""
        pairs: List[Tuple[str, float]] = []
        for label, template_list in self.templates.items():
            for tpl in template_list:
                pairs.append((label, self._pairwise_distance(test_features, tpl)))

        pairs.sort(key=lambda p: p[1])
        k = min(self.knn_k, len(pairs))
        neighbors = pairs[:k]

        votes: Dict[str, int] = {}
        for label, _ in neighbors:
            votes[label] = votes.get(label, 0) + 1

        # 레이블별 평균 거리 (이웃에 없는 레이블은 inf)
        label_scores: Dict[str, float] = {}
        for label in self.templates:
            ds = [d for lbl, d in neighbors if lbl == label]
            label_scores[label] = float(np.mean(ds)) if ds else float("inf")

        # 표 우선, 동률은 평균 거리 작은 쪽
        predicted_label = sorted(
            votes.keys(),
            key=lambda lbl: (-votes[lbl], label_scores[lbl]),
        )[0]
        # 반환 distance: 예측 레이블의 가장 가까운 이웃 거리
        nearest_distance = next(d for lbl, d in pairs if lbl == predicted_label)

        sorted_scores = sorted(label_scores.items(), key=lambda x: x[1])
        top_k_results = sorted_scores[:top_k]

        if return_scores:
            return predicted_label, nearest_distance, label_scores, top_k_results
        return predicted_label, nearest_distance
    
    def get_template_count(self) -> Dict[str, int]:
        """각 클래스의 템플릿 개수 반환"""
        return {label: len(templates) for label, templates in self.templates.items()}
    
    def get_labels(self) -> List[str]:
        """등록된 레이블 목록 반환"""
        return list(self.templates.keys())
    
    def save_templates(self, filepath: str):
        """
        템플릿을 파일로 저장
        
        Args:
            filepath: 저장할 파일 경로
        """
        save_data = {
            'templates': self.templates,
            'metadata': self.template_metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        log.info("templates saved to %s", filepath)
    
    def load_templates(self, filepath: str):
        """
        템플릿을 파일에서 로드
        
        Args:
            filepath: 로드할 파일 경로
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.templates = save_data['templates']
        self.template_metadata = save_data.get('metadata', {})
        
        log.info(
            "templates loaded from %s: %d classes %s",
            filepath,
            len(self.templates),
            self.get_template_count(),
        )


class EnsembleRecognizer:
    """
    앙상블 음성 인식기
    
    여러 인식기의 결과를 결합합니다.
    """
    
    def __init__(self, recognizers: List[SpeechRecognizer]):
        """
        Args:
            recognizers: 인식기 리스트
        """
        self.recognizers = recognizers
    
    def recognize_voting(
        self,
        audio_path: str
    ) -> Tuple[str, float]:
        """
        다수결 투표로 인식
        
        Args:
            audio_path: 오디오 파일 경로
        
        Returns:
            predicted_label: 예측된 레이블
            confidence: 신뢰도 (투표 비율)
        """
        votes = {}
        
        for recognizer in self.recognizers:
            label, _ = recognizer.recognize(audio_path)
            votes[label] = votes.get(label, 0) + 1
        
        # 최다 득표
        predicted_label = max(votes, key=votes.get)
        confidence = votes[predicted_label] / len(self.recognizers)
        
        return predicted_label, confidence
    
    def recognize_weighted(
        self,
        audio_path: str,
        weights: Optional[List[float]] = None
    ) -> Tuple[str, float]:
        """
        가중치 투표로 인식
        
        Args:
            audio_path: 오디오 파일 경로
            weights: 각 인식기의 가중치
        
        Returns:
            predicted_label: 예측된 레이블
            total_score: 종합 점수
        """
        if weights is None:
            weights = [1.0] * len(self.recognizers)
        
        scores = {}
        
        for recognizer, weight in zip(self.recognizers, weights):
            label, distance = recognizer.recognize(audio_path)
            # 거리를 점수로 변환 (작을수록 높은 점수)
            score = weight / (distance + 1e-8)
            scores[label] = scores.get(label, 0) + score
        
        predicted_label = max(scores, key=scores.get)
        total_score = scores[predicted_label]
        
        return predicted_label, total_score


class OnlineRecognizer:
    """
    온라인 음성 인식기
    
    스트리밍 오디오를 실시간으로 처리합니다.
    """
    
    def __init__(
        self,
        recognizer: SpeechRecognizer,
        buffer_size: int = 16000,  # 1초
        hop_size: int = 8000  # 0.5초
    ):
        """
        Args:
            recognizer: 기본 인식기
            buffer_size: 버퍼 크기 (샘플 수)
            hop_size: 홉 크기 (슬라이딩 윈도우)
        """
        self.recognizer = recognizer
        self.buffer_size = buffer_size
        self.hop_size = hop_size
        self.buffer = []
    
    def process_chunk(
        self,
        audio_chunk: np.ndarray
    ) -> Optional[Tuple[str, float]]:
        """
        오디오 청크 처리
        
        Args:
            audio_chunk: 오디오 청크
        
        Returns:
            인식 결과 (버퍼가 충분할 때만)
        """
        # 버퍼에 추가
        self.buffer.extend(audio_chunk)
        
        # 버퍼가 충분히 찼으면 인식
        if len(self.buffer) >= self.buffer_size:
            # 버퍼에서 윈도우 추출
            window = np.array(self.buffer[:self.buffer_size])
            
            # 인식
            label, distance = self.recognizer.recognize_from_array(window)
            
            # 버퍼 업데이트 (홉 크기만큼 이동)
            self.buffer = self.buffer[self.hop_size:]
            
            return label, distance
        
        return None
    
    def reset(self):
        """버퍼 초기화"""
        self.buffer = []


if __name__ == "__main__":
    print("Speech Recognizer Module (NumPy only)")
    print("=" * 60)
    
    print("\n[사용 예제]")
    print("""
    # 기본 인식기
    recognizer = SpeechRecognizer(mfcc_extractor, dtw_algorithm)
    recognizer.add_template('hello', 'audio/hello_1.wav')
    label, distance = recognizer.recognize('audio/test.wav')
    
    # 템플릿 저장/로드
    recognizer.save_templates('templates.pkl')
    recognizer.load_templates('templates.pkl')
    
    # 앙상블 인식기
    ensemble = EnsembleRecognizer([recognizer1, recognizer2])
    label, confidence = ensemble.recognize_voting('audio/test.wav')
    
    # 온라인 인식기
    online = OnlineRecognizer(recognizer)
    result = online.process_chunk(audio_chunk)
    """)
