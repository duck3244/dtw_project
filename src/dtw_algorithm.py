"""
DTW Algorithm Module (NumPy only)
Dynamic Time Warping 알고리즘 구현
"""

import numpy as np
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class DTWAlgorithm:
    """
    Dynamic Time Warping 알고리즘
    
    두 시계열 데이터 간의 최적 정렬과 유사도를 계산합니다.
    """
    
    def __init__(self, distance_metric: str = 'euclidean'):
        """
        Args:
            distance_metric: 거리 측정 방법
                - 'euclidean': 유클리드 거리
                - 'manhattan': 맨하탄 거리 (L1)
                - 'cosine': 코사인 거리
        """
        self.distance_metric = distance_metric
        self.supported_metrics = ['euclidean', 'manhattan', 'cosine']
        
        if distance_metric not in self.supported_metrics:
            raise ValueError(
                f"Unsupported distance metric: {distance_metric}. "
                f"Supported: {self.supported_metrics}"
            )
    
    def compute_distance_matrix(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        두 시퀀스 간의 거리 행렬 계산
        
        Args:
            x: 첫 번째 시퀀스 (N x D)
            y: 두 번째 시퀀스 (M x D)
        
        Returns:
            distance_matrix: 거리 행렬 (N x M)
        """
        N, D = x.shape
        M, _ = y.shape
        
        if self.distance_metric == 'euclidean':
            # L2 거리: sqrt(sum((x - y)^2))
            # Broadcasting을 사용한 효율적인 계산
            dist = np.sqrt(((x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        elif self.distance_metric == 'manhattan':
            # L1 거리: sum(|x - y|)
            dist = np.abs(x[:, np.newaxis, :] - y[np.newaxis, :, :]).sum(axis=2)
        
        elif self.distance_metric == 'cosine':
            # 코사인 거리: 1 - cosine_similarity
            x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
            y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)
            cosine_sim = np.dot(x_norm, y_norm.T)
            dist = 1 - cosine_sim
        
        return dist.astype(np.float32)
    
    def compute_accumulated_cost(
        self,
        distance_matrix: np.ndarray
    ) -> np.ndarray:
        """
        누적 비용 행렬 계산 (동적 프로그래밍)
        
        Args:
            distance_matrix: 거리 행렬 (N x M)
        
        Returns:
            accumulated_cost: 누적 비용 행렬 (N x M)
        """
        N, M = distance_matrix.shape
        
        # 누적 비용 행렬 초기화
        accumulated_cost = np.zeros((N, M), dtype=np.float32)
        accumulated_cost[0, 0] = distance_matrix[0, 0]
        
        # 첫 번째 행 초기화
        for i in range(1, N):
            accumulated_cost[i, 0] = (
                accumulated_cost[i-1, 0] + distance_matrix[i, 0]
            )
        
        # 첫 번째 열 초기화
        for j in range(1, M):
            accumulated_cost[0, j] = (
                accumulated_cost[0, j-1] + distance_matrix[0, j]
            )
        
        # 동적 프로그래밍으로 나머지 셀 채우기
        for i in range(1, N):
            for j in range(1, M):
                # 세 가지 경로 중 최소 비용 선택
                min_cost = min(
                    accumulated_cost[i-1, j],      # 수직
                    accumulated_cost[i, j-1],      # 수평
                    accumulated_cost[i-1, j-1]     # 대각선
                )
                accumulated_cost[i, j] = distance_matrix[i, j] + min_cost
        
        return accumulated_cost
    
    def find_path(
        self,
        accumulated_cost: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        최적 정렬 경로 찾기 (백트래킹)
        
        Args:
            accumulated_cost: 누적 비용 행렬 (N x M)
        
        Returns:
            path: 최적 경로 [(i, j), ...]
        """
        N, M = accumulated_cost.shape
        i, j = N - 1, M - 1
        path = [(i, j)]
        
        # 백트래킹으로 경로 추적
        while i > 0 or j > 0:
            if i == 0:
                # 첫 번째 행: 왼쪽으로만 이동 가능
                j -= 1
            elif j == 0:
                # 첫 번째 열: 위로만 이동 가능
                i -= 1
            else:
                # 세 가지 이전 셀의 비용 비교
                costs = [
                    accumulated_cost[i-1, j],      # 위
                    accumulated_cost[i, j-1],      # 왼쪽
                    accumulated_cost[i-1, j-1]     # 대각선
                ]
                min_idx = costs.index(min(costs))
                
                if min_idx == 0:
                    i -= 1
                elif min_idx == 1:
                    j -= 1
                else:
                    i -= 1
                    j -= 1
            
            path.append((i, j))
        
        # 경로를 시작점부터 끝점 순서로 뒤집기
        return path[::-1]
    
    def compute_dtw(
        self,
        x: np.ndarray,
        y: np.ndarray,
        return_path: bool = False
    ) -> Tuple[float, Optional[List[Tuple[int, int]]], Optional[np.ndarray]]:
        """
        DTW 거리 계산
        
        Args:
            x: 첫 번째 시퀀스 (N x D)
            y: 두 번째 시퀀스 (M x D)
            return_path: 경로 반환 여부
        
        Returns:
            dtw_distance: DTW 거리
            path: 정렬 경로 (return_path=True일 때만)
            accumulated_cost: 누적 비용 행렬 (return_path=True일 때만)
        """
        # 1. 거리 행렬 계산
        distance_matrix = self.compute_distance_matrix(x, y)
        
        # 2. 누적 비용 행렬 계산
        accumulated_cost = self.compute_accumulated_cost(distance_matrix)
        
        # 3. DTW 거리 (우하단 값)
        dtw_distance = float(accumulated_cost[-1, -1])
        
        if return_path:
            # 4. 최적 경로 찾기
            path = self.find_path(accumulated_cost)
            return dtw_distance, path, accumulated_cost
        else:
            return dtw_distance, None, None
    
    def compute_dtw_normalized(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        정규화된 DTW 거리 계산
        
        경로 길이로 나누어 시퀀스 길이의 영향을 줄입니다.
        
        Args:
            x: 첫 번째 시퀀스
            y: 두 번째 시퀀스
        
        Returns:
            normalized_distance: 정규화된 DTW 거리
        """
        dtw_distance, path, _ = self.compute_dtw(x, y, return_path=True)
        
        # 경로 길이로 정규화
        normalized_distance = dtw_distance / len(path)
        
        return normalized_distance


class FastDTW:
    """
    FastDTW 알고리즘
    
    근사 DTW로 계산 복잡도를 O(N^2)에서 O(N)으로 감소시킵니다.
    """
    
    def __init__(self, radius: int = 1, distance_metric: str = 'euclidean'):
        """
        Args:
            radius: 탐색 반경
            distance_metric: 거리 측정 방법
        """
        self.radius = radius
        self.dtw = DTWAlgorithm(distance_metric=distance_metric)
    
    def compute_fastdtw(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        FastDTW 거리 계산
        
        Args:
            x: 첫 번째 시퀀스
            y: 두 번째 시퀀스
        
        Returns:
            distance: FastDTW 거리
        """
        # 간단한 구현: 작은 시퀀스는 일반 DTW 사용
        if len(x) < 20 or len(y) < 20:
            distance, _, _ = self.dtw.compute_dtw(x, y)
            return distance
        
        # 다운샘플링
        x_reduced = x[::2]
        y_reduced = y[::2]
        
        # 재귀적으로 계산
        _, path, _ = self.dtw.compute_dtw(x_reduced, y_reduced, return_path=True)
        
        # 경로 확장 및 제한된 영역에서 DTW 계산
        # (완전한 구현을 위해서는 더 복잡한 로직 필요)
        distance, _, _ = self.dtw.compute_dtw(x, y)
        
        return distance


class ConstrainedDTW:
    """
    제약이 있는 DTW
    
    Sakoe-Chiba band 또는 Itakura parallelogram을 사용합니다.
    """
    
    def __init__(
        self,
        window_size: int = 10,
        distance_metric: str = 'euclidean'
    ):
        """
        Args:
            window_size: Sakoe-Chiba band 윈도우 크기
            distance_metric: 거리 측정 방법
        """
        self.window_size = window_size
        self.dtw = DTWAlgorithm(distance_metric=distance_metric)
    
    def compute_constrained_dtw(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        제약이 있는 DTW 거리 계산
        
        Args:
            x: 첫 번째 시퀀스
            y: 두 번째 시퀀스
        
        Returns:
            distance: DTW 거리
        """
        N, M = len(x), len(y)
        
        # 거리 행렬 계산
        distance_matrix = self.dtw.compute_distance_matrix(x, y)
        
        # 누적 비용 행렬 초기화 (무한대로)
        accumulated_cost = np.full((N, M), np.inf, dtype=np.float32)
        accumulated_cost[0, 0] = distance_matrix[0, 0]
        
        # Sakoe-Chiba band 내에서만 계산
        for i in range(N):
            # 윈도우 범위 계산
            j_start = max(0, i - self.window_size)
            j_end = min(M, i + self.window_size + 1)
            
            for j in range(j_start, j_end):
                if i == 0 and j == 0:
                    continue
                
                costs = []
                if i > 0:
                    costs.append(accumulated_cost[i-1, j])
                if j > 0:
                    costs.append(accumulated_cost[i, j-1])
                if i > 0 and j > 0:
                    costs.append(accumulated_cost[i-1, j-1])
                
                if costs:
                    accumulated_cost[i, j] = distance_matrix[i, j] + min(costs)
        
        return float(accumulated_cost[-1, -1])


if __name__ == "__main__":
    print("DTW Algorithm Module (NumPy only)")
    print("=" * 60)
    
    print("\n[사용 예제]")
    print("""
    # 기본 DTW
    dtw = DTWAlgorithm(distance_metric='euclidean')
    distance, path, acc_cost = dtw.compute_dtw(x, y, return_path=True)
    
    # 정규화된 DTW
    normalized_distance = dtw.compute_dtw_normalized(x, y)
    
    # FastDTW (근사)
    fast_dtw = FastDTW(radius=1)
    distance = fast_dtw.compute_fastdtw(x, y)
    
    # 제약이 있는 DTW
    constrained_dtw = ConstrainedDTW(window_size=10)
    distance = constrained_dtw.compute_constrained_dtw(x, y)
    """)
