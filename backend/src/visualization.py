"""
Visualization Module
DTW 및 음성 인식 결과 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import List, Tuple, Optional, Dict
import seaborn as sns


class DTWVisualizer:
    """
    DTW 관련 시각화
    """
    
    @staticmethod
    def plot_dtw_alignment(
        x: np.ndarray,
        y: np.ndarray,
        path: List[Tuple[int, int]],
        accumulated_cost: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        DTW 정렬 시각화
        
        Args:
            x: 첫 번째 시퀀스 (T x D)
            y: 두 번째 시퀀스 (T x D)
            path: 정렬 경로
            accumulated_cost: 누적 비용 행렬
            save_path: 저장 경로
            figsize: 그림 크기
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 시퀀스 1 (첫 번째 차원만)
        axes[0, 0].plot(x[:, 0].numpy(), linewidth=2)
        axes[0, 0].set_title('Sequence 1 (Test)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Time Frame')
        axes[0, 0].set_ylabel('Feature Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 시퀀스 2 (첫 번째 차원만)
        axes[0, 1].plot(y[:, 0].numpy(), linewidth=2, color='orange')
        axes[0, 1].set_title('Sequence 2 (Template)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Time Frame')
        axes[0, 1].set_ylabel('Feature Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 누적 비용 행렬
        im = axes[1, 0].imshow(
            accumulated_cost.numpy().T,
            origin='lower',
            cmap='viridis',
            aspect='auto',
            interpolation='nearest'
        )
        axes[1, 0].set_title('Accumulated Cost Matrix', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Sequence 1')
        axes[1, 0].set_ylabel('Sequence 2')
        plt.colorbar(im, ax=axes[1, 0], label='Cost')
        
        # 4. 정렬 경로
        path_array = np.array(path)
        axes[1, 1].imshow(
            accumulated_cost.numpy().T,
            origin='lower',
            cmap='gray',
            aspect='auto',
            interpolation='nearest'
        )
        axes[1, 1].plot(
            path_array[:, 0],
            path_array[:, 1],
            'r-',
            linewidth=3,
            label='Warping Path'
        )
        axes[1, 1].set_title('DTW Alignment Path', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Sequence 1')
        axes[1, 1].set_ylabel('Sequence 2')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_distance_matrix(
        distance_matrix: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        거리 행렬 시각화
        
        Args:
            distance_matrix: 거리 행렬
            save_path: 저장 경로
            figsize: 그림 크기
        """
        plt.figure(figsize=figsize)
        
        im = plt.imshow(
            distance_matrix.numpy().T,
            origin='lower',
            cmap='hot',
            aspect='auto'
        )
        plt.colorbar(im, label='Distance')
        plt.title('Distance Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Sequence 1 Frame')
        plt.ylabel('Sequence 2 Frame')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()


class FeatureVisualizer:
    """
    특징 시각화
    """
    
    @staticmethod
    def plot_mfcc(
        mfcc: np.ndarray,
        sr: int = 16000,
        hop_length: int = 512,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        MFCC 특징 시각화
        
        Args:
            mfcc: MFCC 특징 (T x D)
            sr: 샘플링 레이트
            hop_length: 홉 길이
            save_path: 저장 경로
            figsize: 그림 크기
        """
        plt.figure(figsize=figsize)
        
        # MFCC를 (D x T) 형태로 변환
        mfcc_display = mfcc.T.numpy()
        
        librosa.display.specshow(
            mfcc_display,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel',
            cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB', label='MFCC Coefficient')
        plt.title('MFCC Features', fontsize=16, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('MFCC Coefficient')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_waveform(
        audio: np.ndarray,
        sr: int = 16000,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4)
    ):
        """
        파형 시각화
        
        Args:
            audio: 오디오 신호
            sr: 샘플링 레이트
            save_path: 저장 경로
            figsize: 그림 크기
        """
        plt.figure(figsize=figsize)
        
        librosa.display.waveshow(audio, sr=sr)
        plt.title('Waveform', fontsize=16, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_spectrogram(
        audio: np.ndarray,
        sr: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        스펙트로그램 시각화
        
        Args:
            audio: 오디오 신호
            sr: 샘플링 레이트
            n_fft: FFT 크기
            hop_length: 홉 길이
            save_path: 저장 경로
            figsize: 그림 크기
        """
        plt.figure(figsize=figsize)
        
        # 멜 스펙트로그램 계산
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        librosa.display.specshow(
            mel_spec_db,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel',
            cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB', label='Power (dB)')
        plt.title('Mel Spectrogram', fontsize=16, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()


class ResultVisualizer:
    """
    인식 결과 시각화
    """
    
    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: Dict[str, Dict[str, int]],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        혼동 행렬 시각화
        
        Args:
            confusion_matrix: {true_label: {pred_label: count}}
            save_path: 저장 경로
            figsize: 그림 크기
        """
        # 행렬 변환
        labels = sorted(confusion_matrix.keys())
        matrix = np.zeros((len(labels), len(labels)))
        
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                matrix[i, j] = confusion_matrix.get(true_label, {}).get(pred_label, 0)
        
        # 정규화 (행 기준)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized_matrix = matrix / (row_sums + 1e-8)
        
        # 시각화
        plt.figure(figsize=figsize)
        sns.heatmap(
            normalized_matrix,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Proportion'}
        )
        plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_recognition_scores(
        scores: Dict[str, float],
        true_label: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        인식 점수 시각화
        
        Args:
            scores: {label: distance}
            true_label: 실제 레이블 (표시용)
            save_path: 저장 경로
            figsize: 그림 크기
        """
        # 점수로 정렬
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        labels = [item[0] for item in sorted_scores]
        distances = [item[1] for item in sorted_scores]
        
        # 색상 설정
        colors = ['green' if label == true_label else 'steelblue' 
                  for label in labels]
        
        plt.figure(figsize=figsize)
        bars = plt.barh(labels, distances, color=colors)
        
        plt.xlabel('DTW Distance', fontsize=12)
        plt.ylabel('Class Label', fontsize=12)
        plt.title('Recognition Scores (Lower is Better)', fontsize=16, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # 값 표시
        for bar, dist in zip(bars, distances):
            plt.text(
                dist + 0.01 * max(distances),
                bar.get_y() + bar.get_height() / 2,
                f'{dist:.2f}',
                va='center'
            )
        
        if true_label:
            plt.legend(['Correct' if c == 'green' else 'Incorrect' 
                       for c in colors], loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_accuracy_comparison(
        results: Dict[str, float],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        정확도 비교 시각화
        
        Args:
            results: {method_name: accuracy}
            save_path: 저장 경로
            figsize: 그림 크기
        """
        methods = list(results.keys())
        accuracies = list(results.values())
        
        plt.figure(figsize=figsize)
        bars = plt.bar(methods, accuracies, color='skyblue', edgecolor='navy', linewidth=2)
        
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.ylim(0, 110)
        plt.grid(axis='y', alpha=0.3)
        
        # 값 표시
        for bar, acc in zip(bars, accuracies):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f'{acc:.1f}%',
                ha='center',
                fontweight='bold'
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_distance_heatmap(
        test_features: List[np.ndarray],
        template_features: List[np.ndarray],
        test_labels: List[str],
        template_labels: List[str],
        dtw_algorithm,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        테스트-템플릿 거리 히트맵
        
        Args:
            test_features: 테스트 특징 리스트
            template_features: 템플릿 특징 리스트
            test_labels: 테스트 레이블
            template_labels: 템플릿 레이블
            dtw_algorithm: DTW 알고리즘
            save_path: 저장 경로
            figsize: 그림 크기
        """
        n_test = len(test_features)
        n_template = len(template_features)
        
        # 거리 행렬 계산
        distances = np.zeros((n_test, n_template))
        
        for i, test_feat in enumerate(test_features):
            for j, template_feat in enumerate(template_features):
                dist, _, _ = dtw_algorithm.compute_dtw(test_feat, template_feat)
                distances[i, j] = dist
        
        # 시각화
        plt.figure(figsize=figsize)
        sns.heatmap(
            distances,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            xticklabels=template_labels,
            yticklabels=test_labels,
            cbar_kws={'label': 'DTW Distance'}
        )
        plt.title('Test-Template Distance Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Template')
        plt.ylabel('Test Sample')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("Visualization Module")
    print("=" * 60)
    
    print("\n[사용 예제]")
    print("""
    # DTW 정렬 시각화
    dtw_viz = DTWVisualizer()
    dtw_viz.plot_dtw_alignment(x, y, path, acc_cost, save_path='dtw.png')
    
    # MFCC 시각화
    feat_viz = FeatureVisualizer()
    feat_viz.plot_mfcc(mfcc_features)
    feat_viz.plot_waveform(audio_signal)
    
    # 결과 시각화
    result_viz = ResultVisualizer()
    result_viz.plot_confusion_matrix(confusion_matrix)
    result_viz.plot_recognition_scores(scores, true_label='hello')
    """)
