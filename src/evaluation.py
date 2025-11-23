"""
Evaluation Module
음성 인식 시스템 평가 및 벤치마크
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class Evaluator:
    """
    음성 인식기 평가
    """
    
    def __init__(self, recognizer):
        """
        Args:
            recognizer: 음성 인식기
        """
        self.recognizer = recognizer
    
    def evaluate(
        self,
        test_data: List[Tuple[str, str]],
        verbose: bool = True
    ) -> Dict:
        """
        인식기 평가
        
        Args:
            test_data: [(audio_path, true_label), ...]
            verbose: 상세 출력 여부
        
        Returns:
            results: 평가 결과 딕셔너리
        """
        correct = 0
        total = len(test_data)
        predictions = []
        true_labels = []
        distances = []
        
        if verbose:
            print("=" * 70)
            print("Evaluation Results")
            print("=" * 70)
        
        for audio_path, true_label in test_data:
            try:
                pred_label, distance = self.recognizer.recognize(audio_path)
                predictions.append(pred_label)
                true_labels.append(true_label)
                distances.append(distance)
                
                is_correct = (pred_label == true_label)
                if is_correct:
                    correct += 1
                
                if verbose:
                    status = "✓" if is_correct else "✗"
                    print(f"{status} True: {true_label:10s} | "
                          f"Pred: {pred_label:10s} | "
                          f"Distance: {distance:8.2f}")
            
            except Exception as e:
                if verbose:
                    print(f"✗ Error processing {audio_path}: {e}")
        
        accuracy = correct / total if total > 0 else 0.0
        
        # 혼동 행렬 생성
        confusion_matrix = self._compute_confusion_matrix(
            true_labels,
            predictions
        )
        
        # 클래스별 정확도
        class_accuracy = self._compute_class_accuracy(confusion_matrix)
        
        # 결과 딕셔너리
        results = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': predictions,
            'true_labels': true_labels,
            'distances': distances,
            'confusion_matrix': confusion_matrix,
            'class_accuracy': class_accuracy
        }
        
        if verbose:
            print("=" * 70)
            print(f"Overall Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
            print("\nClass-wise Accuracy:")
            for label, acc in class_accuracy.items():
                print(f"  {label:10s}: {acc * 100:.2f}%")
        
        return results
    
    def evaluate_from_arrays(
        self,
        test_data: List[Tuple[np.ndarray, str]],
        verbose: bool = True
    ) -> Dict:
        """
        numpy 배열에서 평가
        
        Args:
            test_data: [(audio_array, true_label), ...]
            verbose: 상세 출력 여부
        
        Returns:
            results: 평가 결과
        """
        correct = 0
        total = len(test_data)
        predictions = []
        true_labels = []
        distances = []
        
        if verbose:
            print("=" * 70)
            print("Evaluation Results")
            print("=" * 70)
        
        for audio_array, true_label in test_data:
            pred_label, distance = self.recognizer.recognize_from_array(audio_array)
            predictions.append(pred_label)
            true_labels.append(true_label)
            distances.append(distance)
            
            is_correct = (pred_label == true_label)
            if is_correct:
                correct += 1
            
            if verbose:
                status = "✓" if is_correct else "✗"
                print(f"{status} True: {true_label:10s} | "
                      f"Pred: {pred_label:10s} | "
                      f"Distance: {distance:8.2f}")
        
        accuracy = correct / total if total > 0 else 0.0
        confusion_matrix = self._compute_confusion_matrix(true_labels, predictions)
        class_accuracy = self._compute_class_accuracy(confusion_matrix)
        
        results = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': predictions,
            'true_labels': true_labels,
            'distances': distances,
            'confusion_matrix': confusion_matrix,
            'class_accuracy': class_accuracy
        }
        
        if verbose:
            print("=" * 70)
            print(f"Overall Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
        
        return results
    
    def _compute_confusion_matrix(
        self,
        true_labels: List[str],
        predictions: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """혼동 행렬 계산"""
        unique_labels = sorted(set(true_labels + predictions))
        confusion = {
            label: {l: 0 for l in unique_labels}
            for label in unique_labels
        }
        
        for true, pred in zip(true_labels, predictions):
            confusion[true][pred] += 1
        
        return confusion
    
    def _compute_class_accuracy(
        self,
        confusion_matrix: Dict[str, Dict[str, int]]
    ) -> Dict[str, float]:
        """클래스별 정확도 계산"""
        class_accuracy = {}
        
        for true_label, predictions in confusion_matrix.items():
            total = sum(predictions.values())
            correct = predictions.get(true_label, 0)
            class_accuracy[true_label] = correct / total if total > 0 else 0.0
        
        return class_accuracy
    
    def cross_validate(
        self,
        dataset: Dict[str, List[str]],
        k_folds: int = 5
    ) -> Dict:
        """
        K-Fold 교차 검증
        
        Args:
            dataset: {class_name: [file_paths, ...]}
            k_folds: fold 수
        
        Returns:
            cv_results: 교차 검증 결과
        """
        fold_results = []
        
        for fold in range(k_folds):
            print(f"\n{'='*70}")
            print(f"Fold {fold + 1}/{k_folds}")
            print('='*70)
            
            # 데이터 분할
            train_data, test_data = self._split_fold(dataset, fold, k_folds)
            
            # 템플릿 재학습
            self.recognizer.templates.clear()
            for label, files in train_data.items():
                for file in files:
                    self.recognizer.add_template(label, file)
            
            # 테스트
            test_list = []
            for label, files in test_data.items():
                for file in files:
                    test_list.append((file, label))
            
            results = self.evaluate(test_list, verbose=False)
            fold_results.append(results)
            
            print(f"Fold {fold + 1} Accuracy: {results['accuracy'] * 100:.2f}%")
        
        # 평균 결과
        avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
        std_accuracy = np.std([r['accuracy'] for r in fold_results])
        
        cv_results = {
            'fold_results': fold_results,
            'avg_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy
        }
        
        print(f"\n{'='*70}")
        print(f"Cross-Validation Results")
        print(f"Average Accuracy: {avg_accuracy * 100:.2f}% ± {std_accuracy * 100:.2f}%")
        print('='*70)
        
        return cv_results
    
    def _split_fold(
        self,
        dataset: Dict[str, List[str]],
        fold: int,
        k_folds: int
    ) -> Tuple[Dict, Dict]:
        """K-Fold 데이터 분할"""
        train_data = defaultdict(list)
        test_data = defaultdict(list)
        
        for label, files in dataset.items():
            n_samples = len(files)
            fold_size = n_samples // k_folds
            
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < k_folds - 1 else n_samples
            
            test_data[label] = files[test_start:test_end]
            train_data[label] = files[:test_start] + files[test_end:]
        
        return train_data, test_data


class Benchmarker:
    """
    성능 벤치마크
    """
    
    @staticmethod
    def benchmark_dtw_metrics(
        x,
        y,
        metrics: List[str] = ['euclidean', 'manhattan', 'cosine']
    ) -> Dict[str, Dict]:
        """
        다양한 DTW 메트릭 벤치마크
        
        Args:
            x: 첫 번째 시퀀스
            y: 두 번째 시퀀스
            metrics: 테스트할 메트릭 리스트
        
        Returns:
            results: 벤치마크 결과
        """
        from dtw_algorithm import DTWAlgorithm
        
        results = {}
        
        print("=" * 70)
        print("DTW Metrics Benchmark")
        print("=" * 70)
        
        for metric in metrics:
            dtw = DTWAlgorithm(distance_metric=metric)
            
            # 시간 측정
            start_time = time.time()
            distance, _, _ = dtw.compute_dtw(x, y)
            elapsed_time = time.time() - start_time
            
            results[metric] = {
                'distance': distance,
                'time': elapsed_time
            }
            
            print(f"{metric:12s}: Distance={distance:8.2f}, Time={elapsed_time*1000:.2f}ms")
        
        return results
    
    @staticmethod
    def benchmark_recognition_speed(
        recognizer,
        test_data: List[Tuple[str, str]],
        num_runs: int = 3
    ) -> Dict:
        """
        인식 속도 벤치마크
        
        Args:
            recognizer: 음성 인식기
            test_data: 테스트 데이터
            num_runs: 실행 횟수
        
        Returns:
            benchmark_results: 벤치마크 결과
        """
        times = []
        
        print("=" * 70)
        print(f"Recognition Speed Benchmark ({num_runs} runs)")
        print("=" * 70)
        
        for run in range(num_runs):
            run_times = []
            
            for audio_path, _ in test_data:
                start_time = time.time()
                recognizer.recognize(audio_path)
                elapsed_time = time.time() - start_time
                run_times.append(elapsed_time)
            
            avg_time = np.mean(run_times)
            times.append(avg_time)
            print(f"Run {run + 1}: {avg_time*1000:.2f}ms per sample")
        
        results = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
        
        print(f"\nAverage: {results['avg_time']*1000:.2f}ms ± {results['std_time']*1000:.2f}ms")
        
        return results
    
    @staticmethod
    def compare_recognizers(
        recognizers: Dict[str, object],
        test_data: List[Tuple[str, str]]
    ) -> Dict:
        """
        여러 인식기 비교
        
        Args:
            recognizers: {name: recognizer}
            test_data: 테스트 데이터
        
        Returns:
            comparison_results: 비교 결과
        """
        results = {}
        
        print("=" * 70)
        print("Recognizer Comparison")
        print("=" * 70)
        
        for name, recognizer in recognizers.items():
            print(f"\nTesting {name}...")
            
            evaluator = Evaluator(recognizer)
            eval_results = evaluator.evaluate(test_data, verbose=False)
            
            # 속도 측정
            start_time = time.time()
            for audio_path, _ in test_data:
                recognizer.recognize(audio_path)
            elapsed_time = time.time() - start_time
            
            results[name] = {
                'accuracy': eval_results['accuracy'],
                'avg_time': elapsed_time / len(test_data)
            }
            
            print(f"  Accuracy: {eval_results['accuracy'] * 100:.2f}%")
            print(f"  Avg Time: {results[name]['avg_time']*1000:.2f}ms")
        
        return results


class MetricCalculator:
    """
    평가 지표 계산
    """
    
    @staticmethod
    def calculate_precision_recall_f1(
        confusion_matrix: Dict[str, Dict[str, int]],
        target_label: str
    ) -> Dict[str, float]:
        """
        정밀도, 재현율, F1 점수 계산
        
        Args:
            confusion_matrix: 혼동 행렬
            target_label: 타겟 레이블
        
        Returns:
            metrics: 평가 지표
        """
        # True Positive
        tp = confusion_matrix[target_label][target_label]
        
        # False Positive
        fp = sum(confusion_matrix[label][target_label] 
                for label in confusion_matrix if label != target_label)
        
        # False Negative
        fn = sum(confusion_matrix[target_label][label]
                for label in confusion_matrix[target_label] if label != target_label)
        
        # 정밀도 (Precision)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # 재현율 (Recall)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 점수
        f1 = 2 * (precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    @staticmethod
    def calculate_macro_metrics(
        confusion_matrix: Dict[str, Dict[str, int]]
    ) -> Dict[str, float]:
        """
        매크로 평균 지표 계산
        
        Args:
            confusion_matrix: 혼동 행렬
        
        Returns:
            macro_metrics: 매크로 평균 지표
        """
        labels = list(confusion_matrix.keys())
        precisions = []
        recalls = []
        f1s = []
        
        for label in labels:
            metrics = MetricCalculator.calculate_precision_recall_f1(
                confusion_matrix,
                label
            )
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1s.append(metrics['f1'])
        
        return {
            'macro_precision': np.mean(precisions),
            'macro_recall': np.mean(recalls),
            'macro_f1': np.mean(f1s)
        }


if __name__ == "__main__":
    print("Evaluation Module")
    print("=" * 60)
    
    print("\n[사용 예제]")
    print("""
    # 기본 평가
    evaluator = Evaluator(recognizer)
    results = evaluator.evaluate(test_data, verbose=True)
    
    # 교차 검증
    cv_results = evaluator.cross_validate(dataset, k_folds=5)
    
    # 벤치마크
    benchmarker = Benchmarker()
    speed_results = benchmarker.benchmark_recognition_speed(recognizer, test_data)
    
    # 메트릭 계산
    calculator = MetricCalculator()
    metrics = calculator.calculate_precision_recall_f1(confusion_matrix, 'hello')
    """)
