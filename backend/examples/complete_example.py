"""
Complete Example: DTW Speech Recognition System
모든 기능을 통합한 완전한 예제
"""

import sys
sys.path.append('./src')

import numpy as np
from feature_extraction import MFCCExtractor, FeatureAugmenter
from dtw_algorithm import DTWAlgorithm
from speech_recognizer import SpeechRecognizer
from data_processing import SyntheticSpeechGenerator, DatasetManager
from visualization import DTWVisualizer, FeatureVisualizer, ResultVisualizer
from evaluation import Evaluator, Benchmarker


def main():
    """메인 실행 함수"""
    
    print("=" * 80)
    print("DTW 기반 음성 인식 시스템 - NumPy 전용 버전")
    print("=" * 80)
    print(f"\n✅ NumPy version: {np.__version__}")
    print("✅ PyTorch 의존성 없음 - 순수 NumPy로 동작\n")
    
    # ========== 1. 데이터 생성 ==========
    print("\n[1단계] 합성 음성 데이터 생성")
    print("-" * 80)
    
    generator = SyntheticSpeechGenerator(sr=16000)
    
    # 클래스 정의
    classes = {
        'hello': 'chirp',
        'goodbye': 'formant',
        'yes': 'pulse'
    }
    
    # 각 클래스별 샘플 생성
    dataset = {}
    for label, pattern in classes.items():
        samples = []
        for i in range(5):  # 클래스당 5개
            duration = 0.8 + np.random.rand() * 0.3
            noise = 0.02 + np.random.rand() * 0.03
            signal = generator.generate_speech_sample(
                pattern=pattern,
                duration=duration,
                noise_level=noise
            )
            samples.append(signal)
        dataset[label] = samples
        print(f"✓ '{label}' 클래스: {len(samples)}개 샘플 생성")
    
    # ========== 2. 특징 추출기 초기화 ==========
    print("\n[2단계] MFCC 특징 추출기 초기화")
    print("-" * 80)
    
    mfcc_extractor = MFCCExtractor(
        sr=16000,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512,
        n_mels=40,
        include_deltas=True
    )
    print(f"✓ MFCC 계수: {mfcc_extractor.n_mfcc}")
    print(f"✓ 특징 차원: {mfcc_extractor.get_feature_dimension()}")
    
    # ========== 3. DTW 알고리즘 초기화 ==========
    print("\n[3단계] DTW 알고리즘 초기화")
    print("-" * 80)
    
    dtw = DTWAlgorithm(distance_metric='euclidean')
    print(f"✓ 거리 메트릭: {dtw.distance_metric}")
    
    # ========== 4. 음성 인식기 훈련 ==========
    print("\n[4단계] 음성 인식기 훈련")
    print("-" * 80)
    
    recognizer = SpeechRecognizer(mfcc_extractor, dtw)
    
    # 훈련 데이터 (처음 3개)
    for label, samples in dataset.items():
        for i in range(3):
            recognizer.add_template_from_array(label, samples[i])
    
    print(f"✓ 총 템플릿 수: {sum(recognizer.get_template_count().values())}")
    for label, count in recognizer.get_template_count().items():
        print(f"  - {label}: {count}개")
    
    # ========== 5. 테스트 및 평가 ==========
    print("\n[5단계] 음성 인식 테스트")
    print("-" * 80)
    
    # 테스트 데이터 (마지막 2개)
    test_data = []
    for label, samples in dataset.items():
        for i in range(3, 5):
            test_data.append((samples[i], label))
    
    # 평가
    evaluator = Evaluator(recognizer)
    results = evaluator.evaluate_from_arrays(test_data, verbose=True)
    
    # ========== 6. 거리 메트릭 비교 ==========
    print("\n[6단계] 다양한 거리 메트릭 비교")
    print("-" * 80)
    
    metrics = ['euclidean', 'manhattan', 'cosine']
    metric_results = {}
    
    for metric in metrics:
        dtw_temp = DTWAlgorithm(distance_metric=metric)
        rec_temp = SpeechRecognizer(mfcc_extractor, dtw_temp)
        
        # 템플릿 추가
        for label, samples in dataset.items():
            for i in range(3):
                rec_temp.add_template_from_array(label, samples[i])
        
        # 평가
        eval_temp = Evaluator(rec_temp)
        res_temp = eval_temp.evaluate_from_arrays(test_data, verbose=False)
        
        metric_results[metric] = res_temp['accuracy']
        print(f"{metric:12s}: {res_temp['accuracy']*100:5.1f}%")
    
    # ========== 7. 시각화 ==========
    print("\n[7단계] 결과 분석")
    print("-" * 80)
    
    # DTW 정렬 예제
    test_feat = mfcc_extractor.extract_from_array(test_data[0][0])
    template_feat = recognizer.templates[test_data[0][1]][0]
    
    distance, path, acc_cost = dtw.compute_dtw(
        test_feat,
        template_feat,
        return_path=True
    )
    
    print(f"✓ DTW 거리: {distance:.2f}")
    print(f"✓ 정렬 경로 길이: {len(path)}")
    
    # ========== 8. 성능 벤치마크 ==========
    print("\n[8단계] 성능 벤치마크")
    print("-" * 80)
    
    # DTW 메트릭 벤치마크
    print("\nDTW 메트릭 벤치마크:")
    bench_results = Benchmarker.benchmark_dtw_metrics(
        test_feat,
        template_feat,
        metrics=['euclidean', 'manhattan', 'cosine']
    )
    
    # ========== 9. 상세 결과 출력 ==========
    print("\n[9단계] 상세 결과")
    print("-" * 80)
    
    print(f"\n최종 정확도: {results['accuracy']*100:.2f}%")
    print(f"정확히 분류된 샘플: {results['correct']}/{results['total']}")
    
    print("\n클래스별 정확도:")
    for label, acc in results['class_accuracy'].items():
        print(f"  {label:10s}: {acc*100:5.1f}%")
    
    print("\n혼동 행렬:")
    for true_label in sorted(results['confusion_matrix'].keys()):
        print(f"  {true_label:10s}: ", end="")
        for pred_label in sorted(results['confusion_matrix'].keys()):
            count = results['confusion_matrix'][true_label][pred_label]
            print(f"{pred_label[:3]}:{count:2d} ", end="")
        print()
    
    # ========== 10. 템플릿 저장 ==========
    print("\n[10단계] 템플릿 저장")
    print("-" * 80)
    
    # data 디렉토리 생성
    import os
    os.makedirs('data', exist_ok=True)
    
    save_path = 'data/templates.pkl'
    recognizer.save_templates(save_path)
    
    print("\n" + "=" * 80)
    print("실행 완료!")
    print("=" * 80)
    print("\n✅ PyTorch 없이 NumPy만으로 완벽하게 동작합니다!")
    
    return results


if __name__ == "__main__":
    results = main()
    
    print("\n\n사용 가능한 추가 기능:")
    print("-" * 80)
    print("""
    1. 데이터 증강:
       augmenter = FeatureAugmenter()
       noisy = augmenter.add_noise(features)
    
    2. 온라인 인식:
       online = OnlineRecognizer(recognizer)
       result = online.process_chunk(audio_chunk)
    
    3. 앙상블 인식:
       ensemble = EnsembleRecognizer([rec1, rec2, rec3])
       label, confidence = ensemble.recognize_voting(audio_path)
    
    4. 제약이 있는 DTW:
       constrained = ConstrainedDTW(window_size=10)
       distance = constrained.compute_constrained_dtw(x, y)
    
    5. 교차 검증:
       cv_results = evaluator.cross_validate(dataset, k_folds=5)
    """)
