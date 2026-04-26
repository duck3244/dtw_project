import sys
sys.path.append('./src')

from feature_extraction import MFCCExtractor
from dtw_algorithm import DTWAlgorithm
from speech_recognizer import SpeechRecognizer

print("=" * 70)
print("실제 오디오 파일을 사용한 음성 인식")
print("=" * 70)

# 1. 초기화
extractor = MFCCExtractor(sr=16000, n_mfcc=13)
dtw = DTWAlgorithm(distance_metric='euclidean')
recognizer = SpeechRecognizer(extractor, dtw)

# 2. 템플릿 등록 (각 단어/문장별로)
print("\n[템플릿 등록]")

# "mothers_milk" 단어 템플릿들
recognizer.add_template('mothers_milk', 'audio/chris_mothers_milk_word.m4a')
recognizer.add_template('mothers_milk', 'audio/elias_mothers_milk_word.m4a')
recognizer.add_template('mothers_milk', 'audio/yaoquan_mothers_milk_word.m4a')

# "warping" 템플릿들
recognizer.add_template('warping', 'audio/chris_warping.m4a')
recognizer.add_template('warping', 'audio/elias_warping.m4a')

# "follow" 템플릿
recognizer.add_template('follow', 'audio/chris_follow.m4a')

# "algorithm" 템플릿
recognizer.add_template('algorithm', 'audio/elias_algorithm.m4a')

print(f"\n총 {sum(recognizer.get_template_count().values())}개 템플릿 등록 완료")
for label, count in recognizer.get_template_count().items():
    print(f"  - {label}: {count}개")

# 3. 테스트 - 다른 발음으로 인식해보기
print("\n" + "=" * 70)
print("테스트 - 단어 인식")
print("=" * 70)

test_files = [
    ('audio/chris_mothers_milk_word_slow.m4a', 'mothers_milk'),
    ('audio/chris_warping_cut.m4a', 'warping'),
    ('audio/elias_algorithm_sentence.m4a', 'algorithm'),
]

for audio_file, expected in test_files:
    try:
        label, distance = recognizer.recognize(audio_file)
        status = "✓" if label == expected else "✗"
        print(f"{status} {audio_file.split('/')[-1]:40s} → {label:15s} (거리: {distance:.2f})")
    except Exception as e:
        print(f"✗ {audio_file.split('/')[-1]:40s} → 오류: {e}")

print("\n" + "=" * 70)