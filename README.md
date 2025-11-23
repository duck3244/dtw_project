# DTW 기반 음성 인식 시스템

## ✨ 주요 특징

- ✅ **순수 NumPy 구현** - 가볍고 빠른 설치
- ✅ **완전한 기능** - 원본과 동일한 모든 기능 지원
- ✅ **즉시 실행 가능** - 복잡한 설정 불필요

---

## 🚀 빠른 시작 (3단계)

### 1. 설치 (1분)

```bash
pip install numpy librosa matplotlib scipy seaborn soundfile
```

### 2. 프로젝트 다운로드

```bash
cd dtw_project
```

### 3. 실행

```bash
python examples/complete_example.py
```

---

## 📦 프로젝트 구조

```
dtw_project/
├── src/                         # 핵심 소스 코드 (NumPy 전용)
│   ├── feature_extraction.py    # MFCC 특징 추출
│   ├── dtw_algorithm.py         # DTW 알고리즘
│   ├── speech_recognizer.py     # 음성 인식기
│   ├── data_processing.py       # 데이터 처리
│   ├── visualization.py         # 시각화
│   └── evaluation.py            # 평가
│
├── examples/
│   └── complete_example.py      # 완전한 통합 예제
│
├── requirements.txt             # NumPy 전용 의존성
└── README.md                    # 이 파일
```

---

## 💻 사용 예제

### 기본 사용

```python
import sys
sys.path.append('./src')

from feature_extraction import MFCCExtractor
from dtw_algorithm import DTWAlgorithm
from speech_recognizer import SpeechRecognizer
from data_processing import SyntheticSpeechGenerator

# 합성 데이터 생성 (실제 오디오 없이도 테스트 가능!)
generator = SyntheticSpeechGenerator(sr=16000)
hello = generator.generate_speech_sample('chirp', duration=0.8)
goodbye = generator.generate_speech_sample('formant', duration=1.0)

# MFCC 추출
extractor = MFCCExtractor(sr=16000, n_mfcc=13)

# DTW 초기화
dtw = DTWAlgorithm(distance_metric='euclidean')

# 인식기 생성
recognizer = SpeechRecognizer(extractor, dtw)

# 템플릿 등록
recognizer.add_template_from_array('hello', hello)
recognizer.add_template_from_array('goodbye', goodbye)

# 테스트
test_audio = generator.generate_speech_sample('chirp', duration=0.85)
label, distance = recognizer.recognize_from_array(test_audio)
print(f"인식 결과: {label} (거리: {distance:.2f})")
```

### 실제 오디오 파일 사용

```python
# WAV 파일로 템플릿 추가
recognizer.add_template('hello', 'audio/hello_1.wav')
recognizer.add_template('hello', 'audio/hello_2.wav')
recognizer.add_template('goodbye', 'audio/goodbye.wav')

# 인식
label, distance = recognizer.recognize('audio/test.wav')
print(f"인식 결과: {label}")
```

---

## 🔧 주요 기능

### 1. 특징 추출 (feature_extraction.py)
```python
extractor = MFCCExtractor(sr=16000, n_mfcc=13, include_deltas=True)
features = extractor.extract('audio.wav')  # (T, 39) NumPy array
normalized = extractor.normalize(features)
```

### 2. DTW 알고리즘 (dtw_algorithm.py)
```python
dtw = DTWAlgorithm(distance_metric='euclidean')
distance, path, acc_cost = dtw.compute_dtw(x, y, return_path=True)
```

### 3. 음성 인식 (speech_recognizer.py)
```python
recognizer = SpeechRecognizer(extractor, dtw)
recognizer.add_template('word', 'word.wav')
label, distance = recognizer.recognize('test.wav')
```

### 4. 데이터 생성 (data_processing.py)
```python
generator = SyntheticSpeechGenerator()
signal = generator.generate_speech_sample('chirp')
```

### 5. 시각화 (visualization.py)
```python
from visualization import DTWVisualizer
viz = DTWVisualizer()
viz.plot_dtw_alignment(x, y, path, acc_cost)
```

### 6. 평가 (evaluation.py)
```python
from evaluation import Evaluator
evaluator = Evaluator(recognizer)
results = evaluator.evaluate(test_data)
print(f"정확도: {results['accuracy']:.2%}")
```

---

## 📚 실제 응용 예시

### 1. 음성 명령 인식
```python
# "켜기", "끄기", "음소거" 명령 인식
commands = ['turn_on', 'turn_off', 'mute']
for cmd in commands:
    for i in range(3):
        recognizer.add_template(cmd, f'commands/{cmd}_{i}.wav')

command, _ = recognizer.recognize('user_command.wav')
execute_command(command)
```

### 2. 화자 검증
```python
# 특정 사용자 음성 등록
for i in range(5):
    recognizer.add_template('user_001', f'enroll_{i}.wav')

# 검증
label, distance = recognizer.recognize('test.wav')
verified = (label == 'user_001' and distance < 50.0)
```

### 3. 실시간 키워드 감지
```python
from speech_recognizer import OnlineRecognizer

online = OnlineRecognizer(recognizer, buffer_size=16000)

for audio_chunk in stream:
    result = online.process_chunk(audio_chunk)
    if result and result[0] == 'wake_word':
        print("웨이크워드 감지!")
```

---

## 📖 참고 자료

- [Dynamic Time Warping - Wikipedia](https://en.wikipedia.org/wiki/Dynamic_time_warping)
- [MFCC 설명](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- [Librosa Documentation](https://librosa.org/)
- [NumPy Documentation](https://numpy.org/)
- [Real Dataset](https://github.com/crawles/dtw)

---
