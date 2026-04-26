# DTW Speech Recognition — 아키텍처 문서

DTW(Dynamic Time Warping) 알고리즘 기반 음성 인식 시스템의 아키텍처 문서입니다. MFCC 특징을 추출해 등록된 템플릿과의 시퀀스 거리를 비교하여 라벨을 분류합니다.

---

## 1. 전체 구성

```
┌───────────────────────────┐         ┌────────────────────────────────┐
│   Frontend (React + TS)   │         │   Backend (FastAPI + Python)   │
│                           │  HTTP   │                                │
│  ┌─────────────────────┐  │ ◀──────▶│  ┌──────────────────────────┐  │
│  │ TemplateManager     │  │  /api   │  │ Routers (templates,      │  │
│  │ ResultPanel         │  │         │  │  recognize, evaluate)    │  │
│  │ EvaluatePanel       │  │         │  └──────────────────────────┘  │
│  │ RecorderPanel       │  │         │  ┌──────────────────────────┐  │
│  └─────────────────────┘  │         │  │ RecognizerService        │  │
│            ▲              │         │  │  (싱글톤 + RLock)         │  │
│            │ useRecorder  │         │  └──────────────────────────┘  │
│            │ (MediaRec.)  │         │     │            │             │
└────────────┼──────────────┘         │     ▼            ▼             │
             │                        │  MFCC        DTWAlgorithm      │
        microphone                    │  Extractor    (core/accel)     │
                                      │     │            │             │
                                      │     └─────┬──────┘             │
                                      │           ▼                    │
                                      │     SpeechRecognizer           │
                                      │           │                    │
                                      │           ▼                    │
                                      │     TemplateStore              │
                                      │       │         │              │
                                      └───────┼─────────┼──────────────┘
                                              ▼         ▼
                                          SQLite    .npy 블롭
                                        (store.db)  (templates/)
```

---

## 2. 기술 스택

### Backend
- **언어/런타임:** Python 3
- **웹 프레임워크:** FastAPI (Uvicorn ASGI)
- **신호 처리:** Librosa (MFCC), SoundFile (오디오 디코딩)
- **수치 연산:** NumPy, 선택적으로 Numba (JIT)
- **저장소:** SQLite (WAL 모드) + `.npy` 블롭
- **설정:** Pydantic Settings

### Frontend
- **프레임워크:** React 18
- **빌드 도구:** Vite 5
- **언어:** TypeScript 5
- **스타일:** Tailwind CSS 3
- **음성 입력:** Web `MediaRecorder` API

---

## 3. 디렉터리 구조

```
dtw_project/
├── backend/
│   ├── app/                      # FastAPI 애플리케이션 계층
│   │   ├── main.py               # 앱 팩토리 + lifespan(웜업)
│   │   ├── api/                  # REST 라우터
│   │   │   ├── health.py         # GET /health
│   │   │   ├── templates.py      # 템플릿 CRUD
│   │   │   ├── recognize.py      # POST /recognize
│   │   │   ├── evaluate.py       # POST /evaluate (배치)
│   │   │   └── admin.py          # POST /admin/snapshot
│   │   ├── core/config.py        # 환경 설정
│   │   ├── middleware/           # 요청 크기 제한
│   │   ├── schemas/responses.py  # Pydantic 응답 모델
│   │   └── services/
│   │       ├── recognizer_service.py  # 도메인 오케스트레이터
│   │       └── template_store.py      # SQLite + 블롭 영속화
│   ├── src/                      # 도메인 코어 (프레임워크 독립)
│   │   ├── feature_extraction.py # MFCCExtractor
│   │   ├── dtw_algorithm.py      # DTWAlgorithm 파사드
│   │   ├── speech_recognizer.py  # SpeechRecognizer
│   │   ├── evaluation.py         # 평가 메트릭
│   │   └── backends/             # 교체 가능한 DTW 백엔드
│   │       ├── core/dtw.py       # 순수 NumPy 구현
│   │       └── accel/dtw.py      # Numba JIT 구현
│   ├── data/                     # 런타임 데이터 (gitignored)
│   │   ├── store.db              # SQLite 메타데이터
│   │   ├── templates/*.npy       # MFCC 특징 블롭
│   │   └── backups/              # 스냅샷 tar.gz
│   ├── tests/                    # pytest 테스트
│   └── run_api.sh
│
├── frontend/
│   ├── src/
│   │   ├── main.tsx              # 진입점
│   │   ├── App.tsx               # 루트 셸
│   │   ├── api.ts                # API 클라이언트 + 타입
│   │   ├── components/
│   │   │   ├── TemplateManager.tsx
│   │   │   ├── ResultPanel.tsx
│   │   │   ├── EvaluatePanel.tsx
│   │   │   ├── RecorderPanel.tsx
│   │   │   └── Notification.tsx
│   │   └── hooks/useRecorder.ts  # MediaRecorder 훅
│   ├── vite.config.ts            # /api → :8000 프록시
│   └── package.json
│
└── docs/                         # 본 문서
```

---

## 4. 백엔드 구성 요소

### 4.1 계층 구조

| 계층 | 위치 | 책임 |
|------|------|------|
| **API** | `app/api/` | HTTP 라우팅, 검증, 응답 직렬화 |
| **Service** | `app/services/` | 트랜잭션 경계, 동시성 제어, 도메인 오케스트레이션 |
| **Domain** | `src/` | DTW·MFCC·인식 알고리즘 (FastAPI에 비종속) |
| **Persistence** | `app/services/template_store.py` | SQLite + `.npy` 블롭 I/O |

도메인 코어(`src/`)는 FastAPI에 의존하지 않으므로 CLI/배치/노트북에서도 재사용 가능합니다.

### 4.2 핵심 클래스

- **`MFCCExtractor`** — 오디오 신호 → `(T, n_mfcc)` 행렬 (Librosa 기반).
- **`DTWAlgorithm`** — `core`(NumPy) / `accel`(Numba) 백엔드를 추상화하는 파사드. `compute_dtw`, `compute_dtw_normalized` 제공.
- **`SpeechRecognizer`** — 라벨별 템플릿 컬렉션을 관리하고 입력 특징과의 최단 거리를 계산. 집계 정책(`min` / `mean` / `knn`) 선택 가능.
- **`TemplateStore`** — SQLite WAL 모드. 메타데이터(라벨/경로/형상)는 DB에, 특징 행렬은 `.npy` 블롭으로 분리 저장.
- **`RecognizerService`** — 위 4개를 묶는 싱글톤. `RLock`으로 쓰기를 직렬화하여 인메모리 캐시와 DB의 일관성을 보장.

### 4.3 API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/health` | 백엔드명·샘플레이트·가속 가용성 |
| GET | `/api/templates` | 라벨 목록과 템플릿 수 |
| GET | `/api/templates/{label}` | 특정 라벨의 템플릿 ID 목록 |
| POST | `/api/templates` | 라벨 + 오디오 업로드 (multipart) |
| DELETE | `/api/templates/{label}` | 라벨 전체 삭제 |
| DELETE | `/api/templates/{label}/{id}` | 단일 템플릿 삭제 |
| POST | `/api/recognize?top_k=N` | 단일 인식 + Top-K |
| POST | `/api/evaluate` | 배치 평가, 라벨별 P/R/F1 |
| POST | `/api/admin/snapshot` | DB+블롭 tar.gz 스냅샷 |

### 4.4 영속화 모델

```
SQLite: templates(id PK, label, blob_path, frames, n_mfcc, created_at)
              │
              └─ 인덱스: label
파일시스템: data/templates/{uuid}.npy   # float32, shape=(T, 13)
백업:        data/backups/snapshot-YYYYMMDDTHHMMSS.tar.gz
```

특징 행렬을 DB가 아닌 파일로 분리한 이유: (1) 행렬은 크고 불변이라 트랜잭션이 불필요, (2) 외부 도구로 그대로 로드/검사 가능, (3) DB가 가벼워져 백업·복구가 빨라짐.

### 4.5 부팅 시 웜업

`main.py`의 lifespan에서 다음을 수행해 첫 요청 지연을 제거합니다.
1. 무음 신호로 MFCC 1회 추출 (Librosa 내부 캐시 채움, ~600ms)
2. 더미 행렬로 DTW 1회 호출 (Numba JIT 컴파일, ~100ms)

---

## 5. 프론트엔드 구성

### 5.1 화면 구성

`App.tsx`는 세 개의 섹션을 순서대로 렌더링하며, 글로벌 상태로 `templates`, `busy`, `backend`를 관리합니다.

| 섹션 | 컴포넌트 | 역할 |
|------|----------|------|
| 1 | `TemplateManager` | 라벨 입력 + 녹음/업로드 → 템플릿 등록, 라벨별 삭제 |
| 2 | `ResultPanel` | 단일 인식, Top-K 결과 표시 |
| 3 | `EvaluatePanel` | 다중 파일/폴더 업로드 → 정확도·P/R/F1·케이스별 결과 |

`RecorderPanel`은 1·2·3 섹션이 공유하는 입력 컴포넌트로, 마이크 녹음과 파일 업로드를 모두 지원합니다. `useRecorder` 훅이 `MediaRecorder` 라이프사이클을 캡슐화합니다.

### 5.2 라벨 자동 추정 (EvaluatePanel)

배치 평가 시 정답 라벨을 자동으로 채워 입력 부담을 줄입니다.
1. **폴더 업로드** (`webkitdirectory`): 부모 디렉터리명 → 라벨 (`down/abc.wav` → `down`)
2. **파일명 패턴**: `label_*`, `label-*`, 또는 등록된 라벨과 정확히 일치
3. **선두 단어**: `hello_001.wav` → `hello`

### 5.3 API 클라이언트

`src/api.ts`는 백엔드 응답 타입과 1:1 대응하는 TypeScript 인터페이스, fetch 래퍼, `ApiError` 클래스를 제공합니다. Pydantic 검증 오류는 `detail` 필드를 평탄화해 사람이 읽을 수 있는 메시지로 변환합니다.

### 5.4 빌드/배포

- **개발**: `npm run dev` (Vite, :5173). `vite.config.ts`가 `/api/*`를 `http://localhost:8000`으로 프록시.
- **프로덕션**: `npm run build` → `dist/` 정적 산출물. CDN 또는 동일 오리진 리버스 프록시 뒤 배포.

---

## 6. 통신 프로토콜

| 항목 | 값 |
|------|----|
| 전송 방식 | HTTP/REST (multipart/form-data, JSON) |
| 인증 | 없음 (로컬 도구) |
| CORS | `CORS_ORIGINS` env, 기본값 `http://localhost:5173` |
| 오디오 포맷 | `.wav`, `.flac`, `.ogg`, `.webm`, `.mp3`, `.m4a` |
| 파일 크기 | 단건 10MB, 배치 50MB / 200파일 |

---

## 7. DTW 백엔드 전략

| 백엔드 | 구현 | 용도 |
|--------|------|------|
| `core` | 순수 NumPy | 의존성 최소, 작은 행렬에 적합 |
| `accel` | Numba JIT | 50~250× 가속, 대량 템플릿 환경 권장 |
| `auto` | accel 가능 시 accel, 아니면 core | 기본값 |

선택은 환경변수 `DTW_BACKEND`로 제어합니다. accel은 지연 임포트되어 Numba 미설치 환경에서도 앱이 정상 기동됩니다.

---

## 8. 점수 집계 정책

`SpeechRecognizer.score_aggregation`로 전환합니다.

| 정책 | 계산 | 특성 |
|------|------|------|
| `min` (기본) | 라벨 내 템플릿들과의 거리 중 최솟값 | 이상치에 강함 |
| `mean` | 거리 평균 | 레거시 호환 |
| `knn` | 전체 템플릿 중 k-최근접 다수결 투표 | 라벨 불균형 시 유용 |

또한 `normalize=True`이면 `distance / path_length`로 길이 영향을 보정합니다.

---

## 9. 동시성 모델

- 모든 쓰기 경로(`add_template`, `remove_*`)는 `RecognizerService`의 `RLock`으로 직렬화됩니다.
- 읽기 경로(`recognize`, `list`)는 동일 락 아래에서 인메모리 dict와 SQLite를 일관되게 조회합니다.
- SQLite는 WAL 모드로 동시 read를 허용하지만, 본 시스템은 단일 프로세스 가정이므로 락은 인메모리 캐시와 DB 동기화에 초점이 있습니다.

---

## 10. 환경 변수

`backend/.env.example` 참조.

| 변수 | 의미 | 기본값 |
|------|------|--------|
| `CORS_ORIGINS` | 허용 오리진 (콤마 구분) | `http://localhost:5173` |
| `SAMPLE_RATE` | 입력 신호 리샘플링 목표 | `16000` |
| `DTW_BACKEND` | `auto` / `core` / `accel` | `auto` |
| `TEMPLATES_DIR` | 블롭 디렉터리 | `data/templates` |
| `STORE_DB_PATH` | SQLite 경로 | `data/store.db` |
| `BACKUP_DIR` | 스냅샷 디렉터리 | `data/backups` |

---

## 11. 테스트

- `backend/tests/test_api.py` — 엔드포인트 e2e (등록 → 인식 → 평가 → 삭제)
- `backend/tests/test_recognizer.py` — `SpeechRecognizer` 단위 테스트
- `backend/tests/test_dtw_backends.py` — `core` ↔ `accel` 결과 동등성 검증

```bash
cd backend && python -m pytest tests/ -v
```

---

## 12. 주요 설계 결정

1. **도메인-프레임워크 분리** — `src/`는 FastAPI에 비종속이라 노트북·CLI에서도 임포트 가능.
2. **메타데이터/특징 분리 저장** — DB는 가볍게, 큰 행렬은 `.npy`로.
3. **싱글톤 서비스 + 락** — 단일 프로세스에서 인메모리 캐시와 디스크 상태의 일관성 보장.
4. **교체 가능한 DTW 백엔드** — 동일 인터페이스, 환경에 맞는 구현 선택.
5. **부팅 웜업** — Librosa·Numba의 첫 호출 지연을 사용자에게 노출하지 않음.
6. **라벨 자동 추정** — 평가 UX를 단순화해 데이터셋 단위 검증을 빠르게.
