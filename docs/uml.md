# DTW Speech Recognition — UML 다이어그램

본 문서는 시스템의 구조와 동작을 UML로 표현합니다. 모든 다이어그램은 GitHub/IDE에서 직접 렌더링되는 [Mermaid](https://mermaid.js.org/) 문법으로 작성되었습니다.

---

## 1. 컴포넌트 다이어그램

시스템 전체 구성 요소와 의존 관계.

```mermaid
graph TB
    subgraph Browser["브라우저 (Frontend)"]
        UI["React 컴포넌트<br/>TemplateManager · ResultPanel · EvaluatePanel"]
        Recorder["RecorderPanel + useRecorder"]
        APIClient["api.ts (fetch 래퍼)"]
        UI --> Recorder
        UI --> APIClient
    end

    subgraph Server["서버 (Backend, FastAPI)"]
        Routers["라우터<br/>/templates · /recognize · /evaluate · /admin"]
        Service["RecognizerService (싱글톤)"]
        MFCC["MFCCExtractor"]
        DTW["DTWAlgorithm"]
        SR["SpeechRecognizer"]
        Store["TemplateStore"]
        Backends["DTW Backends<br/>core (NumPy) | accel (Numba)"]

        Routers --> Service
        Service --> MFCC
        Service --> SR
        SR --> DTW
        DTW --> Backends
        Service --> Store
    end

    subgraph Disk["영속 저장소"]
        SQLite[("SQLite<br/>store.db")]
        Blobs[("npy 블롭<br/>templates/*.npy")]
        Backup[("백업<br/>backups/*.tar.gz")]
    end

    APIClient -- "HTTP /api/*" --> Routers
    Recorder -- "MediaRecorder" --> MicHW(("🎤 마이크"))
    Store --> SQLite
    Store --> Blobs
    Store --> Backup
```

---

## 2. 클래스 다이어그램 (백엔드 도메인)

핵심 도메인 클래스와 관계.

```mermaid
classDiagram
    class MFCCExtractor {
        +int sr
        +int n_mfcc
        +int n_fft
        +int hop_length
        +int n_mels
        +extract(path) ndarray
        +extract_from_array(signal) ndarray
    }

    class DTWBackend {
        <<interface>>
        +compute(x, y, band) tuple
    }

    class CoreBackend {
        +compute(x, y, band) tuple
    }

    class AccelBackend {
        +compute(x, y, band) tuple
    }

    class DTWAlgorithm {
        -DTWBackend _backend
        +str distance_metric
        +int band
        +compute_dtw(x, y, return_path, band) tuple
        +compute_dtw_normalized(x, y, band) float
    }

    class SpeechRecognizer {
        -MFCCExtractor extractor
        -DTWAlgorithm dtw
        +dict templates
        +dict template_metadata
        +bool normalize
        +str score_aggregation
        +int knn_k
        +add_template(label, features)
        +recognize_from_array(features, top_k) tuple
        +remove_template(label, idx)
        +get_template_count(label) int
    }

    class TemplateStore {
        -Path db_path
        -Path blob_dir
        -Connection _conn
        +add(label, features) str
        +delete_id(template_id)
        +delete_label(label)
        +iter_all() Iterator
        +list_counts() dict
        +snapshot() Path
    }

    class RecognizerService {
        -SpeechRecognizer _recognizer
        -TemplateStore _store
        -dict _ids
        -RLock _lock
        +add_template(label, audio_bytes) tuple
        +recognize(audio_bytes, top_k) RecognizeResult
        +list_templates() list
        +remove_label(label)
        +remove_template(label, template_id)
        +snapshot() Path
    }

    DTWBackend <|.. CoreBackend
    DTWBackend <|.. AccelBackend
    DTWAlgorithm o--> DTWBackend : delegates
    SpeechRecognizer o--> MFCCExtractor
    SpeechRecognizer o--> DTWAlgorithm
    RecognizerService o--> SpeechRecognizer
    RecognizerService o--> TemplateStore
    RecognizerService o--> MFCCExtractor
```

---

## 3. 클래스 다이어그램 (응답 스키마)

Pydantic 응답 모델.

```mermaid
classDiagram
    class HealthResponse {
        +str status
        +str backend
        +int sample_rate
        +bool accel_available
    }

    class TemplateInfo {
        +str label
        +int count
    }

    class TemplatesResponse {
        +List~TemplateInfo~ labels
        +int total
    }

    class LabelDetail {
        +str label
        +List~str~ template_ids
    }

    class AddTemplateResponse {
        +str label
        +str template_id
        +int count
    }

    class RecognitionScore {
        +str label
        +float distance
    }

    class RecognizeResponse {
        +str label
        +float distance
        +List~RecognitionScore~ top_k
    }

    class EvaluationCase {
        +str filename
        +str expected
        +str predicted
        +float distance
        +float latency_ms
        +bool correct
        +str error
    }

    class LabelStats {
        +str label
        +int support
        +int tp
        +int fp
        +int fn
        +float precision
        +float recall
        +float f1
    }

    class EvaluateResponse {
        +int n_total
        +int n_scored
        +int n_correct
        +float accuracy
        +float avg_latency_ms
        +List~LabelStats~ per_label
        +List~EvaluationCase~ cases
    }

    TemplatesResponse o--> TemplateInfo
    RecognizeResponse o--> RecognitionScore
    EvaluateResponse o--> LabelStats
    EvaluateResponse o--> EvaluationCase
```

---

## 4. 시퀀스 다이어그램 — 템플릿 등록

`POST /api/templates` 흐름.

```mermaid
sequenceDiagram
    actor User
    participant TM as TemplateManager
    participant Rec as RecorderPanel
    participant API as api.ts
    participant R as templates router
    participant S as RecognizerService
    participant E as MFCCExtractor
    participant ST as TemplateStore
    participant SR as SpeechRecognizer
    participant DB as SQLite + npy

    User->>TM: 라벨 입력 + 녹음 시작
    TM->>Rec: state="recording"
    Rec->>Rec: MediaRecorder.start()
    User->>Rec: 정지
    Rec-->>TM: Blob (audio)
    User->>TM: 등록 클릭
    TM->>API: addTemplate(label, blob)
    API->>R: POST /api/templates (multipart)
    R->>S: add_template(label, bytes)
    activate S
    S->>S: _decode(bytes) → signal
    S->>E: extract_from_array(signal)
    E-->>S: features (T × 13)
    S->>ST: add(label, features)
    ST->>DB: save .npy + INSERT row
    ST-->>S: template_id
    S->>SR: templates[label].append(features)
    S-->>R: (template_id, count)
    deactivate S
    R-->>API: 201 AddTemplateResponse
    API-->>TM: {label, template_id, count}
    TM->>API: listTemplates()
    API-->>TM: 갱신된 라벨 목록
    TM-->>User: 알림 + 목록 갱신
```

---

## 5. 시퀀스 다이어그램 — 단일 인식

`POST /api/recognize` 흐름.

```mermaid
sequenceDiagram
    actor User
    participant RP as ResultPanel
    participant API as api.ts
    participant R as recognize router
    participant S as RecognizerService
    participant E as MFCCExtractor
    participant SR as SpeechRecognizer
    participant DTW as DTWAlgorithm

    User->>RP: 녹음/업로드 후 인식 요청
    RP->>API: recognize(blob, topK=3)
    API->>R: POST /api/recognize?top_k=3
    R->>S: recognize(bytes, top_k=3)
    activate S
    S->>S: _decode(bytes)
    S->>E: extract_from_array(signal)
    E-->>S: query_features
    S->>SR: recognize_from_array(query, top_k=3)
    activate SR
    loop 각 라벨의 템플릿
        SR->>DTW: compute_dtw_normalized(query, template_i)
        DTW-->>SR: distance_i
    end
    SR->>SR: 라벨별 집계 (min / mean / knn)
    SR->>SR: 거리 오름차순 정렬
    SR-->>S: (best_label, best_distance, top_k)
    deactivate SR
    S-->>R: RecognizeResult
    deactivate S
    R-->>API: 200 RecognizeResponse
    API-->>RP: {label, distance, top_k}
    RP-->>User: Top-K 카드 렌더링
```

---

## 6. 시퀀스 다이어그램 — 배치 평가

`POST /api/evaluate` 흐름.

```mermaid
sequenceDiagram
    actor User
    participant EP as EvaluatePanel
    participant API as api.ts
    participant R as evaluate router
    participant S as RecognizerService

    User->>EP: 폴더/파일 선택
    EP->>EP: 파일명·디렉터리에서 라벨 자동 추정
    User->>EP: (필요 시) 라벨 수정 → 평가 클릭
    EP->>API: evaluate([{file, expected}, ...])
    API->>R: POST /api/evaluate (multipart)
    activate R
    loop 각 파일
        R->>R: 확장자/크기 검증
        R->>S: recognize(bytes)
        S-->>R: (label, distance, latency_ms)
        R->>R: case 기록 (정답 비교)
    end
    R->>R: _summarize(cases)<br/>per_label P/R/F1, accuracy, avg_latency
    R-->>API: 200 EvaluateResponse
    deactivate R
    API-->>EP: {accuracy, per_label, cases}
    EP-->>User: 메트릭 + 라벨별 표 + 케이스 표
```

---

## 7. 상태 다이어그램 — 마이크 녹음 (`useRecorder`)

```mermaid
stateDiagram-v2
    [*] --> idle
    idle --> recording: start() 권한 허용
    idle --> error: getUserMedia 실패
    recording --> stopped: stop()
    recording --> error: MediaRecorder onerror
    stopped --> recording: 재녹음 (start)
    stopped --> idle: reset()
    error --> idle: reset()
    stopped --> [*]: 컴포넌트 언마운트
```

---

## 8. 활동 다이어그램 — DTW 백엔드 선택

`backends/__init__.py`의 `get_backend("auto")` 동작.

```mermaid
flowchart TD
    Start([get_backend 호출]) --> Q1{backend 인자}
    Q1 -- "core" --> Core[CoreBackend 반환]
    Q1 -- "accel" --> TryAccel{numba<br/>임포트 가능?}
    Q1 -- "auto" --> TryAccel
    TryAccel -- 성공 --> Accel[AccelBackend 반환]
    TryAccel -- 실패 --> Q2{auto?}
    Q2 -- 예 --> Core
    Q2 -- 아니오 --> Err([ImportError 발생])
    Core --> End([반환])
    Accel --> End
```

---

## 9. ER 다이어그램 — 영속 저장소

```mermaid
erDiagram
    TEMPLATES {
        TEXT id PK "UUID hex"
        TEXT label "인덱스됨"
        TEXT blob_path "templates_dir 기준 상대경로"
        INTEGER frames "시간 프레임 수"
        INTEGER n_mfcc "특징 차원 (보통 13)"
        REAL created_at "유닉스 타임스탬프"
    }

    NPY_BLOB {
        FILE blob_path PK "data/templates/{uuid}.npy"
        BINARY data "float32 ndarray (frames × n_mfcc)"
    }

    TEMPLATES ||--|| NPY_BLOB : "blob_path 참조"
```

> 메타데이터는 SQLite, 특징 행렬은 별도 `.npy` 파일에 저장됩니다. `blob_path` 컬럼이 둘을 묶는 외래 참조 역할을 합니다 (FK 제약은 없음 — 파일시스템 참조).

---

## 10. 배포 다이어그램

```mermaid
graph LR
    subgraph Client["사용자 머신"]
        Browser["브라우저<br/>(React SPA)"]
        Mic[("🎤 마이크")]
    end

    subgraph WebTier["웹 계층"]
        Static["정적 파일<br/>(dist/)"]
        Proxy["Nginx / Vite Dev<br/>리버스 프록시"]
    end

    subgraph AppTier["애플리케이션 계층"]
        Uvicorn["Uvicorn ASGI<br/>app.main:app"]
        FastAPI["FastAPI 앱"]
        Uvicorn --> FastAPI
    end

    subgraph DataTier["데이터 계층"]
        SQLite[("store.db<br/>(SQLite WAL)")]
        BlobDir[("templates/<br/>(.npy 파일)")]
        BackupDir[("backups/<br/>(tar.gz)")]
    end

    Browser -->|"HTTPS"| Proxy
    Mic -. MediaStream .-> Browser
    Proxy --> Static
    Proxy -->|"/api/*"| Uvicorn
    FastAPI --> SQLite
    FastAPI --> BlobDir
    FastAPI --> BackupDir
```

---

## 11. 패키지 의존 다이어그램

```mermaid
graph TD
    main["app.main"] --> api_health["app.api.health"]
    main --> api_tpl["app.api.templates"]
    main --> api_rec["app.api.recognize"]
    main --> api_eval["app.api.evaluate"]
    main --> api_admin["app.api.admin"]
    main --> svc["app.services.recognizer_service"]

    api_tpl --> svc
    api_rec --> svc
    api_eval --> svc
    api_admin --> svc

    svc --> store["app.services.template_store"]
    svc --> mfcc["src.feature_extraction"]
    svc --> recog["src.speech_recognizer"]

    recog --> mfcc
    recog --> dtw["src.dtw_algorithm"]
    dtw --> backends["src.backends"]
    backends --> core_be["src.backends.core.dtw"]
    backends --> accel_be["src.backends.accel.dtw"]

    api_health --> svc
    main --> cfg["app.core.config"]
    svc --> cfg
    store --> cfg
```

---

## 부록: Mermaid 렌더링 안내

- **GitHub**: `.md` 미리보기에서 자동 렌더링됩니다.
- **VSCode / JetBrains**: Mermaid 플러그인 또는 내장 미리보기 사용.
- **로컬 PNG/SVG 추출**: [`mermaid-cli`](https://github.com/mermaid-js/mermaid-cli) (`mmdc -i uml.md -o uml.png`).
