from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    backend: str
    sample_rate: int
    accel_available: bool


class TemplateInfo(BaseModel):
    label: str
    count: int


class TemplatesResponse(BaseModel):
    labels: list[TemplateInfo]
    total: int


class LabelDetail(BaseModel):
    label: str
    template_ids: list[str]


class AddTemplateResponse(BaseModel):
    label: str
    template_id: str
    count: int = Field(..., description="number of templates registered for this label")


class RecognitionScore(BaseModel):
    label: str
    distance: float


class RecognizeResponse(BaseModel):
    label: str
    distance: float
    top_k: list[RecognitionScore]


class SnapshotResponse(BaseModel):
    path: str


class EvaluationCase(BaseModel):
    filename: str
    expected: str | None
    predicted: str | None = None
    distance: float | None = None
    latency_ms: float
    correct: bool | None = None
    error: str | None = None


class LabelStats(BaseModel):
    label: str
    support: int = Field(..., description="number of expected==label cases")
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


class EvaluateResponse(BaseModel):
    n_total: int
    n_scored: int = Field(..., description="cases where expected was provided and recognition succeeded")
    n_correct: int
    accuracy: float
    avg_latency_ms: float
    per_label: list[LabelStats]
    cases: list[EvaluationCase]
