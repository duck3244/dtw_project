export interface HealthResponse {
  status: string;
  backend: string;
  sample_rate: number;
  accel_available: boolean;
}

export interface TemplateInfo {
  label: string;
  count: number;
}

export interface TemplatesResponse {
  labels: TemplateInfo[];
  total: number;
}

export interface AddTemplateResponse {
  label: string;
  template_id: string;
  count: number;
}

export interface LabelDetail {
  label: string;
  template_ids: string[];
}

export interface RecognitionScore {
  label: string;
  distance: number;
}

export interface RecognizeResponse {
  label: string;
  distance: number;
  top_k: RecognitionScore[];
}

export interface EvaluationCase {
  filename: string;
  expected: string | null;
  predicted: string | null;
  distance: number | null;
  latency_ms: number;
  correct: boolean | null;
  error: string | null;
}

export interface LabelStats {
  label: string;
  support: number;
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface EvaluateResponse {
  n_total: number;
  n_scored: number;
  n_correct: number;
  accuracy: number;
  avg_latency_ms: number;
  per_label: LabelStats[];
  cases: EvaluationCase[];
}

const BASE = "/api";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

function extractDetail(text: string, fallback: string): string {
  try {
    const j = JSON.parse(text);
    if (typeof j?.detail === "string") return j.detail;
    if (Array.isArray(j?.detail) && j.detail.length > 0) {
      return j.detail.map((d: { msg?: string }) => d?.msg ?? "").filter(Boolean).join("; ") || fallback;
    }
  } catch {
    // not JSON — fall through
  }
  return fallback;
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    const msg = extractDetail(text, `${res.status} ${res.statusText}`);
    throw new ApiError(res.status, msg);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetch(`${BASE}/health`).then(asJson<HealthResponse>),
  listTemplates: () => fetch(`${BASE}/templates`).then(asJson<TemplatesResponse>),
  addTemplate: (label: string, blob: Blob, filename = "audio.wav") => {
    const fd = new FormData();
    fd.append("label", label);
    fd.append("file", blob, filename);
    return fetch(`${BASE}/templates`, { method: "POST", body: fd }).then(
      asJson<AddTemplateResponse>,
    );
  },
  labelDetail: (label: string) =>
    fetch(`${BASE}/templates/${encodeURIComponent(label)}`).then(asJson<LabelDetail>),
  deleteLabel: (label: string) =>
    fetch(`${BASE}/templates/${encodeURIComponent(label)}`, { method: "DELETE" }).then(
      asJson<void>,
    ),
  deleteTemplate: (label: string, templateId: string) =>
    fetch(
      `${BASE}/templates/${encodeURIComponent(label)}/${encodeURIComponent(templateId)}`,
      { method: "DELETE" },
    ).then(asJson<void>),
  snapshot: () =>
    fetch(`${BASE}/admin/snapshot`, { method: "POST" }).then(asJson<{ path: string }>),
  recognize: (blob: Blob, filename = "audio.wav", topK = 3) => {
    const fd = new FormData();
    fd.append("file", blob, filename);
    return fetch(`${BASE}/recognize?top_k=${topK}`, { method: "POST", body: fd }).then(
      asJson<RecognizeResponse>,
    );
  },
  evaluate: (items: { file: File; expected: string }[]) => {
    const fd = new FormData();
    for (const it of items) {
      fd.append("files", it.file, it.file.name);
      fd.append("expected", it.expected);
    }
    return fetch(`${BASE}/evaluate`, { method: "POST", body: fd }).then(
      asJson<EvaluateResponse>,
    );
  },
};
