import { useMemo, useState } from "react";
import { api, type EvaluateResponse, type TemplatesResponse } from "../api";
import { useNotify } from "./Notification";

interface Props {
  templates: TemplatesResponse | null;
  busy: boolean;
  setBusy: (b: boolean) => void;
}

interface Item {
  file: File;
  expected: string;
}

function detectLabel(file: File, knownLabels: string[]): string {
  // 1) Folder upload (webkitdirectory): file.webkitRelativePath looks like
  //    "mini_speech_commands/down/abc.wav" — parent dir is the label.
  const rel =
    (file as File & { webkitRelativePath?: string }).webkitRelativePath || "";
  if (rel.includes("/")) {
    const parts = rel.split("/").filter(Boolean);
    if (parts.length >= 2) return parts[parts.length - 2];
  }

  // 2) Filename pattern: "<label>_*", "<label>-*", "<label>.*", or exact match
  const base = file.name.replace(/\.[^.]+$/, "");
  for (const lbl of knownLabels) {
    if (
      base === lbl ||
      base.startsWith(`${lbl}_`) ||
      base.startsWith(`${lbl}-`) ||
      base.startsWith(`${lbl}.`)
    ) {
      return lbl;
    }
  }

  // 3) Leading word: "hello_001.wav" → "hello"
  const m = base.match(/^([a-zA-Z][a-zA-Z0-9]*)/);
  return m ? m[1] : "";
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

export default function EvaluatePanel({ templates, busy, setBusy }: Props) {
  const [items, setItems] = useState<Item[]>([]);
  const [result, setResult] = useState<EvaluateResponse | null>(null);
  const notify = useNotify();
  const noTemplates = !templates || templates.total === 0;

  const knownLabels = useMemo(
    () => (templates ? templates.labels.map((l) => l.label) : []),
    [templates],
  );

  const onFiles = (filelist: FileList | null) => {
    if (!filelist) return;
    const next: Item[] = [];
    for (const f of Array.from(filelist)) {
      // Skip non-audio files when a folder is dropped (e.g., README.md, .DS_Store)
      const ext = f.name.toLowerCase().match(/\.[^.]+$/)?.[0] ?? "";
      if (![".wav", ".flac", ".ogg", ".webm", ".mp3", ".m4a"].includes(ext)) {
        continue;
      }
      next.push({ file: f, expected: detectLabel(f, knownLabels) });
    }
    setItems((prev) => [...prev, ...next]);
  };

  const updateLabel = (i: number, expected: string) =>
    setItems((prev) => prev.map((it, idx) => (idx === i ? { ...it, expected } : it)));

  const removeAt = (i: number) =>
    setItems((prev) => prev.filter((_, idx) => idx !== i));

  const clearAll = () => {
    setItems([]);
    setResult(null);
  };

  const submit = async () => {
    if (items.length === 0) return;
    setBusy(true);
    setResult(null);
    try {
      setResult(await api.evaluate(items));
      notify.success(`평가 완료: ${items.length}개 파일`);
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="space-y-4 rounded-lg border bg-white p-4 shadow-sm">
      <header className="flex items-center justify-between">
        <h2 className="font-semibold">3. 성능 평가</h2>
        {noTemplates && (
          <span className="text-xs text-amber-600">먼저 템플릿을 등록하세요</span>
        )}
      </header>

      <div className="flex flex-wrap items-center gap-2">
        <label className="cursor-pointer rounded-md border border-slate-300 bg-white px-3 py-2 text-sm hover:bg-slate-50">
          파일 추가 (다중 선택)
          <input
            type="file"
            multiple
            accept="audio/*"
            className="hidden"
            disabled={busy}
            onChange={(e) => {
              onFiles(e.target.files);
              e.target.value = "";
            }}
          />
        </label>
        <label className="cursor-pointer rounded-md border border-slate-300 bg-white px-3 py-2 text-sm hover:bg-slate-50">
          폴더 추가
          <input
            type="file"
            multiple
            // webkitdirectory is a non-standard attribute supported by Chrome/Edge/Safari/Firefox
            // It exposes file.webkitRelativePath ("dirname/file.wav") which we use as the expected label.
            {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
            className="hidden"
            disabled={busy}
            onChange={(e) => {
              onFiles(e.target.files);
              e.target.value = "";
            }}
          />
        </label>
        {items.length > 0 && (
          <>
            <button
              onClick={submit}
              disabled={busy || noTemplates}
              className="rounded-md bg-emerald-600 px-3 py-2 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
            >
              평가 실행 ({items.length}개)
            </button>
            <button
              onClick={clearAll}
              disabled={busy}
              className="rounded-md border px-3 py-2 text-sm text-slate-600 hover:bg-slate-50 disabled:opacity-50"
            >
              모두 비우기
            </button>
          </>
        )}
        <span className="text-xs text-slate-500">
          폴더 업로드 시 부모 폴더명을 expected 라벨로 자동 인식
          (예: <code>down/abc.wav</code> → <code>down</code>). 직접 수정 가능, 빈 라벨은 정확도 계산 제외.
        </span>
      </div>

      {items.length > 0 && (
        <div className="max-h-72 overflow-auto rounded-md border">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-2 py-1 text-left">파일</th>
                <th className="px-2 py-1 text-left">expected</th>
                <th className="px-2 py-1 text-right">크기</th>
                <th className="px-2 py-1"></th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {items.map((it, i) => (
                <tr key={i}>
                  <td className="truncate px-2 py-1" title={it.file.name}>
                    {it.file.name}
                  </td>
                  <td className="px-2 py-1">
                    <input
                      value={it.expected}
                      onChange={(e) => updateLabel(i, e.target.value)}
                      className="w-32 rounded border border-slate-200 px-2 py-0.5 text-xs"
                      placeholder="(미지정)"
                    />
                  </td>
                  <td className="px-2 py-1 text-right text-xs text-slate-500">
                    {(it.file.size / 1024).toFixed(1)} KB
                  </td>
                  <td className="px-2 py-1 text-right">
                    <button
                      onClick={() => removeAt(i)}
                      className="text-xs text-red-600 hover:underline"
                    >
                      ×
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {result && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Metric label="정확도" value={pct(result.accuracy)} hi />
            <Metric
              label="정답"
              value={`${result.n_correct}/${result.n_scored}`}
            />
            <Metric label="총 파일" value={String(result.n_total)} />
            <Metric label="평균 지연" value={`${result.avg_latency_ms.toFixed(1)} ms`} />
          </div>

          {result.per_label.length > 0 && (
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-xs uppercase text-slate-500">
                  <tr>
                    <th className="px-2 py-1 text-left">라벨</th>
                    <th className="px-2 py-1 text-right">support</th>
                    <th className="px-2 py-1 text-right">TP</th>
                    <th className="px-2 py-1 text-right">FP</th>
                    <th className="px-2 py-1 text-right">FN</th>
                    <th className="px-2 py-1 text-right">precision</th>
                    <th className="px-2 py-1 text-right">recall</th>
                    <th className="px-2 py-1 text-right">F1</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {result.per_label.map((s) => (
                    <tr key={s.label}>
                      <td className="px-2 py-1 font-medium">{s.label}</td>
                      <td className="px-2 py-1 text-right">{s.support}</td>
                      <td className="px-2 py-1 text-right text-emerald-700">{s.tp}</td>
                      <td className="px-2 py-1 text-right text-red-600">{s.fp}</td>
                      <td className="px-2 py-1 text-right text-red-600">{s.fn}</td>
                      <td className="px-2 py-1 text-right">{pct(s.precision)}</td>
                      <td className="px-2 py-1 text-right">{pct(s.recall)}</td>
                      <td className="px-2 py-1 text-right">{pct(s.f1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="max-h-72 overflow-auto rounded-md border">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-xs uppercase text-slate-500">
                <tr>
                  <th className="px-2 py-1 text-left">파일</th>
                  <th className="px-2 py-1 text-left">expected</th>
                  <th className="px-2 py-1 text-left">predicted</th>
                  <th className="px-2 py-1 text-right">distance</th>
                  <th className="px-2 py-1 text-right">latency</th>
                  <th className="px-2 py-1 text-center">결과</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {result.cases.map((c, i) => (
                  <tr key={i} className={c.correct === false ? "bg-red-50" : ""}>
                    <td className="truncate px-2 py-1" title={c.filename}>{c.filename}</td>
                    <td className="px-2 py-1">{c.expected ?? <span className="text-slate-400">—</span>}</td>
                    <td className="px-2 py-1">
                      {c.error ? (
                        <span className="text-red-600">{c.error}</span>
                      ) : (
                        c.predicted ?? <span className="text-slate-400">—</span>
                      )}
                    </td>
                    <td className="px-2 py-1 text-right">
                      {c.distance != null ? c.distance.toFixed(3) : "—"}
                    </td>
                    <td className="px-2 py-1 text-right text-xs text-slate-500">
                      {c.latency_ms.toFixed(1)} ms
                    </td>
                    <td className="px-2 py-1 text-center">
                      {c.correct === true && <span className="text-emerald-600">✓</span>}
                      {c.correct === false && <span className="text-red-600">✗</span>}
                      {c.correct === null && <span className="text-slate-400">—</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}

function Metric({ label, value, hi }: { label: string; value: string; hi?: boolean }) {
  return (
    <div className={`rounded-md border p-3 ${hi ? "bg-slate-900 text-white" : "bg-white"}`}>
      <div className={`text-xs ${hi ? "text-slate-400" : "text-slate-500"}`}>{label}</div>
      <div className="mt-1 text-xl font-bold">{value}</div>
    </div>
  );
}
