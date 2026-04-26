import { useState } from "react";
import { api, type RecognizeResponse, type TemplatesResponse } from "../api";
import { useNotify } from "./Notification";
import RecorderPanel from "./RecorderPanel";

interface Props {
  templates: TemplatesResponse | null;
  busy: boolean;
  setBusy: (b: boolean) => void;
}

export default function ResultPanel({ templates, busy, setBusy }: Props) {
  const [result, setResult] = useState<RecognizeResponse | null>(null);
  const notify = useNotify();
  const noTemplates = !templates || templates.total === 0;

  const submit = async (file: File) => {
    setBusy(true);
    setResult(null);
    try {
      const r = await api.recognize(file, file.name);
      setResult(r);
      notify.success(`예측: ${r.label}`);
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="space-y-4 rounded-lg border bg-white p-4 shadow-sm">
      <header className="flex items-center justify-between">
        <h2 className="font-semibold">2. 인식</h2>
        {noTemplates && (
          <span className="text-xs text-amber-600">
            먼저 템플릿을 등록하세요
          </span>
        )}
      </header>

      <RecorderPanel
        disabled={busy || noTemplates}
        onSubmit={submit}
        submitLabel="인식하기"
      />

      {result && (
        <div className="rounded-md bg-slate-900 p-4 text-white">
          <div className="text-sm text-slate-400">예측 결과</div>
          <div className="mt-1 text-2xl font-bold">{result.label}</div>
          <div className="mt-1 text-xs text-slate-400">
            거리 {result.distance.toFixed(3)}
          </div>
          <ul className="mt-3 space-y-1 text-xs">
            {result.top_k.map((s, i) => (
              <li
                key={s.label}
                className={`flex items-center justify-between rounded px-2 py-1 ${
                  i === 0 ? "bg-slate-700" : ""
                }`}
              >
                <span>
                  {i + 1}. {s.label}
                </span>
                <span className="text-slate-400">{s.distance.toFixed(3)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
