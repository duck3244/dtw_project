import { useCallback, useEffect, useState } from "react";
import { api, type TemplatesResponse } from "./api";
import EvaluatePanel from "./components/EvaluatePanel";
import { NotificationProvider, useNotify } from "./components/Notification";
import ResultPanel from "./components/ResultPanel";
import TemplateManager from "./components/TemplateManager";

function Shell() {
  const [templates, setTemplates] = useState<TemplatesResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [backend, setBackend] = useState<string | null>(null);
  const notify = useNotify();

  const refresh = useCallback(async () => {
    try {
      setTemplates(await api.listTemplates());
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e));
    }
  }, [notify]);

  useEffect(() => {
    api
      .health()
      .then((h) => setBackend(`${h.backend}${h.accel_available ? " · numba" : ""}`))
      .catch(() => setBackend("offline"));
    refresh();
  }, [refresh]);

  return (
    <main className="mx-auto max-w-3xl space-y-6 p-6">
      <header className="flex items-end justify-between border-b pb-3">
        <div>
          <h1 className="text-2xl font-bold">DTW Speech Recognition</h1>
          <p className="text-sm text-slate-500">템플릿 등록 → 새로운 음성 인식 (MVP)</p>
        </div>
        <span className="text-xs text-slate-400">backend: {backend ?? "…"}</span>
      </header>

      <TemplateManager
        templates={templates}
        busy={busy}
        setBusy={setBusy}
        refresh={refresh}
      />
      <ResultPanel templates={templates} busy={busy} setBusy={setBusy} />
      <EvaluatePanel templates={templates} busy={busy} setBusy={setBusy} />
    </main>
  );
}

export default function App() {
  return (
    <NotificationProvider>
      <Shell />
    </NotificationProvider>
  );
}
