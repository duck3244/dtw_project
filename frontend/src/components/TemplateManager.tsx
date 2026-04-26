import { useState } from "react";
import { api, type TemplatesResponse } from "../api";
import { useNotify } from "./Notification";
import RecorderPanel from "./RecorderPanel";

interface Props {
  templates: TemplatesResponse | null;
  busy: boolean;
  setBusy: (b: boolean) => void;
  refresh: () => Promise<void> | void;
}

export default function TemplateManager({ templates, busy, setBusy, refresh }: Props) {
  const [label, setLabel] = useState("");
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const notify = useNotify();

  const trimmed = label.trim();
  const existingCount = templates?.labels.find((t) => t.label === trimmed)?.count ?? 0;

  const submit = async (file: File) => {
    if (!trimmed) {
      notify.error("라벨을 입력하세요.");
      return;
    }
    setBusy(true);
    try {
      const res = await api.addTemplate(trimmed, file, file.name);
      notify.success(`'${res.label}' 템플릿 등록 (총 ${res.count}개)`);
      await refresh();
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const doDelete = async (lbl: string) => {
    setBusy(true);
    try {
      await api.deleteLabel(lbl);
      notify.success(`'${lbl}' 라벨 삭제`);
      await refresh();
    } catch (e) {
      notify.error(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setConfirmDelete(null);
    }
  };

  return (
    <section className="space-y-4 rounded-lg border bg-white p-4 shadow-sm">
      <header className="flex items-center justify-between">
        <h2 className="font-semibold">1. 템플릿 등록</h2>
        <span className="text-xs text-slate-500">
          총 {templates?.total ?? 0}개 · {templates?.labels.length ?? 0}개 라벨
        </span>
      </header>

      <div>
        <label className="mb-1 block text-xs font-medium text-slate-600">
          라벨 {trimmed && existingCount > 0 && (
            <span className="ml-2 text-amber-600">
              · '{trimmed}'에 이미 {existingCount}개 있음 (추가됨)
            </span>
          )}
        </label>
        <input
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          maxLength={64}
          className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
          placeholder="라벨 입력 후 녹음/업로드"
        />
      </div>

      <RecorderPanel
        disabled={busy || !trimmed}
        onSubmit={submit}
        submitLabel="이 라벨로 등록"
      />

      {templates && templates.labels.length > 0 && (
        <ul className="divide-y rounded-md border">
          {templates.labels.map((t) => (
            <li
              key={t.label}
              className="flex items-center justify-between px-3 py-2 text-sm"
            >
              <span>
                <strong>{t.label}</strong>{" "}
                <span className="text-slate-500">· {t.count}개</span>
              </span>
              {confirmDelete === t.label ? (
                <span className="flex items-center gap-2 text-xs">
                  <span className="text-slate-500">정말 삭제?</span>
                  <button
                    onClick={() => doDelete(t.label)}
                    disabled={busy}
                    className="rounded bg-red-600 px-2 py-0.5 text-white hover:bg-red-700"
                  >
                    예
                  </button>
                  <button
                    onClick={() => setConfirmDelete(null)}
                    className="rounded border px-2 py-0.5 hover:bg-slate-50"
                  >
                    취소
                  </button>
                </span>
              ) : (
                <button
                  onClick={() => setConfirmDelete(t.label)}
                  disabled={busy}
                  className="text-xs text-red-600 hover:underline disabled:opacity-50"
                >
                  삭제
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
