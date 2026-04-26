import { useEffect, useState } from "react";
import { recordingFilename, useRecorder } from "../hooks/useRecorder";

interface Props {
  disabled?: boolean;
  onSubmit: (file: File) => Promise<void> | void;
  submitLabel: string;
  acceptUploadFallback?: boolean;
}

function formatDuration(ms: number): string {
  return `${(ms / 1000).toFixed(1)}s`;
}

export default function RecorderPanel({
  disabled,
  onSubmit,
  submitLabel,
  acceptUploadFallback = true,
}: Props) {
  const { state, error, result, start, stop, reset } = useRecorder();
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (state !== "recording") {
      setElapsed(0);
      return;
    }
    const startedAt = Date.now();
    const id = window.setInterval(() => setElapsed(Date.now() - startedAt), 100);
    return () => window.clearInterval(id);
  }, [state]);

  const submitRecording = async () => {
    if (!result) return;
    const file = new File([result.blob], recordingFilename(result.mimeType), {
      type: result.mimeType,
    });
    await onSubmit(file);
    reset();
  };

  const submitUpload = async (f: File) => {
    await onSubmit(f);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        {state !== "recording" ? (
          <button
            type="button"
            disabled={disabled}
            onClick={start}
            className="rounded-md bg-rose-600 px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-rose-700 disabled:opacity-50"
          >
            ● 녹음 시작
          </button>
        ) : (
          <button
            type="button"
            onClick={stop}
            className="rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-slate-800"
          >
            ■ 중지 ({formatDuration(elapsed)})
          </button>
        )}
        {state === "recording" && (
          <span className="inline-flex h-2 w-2 animate-pulse rounded-full bg-rose-500" />
        )}
        {acceptUploadFallback && (
          <label className="cursor-pointer rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 hover:bg-slate-50">
            파일 업로드
            <input
              type="file"
              accept="audio/*"
              className="hidden"
              disabled={disabled || state === "recording"}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) submitUpload(f);
                e.target.value = "";
              }}
            />
          </label>
        )}
      </div>

      {result && (
        <div className="rounded-md border bg-slate-50 p-3">
          <audio src={result.url} controls className="w-full" />
          <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
            <span>
              {formatDuration(result.durationMs)} · {result.mimeType}
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={reset}
                className="rounded px-2 py-1 hover:bg-slate-200"
              >
                다시 녹음
              </button>
              <button
                type="button"
                disabled={disabled}
                onClick={submitRecording}
                className="rounded bg-emerald-600 px-2 py-1 text-white hover:bg-emerald-700 disabled:opacity-50"
              >
                {submitLabel}
              </button>
            </div>
          </div>
        </div>
      )}

      {error && (
        <p className="text-xs text-red-600">{error}</p>
      )}
    </div>
  );
}
