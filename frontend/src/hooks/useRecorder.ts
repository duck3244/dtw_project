import { useCallback, useEffect, useRef, useState } from "react";

type State = "idle" | "recording" | "stopped" | "error";

export interface RecorderResult {
  blob: Blob;
  url: string;
  mimeType: string;
  durationMs: number;
}

function pickMimeType(): string {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];
  if (typeof MediaRecorder === "undefined") return "";
  for (const m of candidates) {
    if (MediaRecorder.isTypeSupported(m)) return m;
  }
  return "";
}

export function useRecorder() {
  const [state, setState] = useState<State>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RecorderResult | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const startedAtRef = useRef<number>(0);
  const lastUrlRef = useRef<string | null>(null);

  const cleanupStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  const reset = useCallback(() => {
    if (lastUrlRef.current) URL.revokeObjectURL(lastUrlRef.current);
    lastUrlRef.current = null;
    setResult(null);
    setError(null);
    setState("idle");
  }, []);

  const start = useCallback(async () => {
    try {
      reset();
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("이 브라우저는 마이크 녹음을 지원하지 않습니다.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mime = pickMimeType();
      const rec = mime
        ? new MediaRecorder(stream, { mimeType: mime })
        : new MediaRecorder(stream);
      chunksRef.current = [];
      rec.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };
      rec.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" });
        const url = URL.createObjectURL(blob);
        lastUrlRef.current = url;
        setResult({
          blob,
          url,
          mimeType: rec.mimeType || "audio/webm",
          durationMs: Date.now() - startedAtRef.current,
        });
        setState("stopped");
        cleanupStream();
      };
      rec.onerror = (e) => {
        setError(String((e as unknown as { error?: Error }).error ?? "record error"));
        setState("error");
        cleanupStream();
      };
      recorderRef.current = rec;
      startedAtRef.current = Date.now();
      rec.start();
      setState("recording");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setState("error");
      cleanupStream();
    }
  }, [cleanupStream, reset]);

  const stop = useCallback(() => {
    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") rec.stop();
  }, []);

  useEffect(() => {
    return () => {
      cleanupStream();
      if (lastUrlRef.current) URL.revokeObjectURL(lastUrlRef.current);
    };
  }, [cleanupStream]);

  return { state, error, result, start, stop, reset };
}

export function recordingFilename(mimeType: string, base = "rec"): string {
  if (mimeType.includes("webm")) return `${base}.webm`;
  if (mimeType.includes("ogg")) return `${base}.ogg`;
  if (mimeType.includes("mp4") || mimeType.includes("aac")) return `${base}.m4a`;
  if (mimeType.includes("wav")) return `${base}.wav`;
  return `${base}.webm`;
}
