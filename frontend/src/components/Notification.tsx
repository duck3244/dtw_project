import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

type Level = "info" | "success" | "error";

interface Toast {
  id: number;
  level: Level;
  message: string;
}

interface Ctx {
  notify: (level: Level, message: string) => void;
  success: (message: string) => void;
  error: (message: string) => void;
  info: (message: string) => void;
}

const NotificationContext = createContext<Ctx | null>(null);

export function useNotify(): Ctx {
  const ctx = useContext(NotificationContext);
  if (!ctx) throw new Error("useNotify must be used within NotificationProvider");
  return ctx;
}

const styles: Record<Level, string> = {
  info: "bg-slate-900 text-white",
  success: "bg-emerald-600 text-white",
  error: "bg-red-600 text-white",
};

let nextId = 1;

export function NotificationProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const remove = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const notify = useCallback(
    (level: Level, message: string) => {
      const id = nextId++;
      setToasts((prev) => [...prev, { id, level, message }]);
      window.setTimeout(() => remove(id), level === "error" ? 6000 : 3500);
    },
    [remove],
  );

  const value = useMemo<Ctx>(
    () => ({
      notify,
      info: (m) => notify("info", m),
      success: (m) => notify("success", m),
      error: (m) => notify("error", m),
    }),
    [notify],
  );

  return (
    <NotificationContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed right-4 top-4 z-50 flex w-80 flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`pointer-events-auto rounded-md px-3 py-2 text-sm shadow-lg ${styles[t.level]}`}
            onClick={() => remove(t.id)}
          >
            {t.message}
          </div>
        ))}
      </div>
    </NotificationContext.Provider>
  );
}
