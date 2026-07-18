import { useState } from "react";

// Collapsible RAG citations. Renders the live `sources` payload shape emitted by
// /api/chat: [{ report_id, fiscal_year, section, score }].
export default function SourceCard({ sources }) {
  const [open, setOpen] = useState(false);
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-3 overflow-hidden rounded-lg border border-slate-200 bg-slate-50">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-itac-blue"
      >
        <span>
          {sources.length} source{sources.length > 1 ? "s" : ""} from the report knowledge base
        </span>
        <span aria-hidden>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <ul className="divide-y divide-slate-200 border-t border-slate-200">
          {sources.map((s, i) => (
            <li key={i} className="px-3 py-2 text-xs text-slate-700">
              <div className="flex items-baseline justify-between gap-2">
                <span>
                  <span className="font-semibold text-itac-blue">{s.report_id || "Unknown report"}</span>
                  {s.fiscal_year ? <span className="text-slate-500"> · {s.fiscal_year}</span> : null}
                </span>
                {typeof s.score === "number" ? (
                  <span className="shrink-0 text-slate-400">{s.score.toFixed(2)}</span>
                ) : null}
              </div>
              {s.section ? <div className="mt-0.5 text-slate-600">{s.section}</div> : null}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
