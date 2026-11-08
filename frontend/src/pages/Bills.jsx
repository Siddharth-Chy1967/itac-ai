import { useCallback, useEffect, useRef, useState } from "react";
import UploadDropzone from "../components/UploadDropzone.jsx";
import {
  billWorkbookUrl, cancelBillJob, confirmBillColumns, getBillColumns,
  getBillJob, uploadBills,
} from "../api/client.js";

// Utility bill analysis: drop a year of bills, get the analysis workbook.
//
// The flow has a deliberate stop in the middle. Column names are the one part a
// model influences, so the proposed layout is shown for confirmation before any
// workbook is written -- a wrong column is a two-second fix here, and a wrong
// spreadsheet somebody forwards to a client is not.

const POLL_MS = 1500;
const SETTLED = ["awaiting_review", "done", "failed", "cancelled"];

const GROUPS = [
  ["usage", "Usage and Cost"],
  ["demand", "Demand and Cost"],
  ["taxes_and_fees", "Taxes and Fees"],
  ["credits", "Credits and Discounts"],
  ["other", "Other"],
];

const STATE_TONE = {
  queued: "bg-slate-100 text-slate-600",
  extracting: "bg-itac-blue-light text-itac-blue",
  awaiting_review: "bg-amber-100 text-amber-800",
  building: "bg-itac-blue-light text-itac-blue",
  done: "bg-itac-green-light text-itac-green-dark",
  failed: "bg-red-100 text-red-700",
  cancelled: "bg-slate-100 text-slate-500",
};

const JOB_KEY = "itac.bills.job";

function remember(id) {
  try {
    if (id) sessionStorage.setItem(JOB_KEY, id);
    else sessionStorage.removeItem(JOB_KEY);
  } catch { /* storage blocked: tab switching still works, refresh just won't resume */ }
}

function recall() {
  try { return sessionStorage.getItem(JOB_KEY); } catch { return null; }
}

export default function Bills() {
  const [job, setJob] = useState(null);
  const [columns, setColumns] = useState([]);
  const [overrides, setOverrides] = useState({});
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const timer = useRef(null);

  const stopPolling = useCallback(() => {
    if (timer.current) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  const poll = useCallback(async (id) => {
    stopPolling();
    try {
      const next = await getBillJob(id);
      setJob(next);
      remember(id);
      if (SETTLED.includes(next.state)) {
        if (next.state === "awaiting_review" || next.state === "done") {
          const layout = await getBillColumns(id);
          setColumns(layout.columns || []);
        }
        return;
      }
      timer.current = setTimeout(() => poll(id), POLL_MS);
    } catch (err) {
      setError(err.message);
    }
  }, [stopPolling]);

  // A browser refresh picks the last job back up. If the backend restarted and no
  // longer knows it, drop it quietly and show the dropzone.
  useEffect(() => {
    const saved = recall();
    if (!saved) return;
    getBillJob(saved)
      .then(() => poll(saved))
      .catch(() => remember(null));
  }, [poll]);

  async function handleFiles(files) {
    setError("");
    setColumns([]);
    setOverrides({});
    setBusy(true);
    try {
      const created = await uploadBills(files);
      setJob(created);
      poll(created.id);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleBuild() {
    setBusy(true);
    setError("");
    try {
      const payload = Object.entries(overrides).map(([label, group]) => ({ label, group }));
      const built = await confirmBillColumns(job.id, payload);
      setJob(built);
      const layout = await getBillColumns(job.id);
      setColumns(layout.columns || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleCancel() {
    try {
      setJob(await cancelBillJob(job.id));
      stopPolling();
    } catch (err) {
      setError(err.message);
    }
  }

  function reset() {
    stopPolling();
    remember(null);
    setJob(null);
    setColumns([]);
    setOverrides({});
    setError("");
  }

  const running = job && !SETTLED.includes(job.state);
  const reviewing = job?.state === "awaiting_review";
  const done = job?.state === "done";

  return (
    <div className="mx-auto h-full max-w-5xl overflow-y-auto px-4 py-6">
      <header className="mb-5">
        <h1 className="text-xl font-semibold text-itac-blue">Utility Bill Analysis</h1>
        <p className="mt-1 text-sm text-slate-600">
          Drop a year of utility bills. They are grouped, read and pivoted into an
          analysis workbook — one sheet per meter, one row per month.
        </p>
      </header>

      {!job ? (
        <UploadDropzone onFiles={handleFiles} disabled={busy} />
      ) : (
        <JobPanel
          job={job}
          running={running}
          onCancel={handleCancel}
          onReset={reset}
        />
      )}

      {error ? (
        <p className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </p>
      ) : null}

      {job?.warnings?.length ? (
        <ul className="mt-4 space-y-1 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {job.warnings.map((w, i) => <li key={i}>{w}</li>)}
        </ul>
      ) : null}

      {(reviewing || done) && columns.length ? (
        <ColumnReview
          columns={columns}
          overrides={overrides}
          onChange={(label, group) =>
            setOverrides((prev) => ({ ...prev, [label]: group }))}
          onBuild={handleBuild}
          busy={busy}
          done={done}
        />
      ) : null}

      {done ? <Result job={job} /> : null}
    </div>
  );
}

function JobPanel({ job, running, onCancel, onReset }) {
  // `unit` is "bills" once value extraction starts. Counting uploaded files there
  // would show a full bar for the whole of the longest phase.
  const { done, total, unit } = job.progress;
  const percent = total ? Math.round((done / total) * 100) : 0;
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-center gap-3">
        <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${STATE_TONE[job.state] || ""}`}>
          {job.state.replace(/_/g, " ")}
        </span>
        <span className="text-sm text-slate-600">{job.message}</span>
        <div className="ml-auto flex gap-2">
          {running ? (
            <button onClick={onCancel}
                    className="rounded-md border border-slate-300 px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-50">
              Cancel
            </button>
          ) : (
            <button onClick={onReset}
                    className="rounded-md border border-slate-300 px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-50">
              Start over
            </button>
          )}
        </div>
      </div>

      <div className="mt-3 flex items-center gap-2">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100">
          <div className="h-full bg-itac-blue transition-all" style={{ width: `${percent}%` }} />
        </div>
        {total ? (
          <span className="shrink-0 text-xs tabular-nums text-slate-500">
            {done} / {total} {unit}
          </span>
        ) : null}
      </div>

      <ul className="mt-3 divide-y divide-slate-100 text-sm">
        {job.files.map((file) => (
          <li key={file.name} className="py-1.5">
            <div className="flex items-center gap-3">
              {/* min-w-0 is what lets the name truncate instead of being squeezed
                  to nothing by a long status message beside it. */}
              <span className="min-w-0 flex-1 truncate text-slate-700">{file.name}</span>
              <span className="shrink-0 text-xs text-slate-500">
                {file.state === "done"
                  ? `${file.bills} bill${file.bills === 1 ? "" : "s"}`
                  : file.state}
              </span>
            </div>
            {file.message ? (
              // Extractor errors carry a full exception string; wrapping it on its
              // own line keeps it readable without scrolling the panel sideways.
              <p className="mt-1 break-words text-xs text-amber-700">{file.message}</p>
            ) : null}
          </li>
        ))}
      </ul>
    </section>
  );
}

function ColumnReview({ columns, overrides, onChange, onBuild, busy, done }) {
  return (
    <section className="mt-5 rounded-xl border border-slate-200 bg-white p-4">
      <h2 className="text-sm font-semibold text-itac-blue">
        {done ? "Columns used" : "Confirm the columns"}
      </h2>
      <p className="mt-1 text-xs text-slate-500">
        Built from the labels these bills actually print. Move anything that landed
        in the wrong band before building — nothing is written until you do.
      </p>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
              <th className="py-2 pr-3 font-medium">Column</th>
              <th className="py-2 pr-3 font-medium">Band</th>
              <th className="py-2 pr-3 font-medium">Unit</th>
              <th className="py-2 pr-3 font-medium">From</th>
              <th className="py-2 font-medium">Printed as</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {columns.map((column) => {
              const key = column.raw_labels[0] || column.canonical;
              return (
                <tr key={column.canonical}>
                  <td className="py-2 pr-3 font-medium text-slate-800">
                    {column.canonical}
                    {column.sign === "credit" ? (
                      <span className="ml-2 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-600">
                        credit
                      </span>
                    ) : null}
                  </td>
                  <td className="py-2 pr-3">
                    <select
                      className="rounded border border-slate-300 px-2 py-1 text-xs disabled:bg-slate-50"
                      value={overrides[key] ?? column.group}
                      disabled={done}
                      onChange={(e) => onChange(key, e.target.value)}
                    >
                      {GROUPS.map(([value, caption]) => (
                        <option key={value} value={value}>{caption}</option>
                      ))}
                    </select>
                  </td>
                  <td className="py-2 pr-3 text-slate-600">{column.unit || "—"}</td>
                  <td className="py-2 pr-3">
                    <span className={`rounded px-1.5 py-0.5 text-[10px] ${
                      column.source === "fallback"
                        ? "bg-amber-100 text-amber-800"
                        : "bg-slate-100 text-slate-600"}`}>
                      {column.source}
                    </span>
                  </td>
                  <td className="py-2 text-xs text-slate-500">
                    {column.raw_labels.join(", ")}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {!done ? (
        <button
          onClick={onBuild}
          disabled={busy}
          className="mt-4 rounded-md bg-itac-blue px-4 py-2 text-sm font-medium text-white hover:bg-itac-blue-dark disabled:opacity-50"
        >
          {busy ? "Building…" : "Build the workbook"}
        </button>
      ) : null}
    </section>
  );
}

function Result({ job }) {
  return (
    <section className="mt-5 rounded-xl border border-itac-green bg-itac-green-light p-4">
      <div className="flex flex-wrap items-center gap-3">
        <div>
          <h2 className="text-sm font-semibold text-itac-blue">Workbook ready</h2>
          <p className="mt-0.5 text-xs text-slate-600">{job.message}</p>
        </div>
        <a
          href={billWorkbookUrl(job.id)}
          className="ml-auto rounded-md bg-itac-blue px-4 py-2 text-sm font-medium text-white hover:bg-itac-blue-dark"
        >
          Download .xlsx
        </a>
      </div>

      {job.workbook_path ? (
        <p className="mt-3 text-xs text-slate-600">
          Also saved on disk at{" "}
          <code className="rounded bg-white/70 px-1 py-0.5 font-mono text-[11px] text-slate-800">
            {job.workbook_path}
          </code>
        </p>
      ) : null}

      <ul className="mt-3 space-y-1 text-xs text-slate-700">
        {job.sheets.map((sheet) => (
          <li key={sheet.title}>
            <span className="font-medium">{sheet.title}</span>
            {" — "}{sheet.rows} month{sheet.rows === 1 ? "" : "s"},
            {" "}{sheet.columns.length} columns
            {sheet.warnings.length ? (
              <span className="ml-2 text-amber-700">{sheet.warnings.join("; ")}</span>
            ) : null}
          </li>
        ))}
      </ul>

      {job.flagged?.length ? (
        <div className="mt-3 rounded-lg border border-amber-200 bg-white px-3 py-2">
          <p className="text-xs font-medium text-amber-800">
            {job.flagged.length} month{job.flagged.length === 1 ? "" : "s"} did not
            reconcile against the printed total — highlighted in the workbook.
          </p>
          <ul className="mt-1 space-y-0.5 text-xs text-slate-600">
            {job.flagged.map((row, i) => (
              <li key={i}>
                {row.sheet} · {row.month}: off by {row.difference}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}
