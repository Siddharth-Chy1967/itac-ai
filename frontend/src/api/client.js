// API client for the ITAC backend. This build uses only /api/health and
// /api/chat; the other endpoints arrive with later frontend pages.

/** GET /api/health -> { status, provider, model } */
export async function getHealth() {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

// Parse one SSE frame ("event: <name>\ndata: <json>") into { event, data }.
function parseFrame(frame) {
  let event = "message";
  const dataLines = [];
  for (const raw of frame.split("\n")) {
    const line = raw.replace(/\r$/, "");
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return null;
  const payload = dataLines.join("\n");
  try {
    return { event, data: JSON.parse(payload) };
  } catch {
    return { event, data: payload };
  }
}

/**
 * POST /api/chat and stream its SSE response, invoking onEvent({event, data})
 * for each of: skills | sources | token | file | done | error (spec §5.4).
 * Resolves when the stream ends. Pass an AbortSignal to cancel mid-stream.
 */
export async function streamChat({ message, history = [], mode = "chat", signal, onEvent }) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, mode }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`Chat request failed: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Frames are separated by a blank line.
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const parsed = parseFrame(frame);
      if (parsed) onEvent(parsed);
    }
  }
}

// --- Utility-bill uploads (/api/bills) ---------------------------------------
//
// Processing a batch takes minutes, so upload returns a job id immediately and
// the page polls. Nothing is written to a workbook until the proposed column
// layout has been confirmed.

async function json(res, what) {
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // A 413 from nginx is HTML, not JSON: keep the status code.
    }
    throw new Error(`${what}: ${detail}`);
  }
  return res.json();
}

/** POST /api/bills/upload -> the created job. `files` is a FileList or array. */
export async function uploadBills(files) {
  const form = new FormData();
  for (const file of files) form.append("files", file, file.name);
  const res = await fetch("/api/bills/upload", { method: "POST", body: form });
  return json(res, "Upload failed");
}

/** GET /api/bills/{id} -> job status, progress, sheets and flagged months. */
export async function getBillJob(id) {
  return json(await fetch(`/api/bills/${id}`), "Could not read job status");
}

/** GET /api/bills/{id}/columns -> the proposed layout for the review gate. */
export async function getBillColumns(id) {
  return json(await fetch(`/api/bills/${id}/columns`), "Could not read columns");
}

/** POST /api/bills/{id}/columns -> accept the layout and build the workbook. */
export async function confirmBillColumns(id, overrides = []) {
  const res = await fetch(`/api/bills/${id}/columns`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ overrides }),
  });
  return json(res, "Could not build the workbook");
}

/** POST /api/bills/{id}/cancel -> stop the batch before its next file. */
export async function cancelBillJob(id) {
  return json(await fetch(`/api/bills/${id}/cancel`, { method: "POST" }),
              "Could not cancel the job");
}

/** The workbook download URL for a finished job. */
export function billWorkbookUrl(id) {
  return `/api/bills/${id}/download`;
}
