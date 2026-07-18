import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import SkillBadge from "./SkillBadge.jsx";
import SourceCard from "./SourceCard.jsx";

// A download chip for a generated .xlsx. `file.url` comes straight from the SSE
// `file` event (served by /api/export/download/<token>); we just link to it.
function DownloadChip({ file }) {
  return (
    <a
      href={file.url}
      download
      className="inline-flex items-center gap-1.5 rounded-md border border-itac-green/40 bg-itac-green-light px-2.5 py-1 text-sm font-medium text-itac-green-dark hover:bg-itac-green/20"
    >
      <span aria-hidden>⬇</span> {file.filename}
    </a>
  );
}

export default function ChatMessage({ message }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-itac-blue px-4 py-2 text-white">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-2xl rounded-bl-sm bg-white px-4 py-3 shadow-sm ring-1 ring-slate-200">
        {message.skills?.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-1.5">
            {message.skills.map((s) => (
              <SkillBadge key={s} name={s} />
            ))}
          </div>
        )}

        {message.content ? (
          <div className="markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          </div>
        ) : message.streaming ? (
          <span className="inline-block h-4 w-1.5 animate-pulse rounded bg-slate-400 align-middle" />
        ) : null}

        {message.error && (
          <div className="mt-2 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
            {message.error}
          </div>
        )}

        {message.files?.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {message.files.map((f, i) => (
              <DownloadChip key={i} file={f} />
            ))}
          </div>
        )}

        <SourceCard sources={message.sources} />
      </div>
    </div>
  );
}
