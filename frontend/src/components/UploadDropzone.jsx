import { useRef, useState } from "react";

// Multi-file drag-and-drop for PDFs. Deliberately generic: the Bills page owns
// what happens to the files, this only collects them.
//
// A drop that contains no PDF reports that rather than silently doing nothing --
// dragging a folder is the usual cause, and a dead-looking dropzone reads as a
// broken page.
export default function UploadDropzone({ onFiles, disabled = false, accept = ".pdf" }) {
  const [dragging, setDragging] = useState(false);
  const [problem, setProblem] = useState("");
  const inputRef = useRef(null);

  function accepted(fileList) {
    const files = Array.from(fileList || []);
    const pdfs = files.filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (files.length && !pdfs.length) {
      setProblem("Those were not PDFs. Drop the bill PDFs themselves, not a folder.");
      return [];
    }
    if (pdfs.length < files.length) {
      setProblem(`Ignored ${files.length - pdfs.length} non-PDF file(s).`);
    } else {
      setProblem("");
    }
    return pdfs;
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    if (disabled) return;
    const pdfs = accepted(event.dataTransfer.files);
    if (pdfs.length) onFiles(pdfs);
  }

  function handlePick(event) {
    const pdfs = accepted(event.target.files);
    if (pdfs.length) onFiles(pdfs);
    // Reset so picking the same file twice still fires a change event.
    event.target.value = "";
  }

  const base =
    "flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors";
  const tone = disabled
    ? "border-slate-200 bg-slate-50 text-slate-400"
    : dragging
      ? "border-itac-green bg-itac-green-light text-itac-blue"
      : "border-slate-300 bg-white text-slate-600 hover:border-itac-blue hover:bg-itac-blue-light";

  return (
    <div>
      <div
        className={`${base} ${tone}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (!disabled && (e.key === "Enter" || e.key === " ")) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
      >
        <svg className="h-8 w-8 opacity-60" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round"
                d="M12 16.5V4.5m0 0L7.5 9M12 4.5 16.5 9M4.5 16.5v1.75A2.25 2.25 0 0 0 6.75 20.5h10.5a2.25 2.25 0 0 0 2.25-2.25V16.5" />
        </svg>
        <p className="mt-3 text-sm font-medium">
          {disabled ? "Processing…" : "Drop utility bill PDFs here"}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          or click to choose files — scanned statements and multi-bill PDFs are both fine
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={accept}
          className="hidden"
          onChange={handlePick}
          disabled={disabled}
        />
      </div>
      {problem ? <p className="mt-2 text-xs text-amber-700">{problem}</p> : null}
    </div>
  );
}
