import { useEffect, useRef, useState } from "react";
import { streamChat } from "../api/client.js";
import ChatInput from "../components/ChatInput.jsx";
import ChatMessage from "../components/ChatMessage.jsx";

// ARchie opens every chat with this line. It is sent as ARchie's first turn, so the
// model knows it has already introduced itself and does not do it again.
const GREETING = "Hey, my name is ARchie, how can I help?";

export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function patchLast(patch) {
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      next[next.length - 1] = typeof patch === "function" ? patch(last) : { ...last, ...patch };
      return next;
    });
  }

  async function handleSend(text) {
    const history = [
      { role: "assistant", content: GREETING },
      ...messages
        .filter((m) => !m.error && m.content)
        .map((m) => ({ role: m.role, content: m.content })),
    ];

    const userMsg = { role: "user", content: text };
    const assistantMsg = {
      role: "assistant",
      content: "",
      sources: [],
      files: [],
      streaming: true,
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat({
        message: text,
        history,
        mode: "chat",
        signal: controller.signal,
        onEvent: ({ event, data }) => {
          if (event === "sources") patchLast({ sources: Array.isArray(data) ? data : [] });
          else if (event === "token") patchLast((m) => ({ ...m, content: m.content + (data.t || "") }));
          else if (event === "file") patchLast((m) => ({ ...m, files: [...m.files, data] }));
          else if (event === "error") patchLast({ error: data.message || "Something went wrong." });
        },
      });
    } catch (err) {
      if (err.name !== "AbortError") {
        patchLast({ error: err.message || "Connection failed. Is the backend running?" });
      }
    } finally {
      patchLast({ streaming: false });
      setStreaming(false);
      abortRef.current = null;
    }
  }

  function handleStop() {
    abortRef.current?.abort();
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="flex h-full flex-col">
      {isEmpty ? (
        <div className="flex-1 overflow-y-auto bg-gradient-to-br from-[#112653] via-[#1B3A2E] to-[#364E1A]">
          <EmptyState />
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl space-y-4 px-4 py-6">
            <ChatMessage message={{ role: "assistant", content: GREETING, sources: [], files: [] }} />
            {messages.map((m, i) => (
              <ChatMessage key={i} message={m} />
            ))}
            <div ref={bottomRef} />
          </div>
        </div>
      )}

      <div className="border-t border-slate-200 bg-slate-50 py-3">
        <div className="mx-auto max-w-3xl px-4">
          <ChatInput onSend={handleSend} disabled={streaming} />
          {streaming && (
            <button onClick={handleStop} className="mt-2 text-xs text-slate-500 hover:text-itac-blue">
              Stop generating
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex h-full flex-col items-center justify-center px-4 text-center text-white">
      <img src="/itac-logo.png" alt="" className="mb-8 h-48 w-48 md:h-64 md:w-64" />
      <p className="text-2xl font-semibold">{GREETING}</p>
      <p className="mt-2 max-w-md text-sm text-white/70">
        Ask what to measure for any AR, which ARs are common for your next assessment's
        SIC or NAICS code, or about past assessment reports.
      </p>
    </div>
  );
}