"use client";

import { useState, useRef, useEffect, KeyboardEvent, FormEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_URL = process.env.NEXT_PUBLIC_RAG_API_URL ?? "http://localhost:8000";

type Message = {
  role: "user" | "assistant";
  content: string;
  error?: boolean;
};

const SUGGESTIONS = [
  "Summarize what these documents are about",
  "What is the notice period in the employment letter?",
  "What does the NDA say about confidentiality?",
  "List my education history from the resume",
];

function SparkleIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className}>
      <path
        d="M12 2c.6 3.6 2.4 5.4 6 6-3.6.6-5.4 2.4-6 6-.6-3.6-2.4-5.4-6-6 3.6-.6 5.4-2.4 6-6Z"
        fill="url(#sparkle-gradient)"
      />
      <defs>
        <linearGradient id="sparkle-gradient" x1="0" y1="0" x2="24" y2="24">
          <stop offset="0%" stopColor="#4285f4" />
          <stop offset="50%" stopColor="#9b72cb" />
          <stop offset="100%" stopColor="#d96570" />
        </linearGradient>
      </defs>
    </svg>
  );
}

function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1 py-1">
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-neutral-400 [animation-delay:-0.3s]" />
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-neutral-400 [animation-delay:-0.15s]" />
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-neutral-400" />
    </span>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <button
      onClick={handleCopy}
      title="Copy"
      className="rounded-full p-1.5 text-neutral-500 transition hover:bg-neutral-200 hover:text-neutral-800 dark:text-neutral-400 dark:hover:bg-neutral-800 dark:hover:text-neutral-100"
    >
      {copied ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <path d="M20 6 9 17l-5-5" />
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="9" y="9" width="13" height="13" rx="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
      )}
    </button>
  );
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streaming]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [input]);

  function updateLastMessage(updater: (m: Message) => Message) {
    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const next = [...prev];
      next[next.length - 1] = updater(next[next.length - 1]);
      return next;
    });
  }

  async function sendQuestion(question: string) {
    if (!question || streaming) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);
    setInput("");
    setStreaming(true);

    try {
      const res = await fetch(`${API_URL}/api/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail ?? "Something went wrong.");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? ""; // keep any incomplete trailing line for next chunk

        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);

          if (event.type === "token") {
            updateLastMessage((m) => ({ ...m, content: m.content + event.text }));
          } else if (event.type === "error") {
            updateLastMessage((m) => ({ ...m, content: event.message, error: true }));
          }
        }
      }
    } catch (err) {
      updateLastMessage((m) => ({
        ...m,
        content: err instanceof Error ? err.message : "Something went wrong.",
        error: true,
      }));
    } finally {
      setStreaming(false);
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    sendQuestion(input.trim());
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuestion(input.trim());
    }
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="flex h-screen flex-col bg-white dark:bg-[#131314]">
      <header className="flex items-center gap-2 px-6 py-4">
        <SparkleIcon className="h-6 w-6" />
        <span className="text-xl text-neutral-700 dark:text-neutral-300">Mini RAG</span>
      </header>

      <main className="flex flex-1 flex-col overflow-y-auto px-4">
        {isEmpty ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-8 pb-24">
            <h1 className="bg-gradient-to-r from-[#4285f4] via-[#9b72cb] to-[#d96570] bg-clip-text text-center text-4xl font-medium text-transparent sm:text-5xl">
              Ask about your documents
            </h1>
            <div className="grid w-full max-w-2xl grid-cols-1 gap-3 sm:grid-cols-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => sendQuestion(s)}
                  className="rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-3 text-left text-sm text-neutral-700 transition hover:bg-neutral-100 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-300 dark:hover:bg-neutral-800"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 py-6">
            {messages.map((m, i) =>
              m.role === "user" ? (
                <div key={i} className="flex justify-end">
                  <div className="max-w-[80%] whitespace-pre-wrap rounded-3xl bg-neutral-100 px-5 py-3 text-[15px] text-neutral-900 dark:bg-neutral-800 dark:text-neutral-100">
                    {m.content}
                  </div>
                </div>
              ) : (
                <div key={i} className="flex gap-4">
                  <SparkleIcon className="mt-1 h-6 w-6 shrink-0" />
                  <div className="min-w-0 flex-1">
                    {m.content ? (
                      <div
                        className={
                          m.error
                            ? "text-[15px] text-red-600 dark:text-red-400"
                            : "prose-chat text-[15px] text-neutral-800 dark:text-neutral-100"
                        }
                      >
                        {m.error ? m.content : <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>}
                      </div>
                    ) : (
                      <TypingDots />
                    )}
                    {m.content && !m.error && (
                      <div className="-ml-1.5 mt-1">
                        <CopyButton text={m.content} />
                      </div>
                    )}
                  </div>
                </div>
              )
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </main>

      <form onSubmit={handleSubmit} className="px-4 pb-6">
        <div className="mx-auto flex max-w-3xl items-center gap-2 rounded-[28px] border border-neutral-200 bg-neutral-50 px-5 py-3 shadow-sm dark:border-neutral-700 dark:bg-neutral-900">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your documents..."
            rows={1}
            className="max-h-40 flex-1 resize-none bg-transparent text-[15px] text-neutral-900 outline-none placeholder:text-neutral-400 dark:text-neutral-100"
            disabled={streaming}
          />
          <button
            type="submit"
            disabled={streaming || !input.trim()}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-neutral-900 text-white transition disabled:opacity-30 dark:bg-neutral-100 dark:text-neutral-900"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 19V5M5 12l7-7 7 7" />
            </svg>
          </button>
        </div>
        <p className="mt-2 text-center text-xs text-neutral-400">
          Mini RAG can make mistakes. Answers are based only on the PDFs in ./data.
        </p>
      </form>
    </div>
  );
}
