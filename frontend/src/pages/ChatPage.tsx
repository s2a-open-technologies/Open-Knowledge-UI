import { useCallback, useEffect, useRef, useState, type DragEvent, type ReactNode } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { addToStoredDraft, loadDraft, saveDraft } from "@/lib/chatDraft";
import { uid } from "@/lib/utils";
import { Composer, ATTACH_ACCEPT } from "@/components/chat/Composer";
import { Message, type SourceInfo, type ToolCall, type ToolResult } from "@/components/chat/Message";
import { useSessions } from "@/context/SessionsContext";
import { useStt, type SttMode } from "@/hooks/useStt";
import { api, ApiError, csrfHeaders, errorMessage } from "@/lib/api";
import {
  MODELS_CHANGED,
  readChosenModel,
  readSeenDefault,
  reconcileChosenModel,
  writeChosenModel,
  writeSeenDefault,
} from "@/lib/modelChoice";
import { consumeChatStream, type ChatStreamEvent } from "@/lib/sse";
import type {
  ChatAttachment,
  ChatMessage,
  Knowledge,
  ModelEntry,
  ModelsResponse,
  SessionWithMessages,
} from "@/lib/types";

interface LiveMessage {
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  tools?: ToolCall[];
  toolResults?: ToolResult[];
  sources?: SourceInfo[];
  streaming?: boolean;
  // Dateinamen der mit dieser Nachricht angehängten Dateien (nur live bekannt).
  attachments?: string[];
  // Hinweis, solange die Antwort auf einen Anhang wartet
  waiting?: string;
}

const KNOWLEDGE_KEY = "rag.selectedKnowledge";
const POLL_MS = 1500;

// Ohne gespeicherte Auswahl ist kein Knowledge gewählt; "all" steht für „alle“ (null).
function loadSelectedKnowledge(): number[] | null {
  try {
    const raw = localStorage.getItem(KNOWLEDGE_KEY);
    if (!raw) return [];
    if (raw === "all") return null;
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.map(Number) : [];
  } catch {
    return [];
  }
}

// Dateien lassen sich auf das ganze Chatfenster ziehen (nicht nur auf das Eingabefeld);
// sonst oeffnet der Browser eine fallengelassene Datei in einem neuen Tab.
function DropZone({
  onFiles,
  children,
}: {
  onFiles: (files: File[]) => void;
  children: ReactNode;
}) {
  const [over, setOver] = useState(false);
  const depth = useRef(0);
  const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes("Files");

  return (
    <div
      className="relative h-full"
      onDragEnter={(e) => {
        if (!hasFiles(e)) return;
        e.preventDefault();
        depth.current += 1;
        setOver(true);
      }}
      onDragOver={(e) => {
        if (!hasFiles(e)) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
      }}
      onDragLeave={(e) => {
        if (!hasFiles(e)) return;
        depth.current = Math.max(0, depth.current - 1);
        if (depth.current === 0) setOver(false);
      }}
      onDrop={(e) => {
        if (!hasFiles(e)) return;
        e.preventDefault();
        depth.current = 0;
        setOver(false);
        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) onFiles(files);
      }}
    >
      {children}
      {over ? (
        <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center rounded-2xl border-2 border-dashed border-primary bg-background/80">
          <p className="text-lg font-medium">Dateien hier ablegen (PDF, DOCX, DOC, ODT, MD)</p>
        </div>
      ) : null}
    </div>
  );
}

export function ChatPage() {
  const { sessionId } = useParams<{ sessionId?: string }>();
  const navigate = useNavigate();
  const { reload: reloadSessions } = useSessions();

  const [messages, setMessages] = useState<LiveMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  // Aktueller Stand fuer wartende Sendevorgaenge (State in Closures waere veraltet)
  const attachmentsRef = useRef<ChatAttachment[]>([]);
  const inputRef = useRef("");
  useEffect(() => {
    attachmentsRef.current = attachments;
  }, [attachments]);
  useEffect(() => {
    inputRef.current = input;
  }, [input]);
  const [knowledge, setKnowledge] = useState<Knowledge[]>([]);
  const [selectedKnowledge, setSelectedKnowledge] = useState<number[] | null>(
    loadSelectedKnowledge,
  );

  const [models, setModels] = useState<ModelEntry[]>([]);
  const [defaultModel, setDefaultModel] = useState("");
  const [chosenModel, setChosenModel] = useState<string>(readChosenModel);
  // Gewähltes Modell, sonst Standardmodell. Bewusst kein Fallback auf ein
  // beliebiges Modell: Daten sollen nicht unbemerkt an einen anderen Endpunkt gehen.
  const selectedModel =
    [chosenModel, defaultModel].find((r) => models.some((m) => m.ref === r)) ?? "";
  const activeModel = models.find((m) => m.ref === selectedModel);
  // Gesperrt nur, wenn der Admin zusätzliches Knowledge nicht erlaubt hat (sonst: Modell-Wissen + eigene Auswahl)
  const knowledgeLocked = Boolean(
    activeModel?.knowledge_locked ||
      (activeModel?.config?.knowledge_ids?.length && activeModel.config.allow_extra_knowledge === false),
  );
  const baseKnowledge = activeModel?.knowledge_base ?? [];

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  // Session, die wir gerade selbst angelegt haben: beim Wechsel der URL darauf
  // darf der lokale Zustand (Nachrichten, Anhänge) nicht neu geladen werden.
  const skipLoadRef = useRef<string | null>(null);
  // Chat, der gerade selbst angelegt wurde: sein Entwurf bleibt im Speicher stehen
  const adoptedRef = useRef<string | null>(null);
  // Schluessel des Entwurfs, der gerade angezeigt wird ("new" oder die Session-ID)
  const draftKeyRef = useRef<string>(sessionId ?? "new");
  const aliveRef = useRef(true);

  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
    };
  }, []);

  // Beim Streamen nur nachziehen, wenn der Nutzer am Ende steht (sonst liest er gerade weiter oben).
  const scrollToBottom = useCallback((force = false) => {
    const el = scrollRef.current;
    if (!el) return;
    if (force || el.scrollHeight - el.scrollTop - el.clientHeight < 160) el.scrollTop = el.scrollHeight;
  }, []);

  useEffect(() => {
    api
      .get<Knowledge[]>("/api/knowledge")
      .then(setKnowledge)
      .catch((e) => console.error(e));
  }, []);

  // Modellliste und Standardmodell: beim Öffnen, nach Änderungen im Admin-Dialog
  // (die Seite bleibt dahinter gemountet) und wenn das Fenster wieder Fokus bekommt.
  useEffect(() => {
    const load = () => {
      api
        .get<ModelsResponse>("/api/models?visible=1")
        .then((res) => {
          setModels(res.models);
          setDefaultModel(res.default);
          // Neuer Standard des Admins gewinnt gegen eine früher gespeicherte Auswahl.
          const next = reconcileChosenModel(
            readChosenModel(),
            res.default,
            readSeenDefault(),
          );
          writeSeenDefault(next.seenDefault);
          setChosenModel((prev) => {
            if (next.chosen === prev) return prev;
            writeChosenModel(next.chosen);
            return next.chosen;
          });
        })
        .catch((e) => console.error(e));
    };
    const onVisible = () => {
      if (document.visibilityState === "visible") load();
    };
    load();
    window.addEventListener(MODELS_CHANGED, load);
    window.addEventListener("focus", load);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.removeEventListener(MODELS_CHANGED, load);
      window.removeEventListener("focus", load);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  const changeModel = (ref: string) => {
    setChosenModel(ref);
    writeChosenModel(ref);
  };

  const changeKnowledge = (ids: number[] | null) => {
    setSelectedKnowledge(ids);
    try {
      if (ids === null) localStorage.setItem(KNOWLEDGE_KEY, "all");
      else localStorage.setItem(KNOWLEDGE_KEY, JSON.stringify(ids));
    } catch {
      // localStorage nicht verfügbar: Auswahl gilt nur für diese Sitzung.
    }
  };

  // Session aus der URL laden bzw. bei "/" einen leeren Chat zeigen.
  useEffect(() => {
    if (sessionId && skipLoadRef.current === sessionId) {
      skipLoadRef.current = null;
      return;
    }
    if (!sessionId) {
      setMessages([]);
      return;
    }
    let cancelled = false;
    api
      .get<SessionWithMessages>(`/api/sessions/${sessionId}`)
      .then((session) => {
        if (cancelled) return;
        setMessages(
          (session.messages || []).map((m: ChatMessage) => ({
            role: m.role,
            content: m.content,
          })),
        );
        requestAnimationFrame(() => scrollToBottom(true));
      })
      .catch((e) => {
        if (cancelled) return;
        toast.error(errorMessage(e));
        navigate("/", { replace: true });
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId, navigate, scrollToBottom]);

  const adoptSession = useCallback(
    (id: string) => {
      skipLoadRef.current = id;
      adoptedRef.current = id;
      navigate(`/c/${id}`, { replace: true });
      void reloadSessions();
    },
    [navigate, reloadSessions],
  );

  // ---- Anhänge -----------------------------------------------------------

  const patchAttachment = useCallback(
    (id: string, patch: Partial<ChatAttachment>) =>
      setAttachments((prev) =>
        prev.map((a) => (a.id === id ? { ...a, ...patch } : a)),
      ),
    [],
  );

  const pollAttachment = useCallback(
    async (id: string) => {
      while (aliveRef.current) {
        await new Promise((r) => setTimeout(r, POLL_MS));
        try {
          const att = await api.get<{ status: string; error: string | null }>(
            `/api/chat/attachments/${id}`,
          );
          if (att.status === "ready") {
            patchAttachment(id, { status: "ready" });
            return;
          }
          if (att.status === "error") {
            patchAttachment(id, { status: "error", error: att.error });
            toast.error(att.error || "Anhang konnte nicht gelesen werden");
            return;
          }
        } catch (e) {
          patchAttachment(id, { status: "error", error: errorMessage(e) });
          return;
        }
      }
    },
    [patchAttachment],
  );

  // Entwurf (Anhaenge und Text) beim Wechsel des Chats wiederherstellen: wer ein Menue
  // besucht und zurueckkehrt, soll seine hochgeladenen Dateien noch sehen.
  useEffect(() => {
    const key = sessionId ?? "new";
    if (adoptedRef.current && adoptedRef.current === sessionId) {
      adoptedRef.current = null;
      draftKeyRef.current = key; // der aktuelle Zustand gehoert jetzt zu diesem Chat
      saveDraft(key, { attachments: attachmentsRef.current, input: inputRef.current });
      saveDraft("new", { attachments: [], input: "" }); // der Entwurf zog in den neuen Chat um
      return;
    }
    draftKeyRef.current = key;
    const draft = loadDraft(key);
    setAttachments(draft.attachments);
    setInput(draft.input);
    // Dateien, die beim Verlassen noch gelesen wurden, weiter beobachten
    for (const a of draft.attachments) {
      if (a.status === "processing" || a.status === "uploading") void pollAttachment(a.id);
    }
  }, [sessionId, pollAttachment]);

  useEffect(() => {
    saveDraft(draftKeyRef.current, { attachments, input });
  }, [attachments, input]);

  const handleAttach = useCallback(
    async (files: File[]) => {
      const allowed = ATTACH_ACCEPT.split(",");
      let sid = sessionId ?? null;

      // Nacheinander: der erste Upload eines neuen Chats legt die Session an.
      for (const file of files) {
        const ext = `.${file.name.split(".").pop()?.toLowerCase() ?? ""}`;
        if (!allowed.includes(ext)) {
          toast.error(
            `„${file.name}“: Dateityp nicht unterstützt (PDF, DOCX, DOC, ODT, MD, Bilder). CSV/Excel bitte über Knowledge hochladen.`,
          );
          continue;
        }

        const localId = `local-${uid()}`;
        setAttachments((prev) => [
          ...prev,
          { id: localId, filename: file.name, status: "uploading" },
        ]);

        const form = new FormData();
        form.append("file", file);
        if (sid) form.append("session_id", sid);

        try {
          const res = await api.post<{
            id: string;
            session_id: string;
            status: "processing";
          }>("/api/chat/attachments", form);
          if (!sid) {
            sid = res.session_id;
            // Nur wenn der Nutzer noch im Chat ist; sonst wuerde der Upload ihn zurueckholen
            if (aliveRef.current) adoptSession(sid);
            else void reloadSessions();
          }
          patchAttachment(localId, { id: res.id, status: "processing" });
          if (!aliveRef.current) {
            // Die Seite wurde waehrend des Uploads verlassen: Datei im Entwurf vermerken,
            // damit sie beim Zurueckkehren wieder erscheint.
            addToStoredDraft(res.session_id, { id: res.id, filename: file.name, status: "processing" });
          }
          void pollAttachment(res.id);
        } catch (e) {
          patchAttachment(localId, { status: "error", error: errorMessage(e) });
          toast.error(errorMessage(e));
        }
      }
    },
    [sessionId, adoptSession, reloadSessions, patchAttachment, pollAttachment],
  );

  // ---- Chat --------------------------------------------------------------

  const stop = useCallback(() => {
    abortRef.current?.abort();
    if (sessionId) {
      api.post("/chat/stop", { session_id: sessionId }).catch(() => {});
    }
  }, [sessionId]);

  const handleEvent = useCallback(
    (event: ChatStreamEvent) => {
      if (event.warning) toast.warning(event.warning);
      if (event.compaction) {
        toast.info("Der bisherige Verlauf wurde zusammengefasst, damit der Kontext des Modells nicht überläuft.");
      }
      if (event.error) {
        toast.error(`Fehler: ${event.error}`);
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last?.role === "assistant") {
            last.streaming = false;
            last.content = "";
          }
          return next;
        });
        return;
      }

      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (!last || last.role !== "assistant") return next;

        if (event.reasoning) {
          last.reasoning = (last.reasoning || "") + event.reasoning;
          last.tools = last.tools ?? [];
        }
        if (event.token) last.content += event.token;
        if (event.sources) last.content += event.sources;
        if (event.tool_start) {
          last.tools = [
            ...(last.tools ?? []),
            { name: event.tool_start, status: "running" },
          ];
        }
        if (event.tool_end) {
          const tools = [...(last.tools ?? [])];
          const idx = tools.findIndex(
            (t) => t.name === event.tool_end && t.status === "running",
          );
          if (idx >= 0) tools[idx] = { ...tools[idx], status: "done" };
          last.tools = tools;
        }
        if (event.tool_result) {
          last.toolResults = [...(last.toolResults ?? []), event.tool_result];
        }
        if (event.source_list) {
          const seen = new Set((last.sources ?? []).map((x) => `${x.filename}|${x.section}`));
          last.sources = [
            ...(last.sources ?? []),
            ...event.source_list.filter((x) => !seen.has(`${x.filename}|${x.section}`)),
          ];
        }
        if (event.done) last.streaming = false;
        return next;
      });
      scrollToBottom();
    },
    [scrollToBottom],
  );

  const sendMessage = useCallback(
    async (message: string) => {
      const controller = new AbortController();
      abortRef.current = controller;
      setStreaming(true);

      // Die Frage erscheint sofort; liegen noch Anhaenge in der Verarbeitung, wartet die
      // Antwort darauf (sonst wuerde das Modell ohne den Dateiinhalt antworten).
      // Bilder warten nicht auf das Lesen, wenn das Modell sie selbst sieht (nur auf den Upload).
      const sees = Boolean(activeModel?.vision);
      const isImage = (a: ChatAttachment) => /\.(png|jpe?g|webp|gif|bmp)$/i.test(a.filename);
      const isBusy = (a: ChatAttachment) =>
        a.status === "uploading" || (a.status === "processing" && !(sees && isImage(a)));
      const initial = attachments.filter((a) => a.status !== "error");
      setMessages((prev) => [
        ...prev,
        {
          role: "user",
          content: message,
          attachments: initial.map((a) => a.filename),
        },
        {
          role: "assistant",
          content: "",
          streaming: true,
          tools: [],
          toolResults: [],
          waiting: attachments.some(isBusy)
            ? `Warte, bis ${attachments.filter(isBusy).map((a) => `„${a.filename}“`).join(", ")} gelesen ${attachments.filter(isBusy).length > 1 ? "sind" : "ist"} …`
            : undefined,
        },
      ]);
      requestAnimationFrame(() => scrollToBottom(true));

      try {
        const deadline = Date.now() + 10 * 60 * 1000;
        while (attachmentsRef.current.some(isBusy)) {
          if (controller.signal.aborted) throw new DOMException("abgebrochen", "AbortError");
          if (Date.now() > deadline) throw new Error("Der Anhang wird zu lange verarbeitet.");
          await new Promise((r) => setTimeout(r, 300));
        }
        const failed = attachmentsRef.current.filter((a) => a.status === "error");
        if (failed.length > 0) {
          toast.error(
            `„${failed.map((a) => a.filename).join("“, „")}“ konnte nicht gelesen werden – die Frage wird ohne diesen Anhang gestellt.`,
          );
        }
        const imageIds = sees
          ? attachmentsRef.current
              .filter((a) => isImage(a) && a.status !== "error" && a.status !== "uploading")
              .map((a) => a.id)
          : [];
        setAttachments([]);
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last?.role === "assistant") last.waiting = undefined;
          return next;
        });

        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json", ...csrfHeaders() },
          body: JSON.stringify({
            message,
            session_id: sessionId ?? null,
            knowledge_ids: selectedKnowledge,
            attachment_ids: imageIds,
            model: selectedModel || undefined,
          }),
          signal: controller.signal,
          credentials: "same-origin",
        });

        if (response.status === 401) {
          throw new ApiError(401, "Sitzung abgelaufen. Bitte neu anmelden.");
        }

        await consumeChatStream(
          response,
          (event) => {
            if (event.done && event.session_id) {
              if (!sessionId) adoptSession(event.session_id);
              else void reloadSessions();
            }
            handleEvent(event);
          },
          controller.signal,
        );
      } catch (e) {
        if ((e as Error).name === "AbortError") {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant") {
              last.streaming = false;
              if (!last.content) last.content = "*Antwort abgebrochen.*";
            }
            return next;
          });
        } else {
          toast.error(errorMessage(e));
        }
      } finally {
        setStreaming(false);
        abortRef.current = null;
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last?.role === "assistant") last.streaming = false;
          return next;
        });
        textareaRef.current?.focus();
      }
    },
    [
      attachments,
      sessionId,
      selectedKnowledge,
      selectedModel,
      activeModel,
      adoptSession,
      reloadSessions,
      handleEvent,
      scrollToBottom,
    ],
  );

  const submit = () => {
    const message = input.trim();
    if (!message) return;
    setInput("");
    void sendMessage(message);
  };

  const [sttConfig, setSttConfig] = useState<{ available: boolean; mode: SttMode }>({
    available: true,
    mode: "stream",
  });
  useEffect(() => {
    api
      .get<{ available: boolean; mode: SttMode }>("/api/stt/config")
      .then(setSttConfig)
      .catch(() => undefined);
  }, []);

  const stt = useStt({
    mode: sttConfig.mode,
    onText: (text) => setInput(text),
    onFinal: (text) => {
      setInput("");
      if (text.trim()) void sendMessage(text.trim());
    },
    onError: (message) => toast.error(message),
  });

  const toggleMic = async () => {
    try {
      if (stt.recording) stt.stop();
      else if (stt.transcribing) return;
      else if (!sttConfig.available) toast.error("Die Spracheingabe ist nicht eingerichtet (Administration › Tools).");
      else await stt.start();
    } catch (e) {
      toast.error(errorMessage(e));
    }
  };

  const composer = (
    <Composer
      seesImages={Boolean(activeModel?.vision)}
      history={messages
        .filter((m) => m.role === "user" && m.content.trim())
        .map((m) => m.content)
        .filter((c, i, all) => i === 0 || c !== all[i - 1])}
      value={input}
      onChange={setInput}
      onSubmit={submit}
      onStop={stop}
      streaming={streaming}
      // Nach dem Senden zeigt die Antwort den Wartezustand; hier nicht noch einmal
      attachments={streaming ? [] : attachments}
      onAttach={(files) => void handleAttach(files)}
      onRemoveAttachment={(id) =>
        setAttachments((prev) => prev.filter((a) => a.id !== id))
      }
      knowledge={knowledge}
      selectedKnowledge={selectedKnowledge}
      onSelectKnowledge={changeKnowledge}
      knowledgeLocked={knowledgeLocked}
      baseKnowledge={baseKnowledge}
      models={models}
      selectedModel={selectedModel}
      onSelectModel={changeModel}
      recording={stt.recording}
      transcribing={stt.transcribing}
      onToggleMic={toggleMic}
      textareaRef={textareaRef}
    />
  );

  if (messages.length === 0) {
    return (
      <DropZone onFiles={(files) => void handleAttach(files)}>
      <div className="flex h-full flex-col items-center justify-center gap-8 px-4">
        <h1 className="text-center text-3xl font-semibold tracking-tight">
          Womit kann ich helfen?
        </h1>
        <div className="w-full max-w-5xl">{composer}</div>
      </div>
      </DropZone>
    );
  }

  return (
    <DropZone onFiles={(files) => void handleAttach(files)}>
    <div className="flex h-full flex-col">
      <div
        ref={scrollRef}
        className="min-h-0 flex-1 overflow-y-auto [scrollbar-gutter:stable]"
      >
        <div className="mx-auto flex w-full max-w-5xl flex-col gap-6 px-4 py-6">
          {messages.map((m, i) => (
            <Message
              key={i}
              role={m.role}
              content={m.content}
              reasoning={m.reasoning}
              tools={m.tools}
              toolResults={m.toolResults}
              sources={m.sources}
              streaming={m.streaming}
              attachments={m.attachments}
              waiting={m.waiting}
            />
          ))}
        </div>
      </div>
      <div className="mx-auto w-full max-w-5xl shrink-0 px-4 pb-4 pt-2">
        {composer}
      </div>
      {stt.recording || stt.transcribing ? (
        <span className="sr-only" aria-live="polite">
          {stt.recording ? "Aufnahme läuft" : "Wird transkribiert"}
        </span>
      ) : null}
    </div>
    </DropZone>
  );
}
