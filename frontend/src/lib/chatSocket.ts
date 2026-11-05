// Chat über den WebSocket /ws/chat (siehe app/api/chat.py).
//
// Eine Verbindung pro Seite; jede Frage bekommt eine im Browser erzeugte stream_id, damit auch der
// allererste Chat (noch ohne Session) sofort stoppbar ist. Stopp = Frame {"type":"stop"}; ist der
// Socket nicht offen, geht der Stopp per POST /chat/stop. Bricht die Verbindung während einer
// Antwort ab, endet sie als "abgebrochen". Es gibt bewusst keinen Rückfall auf HTTP-Streaming.

import { api } from "@/lib/api";
import { uid } from "@/lib/utils";

export interface ChatStreamEvent {
  type?: "ready" | "cancelled";
  stream_id?: string;
  token?: string;
  reasoning?: string;
  reasoning_done?: boolean;
  tool_start?: string;
  tool_end?: string;
  tool_result?: { name: string; output: string; input?: string };
  sources?: string;
  source_list?: import("@/components/chat/Message").SourceInfo[];
  warning?: string;
  compaction?: { before: number; after: number; summarized?: number; clipped?: boolean };
  done?: boolean;
  session_id?: string;
  error?: string;
}

export interface ChatRequest {
  message: string;
  session_id: string | null;
  knowledge_ids: number[] | null;
  attachment_ids: string[];
  model?: string;
}

/** Wie eine Antwort endete: fertig, mit Fehler oder abgebrochen (Stopp oder Verbindung weg). */
export type ChatOutcome = "done" | "error" | "cancelled";

export class ChatSocketError extends Error {}

/** Wartezeiten zwischen Verbindungsversuchen (ms). */
export const RECONNECT_DELAYS = [0, 500, 1500];

type SocketFactory = (url: string) => WebSocket;

interface Active {
  streamId: string;
  onEvent: (event: ChatStreamEvent) => void;
  finish: (outcome: ChatOutcome) => void;
}

export function chatSocketUrl(location: Location = window.location): string {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}/ws/chat`;
}

export class ChatSocket {
  private socket: WebSocket | null = null;
  private opening: Promise<WebSocket> | null = null;
  private active: Active | null = null;

  constructor(
    private readonly url: string = chatSocketUrl(),
    private readonly factory: SocketFactory = (u) => new WebSocket(u),
    private readonly sleep: (ms: number) => Promise<void> = (ms) =>
      new Promise((r) => setTimeout(r, ms)),
  ) {}

  /** Stellt eine Frage; ruft onEvent für jedes Event und endet mit dem Ausgang der Antwort. */
  async ask(
    request: ChatRequest,
    onEvent: (event: ChatStreamEvent) => void,
  ): Promise<ChatOutcome> {
    const socket = await this.connect();
    const streamId = uid();
    return new Promise<ChatOutcome>((resolve) => {
      this.active = {
        streamId,
        onEvent,
        finish: (outcome) => {
          if (this.active?.streamId === streamId) this.active = null;
          resolve(outcome);
        },
      };
      socket.send(JSON.stringify({ type: "message", stream_id: streamId, ...request }));
    });
  }

  /** Bricht die laufende Antwort ab (sofort, ohne auf das Modell zu warten). */
  stop(): void {
    const active = this.active;
    if (!active) return;
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: "stop", stream_id: active.streamId }));
    } else {
      api.post("/chat/stop", { stream_id: active.streamId }).catch(() => {});
      active.finish("cancelled");
    }
  }

  get busy(): boolean {
    return this.active !== null;
  }

  close(): void {
    this.active?.finish("cancelled");
    this.socket?.close();
    this.socket = null;
  }

  private async connect(): Promise<WebSocket> {
    if (this.socket?.readyState === WebSocket.OPEN) return this.socket;
    if (!this.opening) {
      this.opening = this.openWithRetry().finally(() => {
        this.opening = null;
      });
    }
    return this.opening;
  }

  private async openWithRetry(): Promise<WebSocket> {
    for (const delay of RECONNECT_DELAYS) {
      if (delay) await this.sleep(delay);
      try {
        return await this.open();
      } catch {
        // nächster Versuch
      }
    }
    throw new ChatSocketError(
      "Keine Verbindung zum Chat. Möglicherweise ist die Sitzung abgelaufen oder ein Proxy blockiert WebSockets.",
    );
  }

  private open(): Promise<WebSocket> {
    return new Promise((resolve, reject) => {
      const socket = this.factory(this.url);
      let opened = false;
      socket.onopen = () => {
        opened = true;
        this.socket = socket;
        resolve(socket);
      };
      socket.onmessage = (message) => this.dispatch(message.data);
      socket.onerror = () => {
        if (!opened) reject(new ChatSocketError("WebSocket-Fehler"));
      };
      socket.onclose = () => {
        if (this.socket === socket) this.socket = null;
        if (!opened) {
          reject(new ChatSocketError("WebSocket geschlossen"));
          return;
        }
        // Verbindung während einer Antwort verloren: der Server bricht den Stream ab
        this.active?.finish("cancelled");
      };
    });
  }

  private dispatch(data: unknown): void {
    let event: ChatStreamEvent;
    try {
      event = JSON.parse(String(data)) as ChatStreamEvent;
    } catch {
      return;
    }
    const active = this.active;
    if (!active || (event.stream_id && event.stream_id !== active.streamId)) return;
    if (event.type === "ready") return;
    if (event.type === "cancelled") {
      active.finish("cancelled");
      return;
    }
    active.onEvent(event);
    if (event.done) active.finish("done");
    else if (event.error) active.finish("error");
  }
}
