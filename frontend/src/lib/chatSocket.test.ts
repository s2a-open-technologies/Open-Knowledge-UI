// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { ChatSocket, ChatSocketError, type ChatStreamEvent } from "@/lib/chatSocket";

// Nachgebauter Browser-WebSocket: der Test steuert Öffnen, Nachrichten und Schließen.
class FakeSocket {
  static OPEN = 1;
  readyState = 0;
  sent: Record<string, unknown>[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((m: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;

  send(data: string) {
    this.sent.push(JSON.parse(data));
  }
  open() {
    this.readyState = FakeSocket.OPEN;
    this.onopen?.();
  }
  receive(event: ChatStreamEvent) {
    this.onmessage?.({ data: JSON.stringify(event) });
  }
  close() {
    this.readyState = 3;
    this.onclose?.();
  }
}

const REQUEST = { message: "Frage", session_id: null, knowledge_ids: null, attachment_ids: [] };

function setup(autoOpen = true) {
  const sockets: FakeSocket[] = [];
  const client = new ChatSocket(
    "ws://test/ws/chat",
    () => {
      const s = new FakeSocket();
      sockets.push(s);
      if (autoOpen) queueMicrotask(() => s.open());
      return s as unknown as WebSocket;
    },
    async () => {},
  );
  return { client, sockets };
}

async function flush() {
  await new Promise((r) => setTimeout(r, 0));
}

afterEach(() => vi.restoreAllMocks());

describe("ChatSocket", () => {
  it("sendet die Frage mit stream_id und reicht die Events bis done weiter", async () => {
    const { client, sockets } = setup();
    const events: ChatStreamEvent[] = [];
    const outcome = client.ask(REQUEST, (e) => events.push(e));
    await flush();
    const [frame] = sockets[0].sent;
    expect(frame).toMatchObject({ type: "message", message: "Frage" });
    const id = frame.stream_id as string;
    sockets[0].receive({ type: "ready", stream_id: id });
    sockets[0].receive({ token: "Hallo", stream_id: id });
    sockets[0].receive({ token: "Fremd", stream_id: "anderer-stream" });
    sockets[0].receive({ done: true, session_id: "s1", stream_id: id });
    await expect(outcome).resolves.toBe("done");
    expect(events.map((e) => e.token ?? (e.done ? "done" : ""))).toEqual(["Hallo", "done"]);
    expect(client.busy).toBe(false);
  });

  it("stoppt per Frame und endet bei cancelled", async () => {
    const { client, sockets } = setup();
    const outcome = client.ask(REQUEST, () => {});
    await flush();
    const id = sockets[0].sent[0].stream_id;
    client.stop();
    expect(sockets[0].sent[1]).toEqual({ type: "stop", stream_id: id });
    sockets[0].receive({ type: "cancelled", stream_id: id as string });
    await expect(outcome).resolves.toBe("cancelled");
  });

  it("gilt als abgebrochen, wenn die Verbindung waehrend der Antwort wegbricht", async () => {
    const { client, sockets } = setup();
    const outcome = client.ask(REQUEST, () => {});
    await flush();
    sockets[0].close();
    await expect(outcome).resolves.toBe("cancelled");
  });

  it("stoppt per HTTP, wenn der Socket nicht mehr offen ist", async () => {
    const post = vi.spyOn(api, "post").mockResolvedValue(undefined as never);
    const { client, sockets } = setup();
    const outcome = client.ask(REQUEST, () => {});
    await flush();
    const id = sockets[0].sent[0].stream_id;
    sockets[0].readyState = 3; // geschlossen, ohne dass onclose schon kam
    client.stop();
    expect(post).toHaveBeenCalledWith("/chat/stop", { stream_id: id });
    await expect(outcome).resolves.toBe("cancelled");
  });

  it("meldet einen Fehler-Frame als error", async () => {
    const { client, sockets } = setup();
    const events: ChatStreamEvent[] = [];
    const outcome = client.ask(REQUEST, (e) => events.push(e));
    await flush();
    const id = sockets[0].sent[0].stream_id as string;
    sockets[0].receive({ error: "Kein Zugriff auf dieses Modell.", stream_id: id });
    await expect(outcome).resolves.toBe("error");
    expect(events[0].error).toBe("Kein Zugriff auf dieses Modell.");
  });

  it("versucht es mehrmals und meldet dann klar, dass kein WebSocket moeglich ist", async () => {
    const { client, sockets } = setup(false);
    const outcome = client.ask(REQUEST, () => {});
    for (let i = 0; i < 3; i++) {
      await flush();
      sockets[i].close(); // Proxy blockiert / Anmeldung abgelaufen
    }
    await expect(outcome).rejects.toBeInstanceOf(ChatSocketError);
    expect(sockets).toHaveLength(3);
  });

  it("nutzt die offene Verbindung fuer die naechste Frage weiter", async () => {
    const { client, sockets } = setup();
    const first = client.ask(REQUEST, () => {});
    await flush();
    sockets[0].receive({ done: true, stream_id: sockets[0].sent[0].stream_id as string });
    await first;
    const second = client.ask(REQUEST, () => {});
    await flush();
    expect(sockets).toHaveLength(1);
    sockets[0].receive({ done: true, stream_id: sockets[0].sent[1].stream_id as string });
    await expect(second).resolves.toBe("done");
  });
});
