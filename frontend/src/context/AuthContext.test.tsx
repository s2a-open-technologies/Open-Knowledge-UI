// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AuthProvider, KEEPALIVE_EVERY_MS, useAuth } from "@/context/AuthContext";
import { api } from "@/lib/api";

const USER = { id: 1, username: "anna", role: "user" };

function mockApi() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (url === "/api/auth/me") return new Response(JSON.stringify({ user: USER }));
      return new Response(JSON.stringify({ user: { ...USER, username: "ben" } }));
    }),
  );
}

function setup() {
  const queryClient = new QueryClient();
  let auth!: ReturnType<typeof useAuth>;
  function Probe() {
    auth = useAuth();
    return null;
  }
  render(
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <Probe />
      </AuthProvider>
    </QueryClientProvider>,
  );
  return { queryClient, auth: () => auth };
}

beforeEach(mockApi);
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("AuthProvider und Zwischenspeicher", () => {
  it("leert den Zwischenspeicher beim Abmelden", async () => {
    const { queryClient, auth } = setup();
    await waitFor(() => expect(auth().loading).toBe(false));
    queryClient.setQueryData(["my-uploads"], [{ id: "geheim" }]);
    await act(async () => auth().logout());
    expect(queryClient.getQueryData(["my-uploads"])).toBeUndefined();
  });

  it("leert den Zwischenspeicher beim Anmelden", async () => {
    const { queryClient, auth } = setup();
    await waitFor(() => expect(auth().loading).toBe(false));
    queryClient.setQueryData(["my-uploads"], [{ id: "vom-vorherigen-nutzer" }]);
    await act(async () => auth().login("ben", "pw"));
    expect(queryClient.getQueryData(["my-uploads"])).toBeUndefined();
  });

  it("meldet ab und leert den Zwischenspeicher, wenn die Sitzung abgelaufen ist", async () => {
    const { queryClient, auth } = setup();
    await waitFor(() => expect(auth().authenticated).toBe(true));
    queryClient.setQueryData(["my-uploads"], [{ id: "geheim" }]);
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { status: 401 })));
    await act(async () => {
      await api.get("/api/my-uploads").catch(() => undefined);
    });
    expect(auth().authenticated).toBe(false);
    expect(queryClient.getQueryData(["my-uploads"])).toBeUndefined();
  });

  it("behandelt eine falsche Anmeldung nicht als abgelaufene Sitzung", async () => {
    const { queryClient, auth } = setup();
    await waitFor(() => expect(auth().authenticated).toBe(true));
    queryClient.setQueryData(["x"], 1);
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { status: 401 })));
    await act(async () => {
      await api.post("/api/auth/login", {}).catch(() => undefined);
    });
    expect(auth().authenticated).toBe(true);
    expect(queryClient.getQueryData(["x"])).toBe(1);
  });

  it("sendet bei Bedienung höchstens alle 5 Minuten ein Lebenszeichen", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    try {
      const { auth } = setup();
      await vi.waitFor(() => expect(auth().authenticated).toBe(true));
      const fetchMock = vi.fn(async (_url: string) => new Response("{}"));
      vi.stubGlobal("fetch", fetchMock);
      const calls = () =>
        fetchMock.mock.calls.filter(([url]) => url === "/api/auth/keepalive").length;

      window.dispatchEvent(new Event("pointerdown"));
      expect(calls()).toBe(0); // gerade erst angemeldet: noch nicht nötig

      vi.setSystemTime(Date.now() + KEEPALIVE_EVERY_MS + 1000);
      window.dispatchEvent(new Event("keydown"));
      window.dispatchEvent(new Event("pointerdown"));
      expect(calls()).toBe(1); // zweites Ereignis gleich danach zählt nicht
    } finally {
      vi.useRealTimers();
    }
  });
});
