import { useQueryClient } from "@tanstack/react-query";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { api, setUnauthorizedHandler } from "@/lib/api";
import type { AuthUser, Role } from "@/lib/types";

// Höchstens alle 5 Minuten ein Signal (die Sitzung läuft nach 30 Minuten Leerlauf ab)
export const KEEPALIVE_EVERY_MS = 5 * 60 * 1000;

const ROLE_ORDER: Role[] = ["user", "editor", "admin"];

interface AuthContextValue {
  authenticated: boolean;
  loading: boolean;
  user: AuthUser | null;
  // true, wenn die Rolle des Nutzers mindestens `role` ist (admin ⊃ editor ⊃ user).
  hasRole: (role: Role) => boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    api
      .get<{ user: AuthUser }>("/api/auth/me")
      .then((res) => active && setUser(res.user))
      .catch(() => active && setUser(null))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, []);

  // Abgelaufene Sitzung: Zwischenspeicher leeren und zur Anmeldung zurück
  useEffect(() => {
    setUnauthorizedHandler(() => {
      queryClient.clear();
      setUser(null);
    });
    return () => setUnauthorizedHandler(null);
  }, [queryClient]);

  // Solange jemand bedient (Klick, Taste), hält ein Signal die Sitzung wach. Das Cookie läuft nach
  // 30 Minuten ohne Aktivität ab; der Chat läuft über einen WebSocket und kann es nicht erneuern.
  useEffect(() => {
    if (user === null) return;
    let last = Date.now();
    const onActivity = () => {
      if (Date.now() - last < KEEPALIVE_EVERY_MS) return;
      last = Date.now();
      api.post("/api/auth/keepalive").catch(() => undefined); // 401 meldet der zentrale Handler
    };
    const events = ["pointerdown", "keydown"] as const;
    events.forEach((name) => window.addEventListener(name, onActivity, { passive: true }));
    return () => events.forEach((name) => window.removeEventListener(name, onActivity));
  }, [user]);

  const login = useCallback(async (username: string, password: string) => {
    const res = await api.post<{ user: AuthUser }>("/api/auth/login", {
      username,
      password,
    });
    // Zwischengespeicherte Daten eines früheren Nutzers dürfen nie im neuen Konto auftauchen
    queryClient.clear();
    setUser(res.user);
  }, [queryClient]);

  const logout = useCallback(async () => {
    await api.post("/api/auth/logout");
    queryClient.clear();
    setUser(null);
  }, [queryClient]);

  const hasRole = useCallback(
    (role: Role) =>
      user !== null && ROLE_ORDER.indexOf(user.role) >= ROLE_ORDER.indexOf(role),
    [user],
  );

  return (
    <AuthContext.Provider
      value={{ authenticated: user !== null, loading, user, hasRole, login, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth muss innerhalb von AuthProvider laufen");
  return ctx;
}
