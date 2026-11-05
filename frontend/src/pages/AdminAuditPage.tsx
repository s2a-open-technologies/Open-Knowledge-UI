import { useCallback, useEffect, useState } from "react";
import { Loader2, RefreshCw, Save, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api, errorMessage } from "@/lib/api";

interface AuditEvent {
  id: number;
  created_at: string | null;
  actor_user_id: number | null;
  actor_name: string | null;
  ip: string | null;
  action: string;
  object_type: string | null;
  object_id: string | null;
  detail: string | null;
}

const ACTIONS: { value: string; label: string }[] = [
  { value: "", label: "Alle Aktionen" },
  { value: "login_ok", label: "Anmeldung" },
  { value: "login_failed", label: "Anmeldung fehlgeschlagen" },
  { value: "login_throttled", label: "Anmeldung gesperrt" },
  { value: "login_external_ok", label: "Externe Anmeldung" },
  { value: "login_external_failed", label: "Externe Anmeldung fehlgeschlagen" },
  { value: "logout", label: "Abmeldung" },
  { value: "password_change", label: "Passwort geändert" },
  { value: "data_export", label: "Datenexport" },
  { value: "user_create", label: "Konto angelegt" },
  { value: "user_update", label: "Konto geändert" },
  { value: "user_delete", label: "Konto gelöscht" },
  { value: "group_create", label: "Gruppe angelegt" },
  { value: "group_update", label: "Gruppe geändert" },
  { value: "group_delete", label: "Gruppe gelöscht" },
  { value: "group_members_set", label: "Mitglieder geändert" },
  { value: "access_set", label: "Freigaben geändert" },
  { value: "password_reset_by_admin", label: "Passwort durch Admin gesetzt" },
  { value: "config_change", label: "Konfiguration geändert" },
  { value: "document_view", label: "Fremdes Dokument angesehen" },
];

const PAGE_SIZE = 100;

function actor(e: AuditEvent): string {
  if (e.actor_name) return e.actor_name;
  if (e.actor_user_id !== null && e.actor_user_id !== undefined) return `Konto ${e.actor_user_id}`;
  return "—";
}

function objectLabel(e: AuditEvent): string {
  if (!e.object_type) return "";
  return `${e.object_type} ${e.object_id ?? ""}`.trim();
}

function detail(e: AuditEvent): string {
  if (!e.detail) return "";
  try {
    return Object.entries(JSON.parse(e.detail))
      .map(([k, v]) => `${k}: ${String(v)}`)
      .join(", ");
  } catch {
    return e.detail;
  }
}

// Aufbewahrungsfrist fuer Chats (Datensparsamkeit): 0 = unbegrenzt.
function RetentionCard() {
  const [days, setDays] = useState("0");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    api
      .get<{ chat_retention_days: number }>("/api/admin/retention")
      .then((d) => setDays(String(d.chat_retention_days)))
      .catch((err) => toast.error(errorMessage(err)));
  }, []);

  async function save() {
    setSaving(true);
    try {
      const d = await api.put<{ chat_retention_days: number }>("/api/admin/retention", {
        chat_retention_days: Number(days),
      });
      setDays(String(d.chat_retention_days));
      toast.success("Aufbewahrungsfrist gespeichert");
    } catch (err) {
      toast.error(errorMessage(err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <Card>
      <CardContent className="flex flex-wrap items-end gap-3">
        <div className="w-40">
          <Label className="text-xs">Chats aufbewahren (Tage)</Label>
          <Input
            type="number"
            min={0}
            max={3650}
            value={days}
            onChange={(e) => setDays(e.target.value)}
            className="mt-1"
          />
        </div>
        <p className="min-w-52 flex-1 text-xs text-muted-foreground">
          Chats, die so viele Tage nicht benutzt wurden, werden täglich samt Anhängen gelöscht.
          0 = unbegrenzt aufbewahren.
        </p>
        <Button type="button" onClick={() => void save()} disabled={saving}>
          {saving ? <Loader2 className="size-4 animate-spin" /> : <Save className="size-4" />}
          Speichern
        </Button>
      </CardContent>
    </Card>
  );
}

// Protokoll sicherheitsrelevanter Ereignisse (nur Admin, nur lesend).
export function AdminAuditPage() {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [action, setAction] = useState("");
  const [search, setSearch] = useState("");
  const [offset, setOffset] = useState(0);

  const load = useCallback(
    async (currentOffset: number) => {
      setLoading(true);
      try {
        const params = new URLSearchParams({
          limit: String(PAGE_SIZE),
          offset: String(currentOffset),
        });
        if (action) params.set("action", action);
        const data = await api.get<{ events: AuditEvent[] }>(`/api/admin/audit?${params}`);
        setEvents(data.events);
      } catch (err) {
        toast.error(errorMessage(err));
      } finally {
        setLoading(false);
      }
    },
    [action],
  );

  useEffect(() => {
    void load(offset);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [action, offset]);

  const visible = search
    ? events.filter((e) =>
        [e.actor_name, e.action, e.object_type, e.detail]
          .filter(Boolean)
          .some((v) => String(v).toLowerCase().includes(search.toLowerCase())),
      )
    : events;

  return (
    <div className="space-y-4 overflow-y-auto p-4">
      <RetentionCard />
      <Card>
        <CardContent className="flex flex-wrap items-end gap-3">
          <div className="min-w-52">
            <Label className="text-xs">Aktion</Label>
            <Select value={action} onValueChange={(v) => { setOffset(0); setAction(v); }}>
              <SelectTrigger className="mt-1">
                <SelectValue placeholder="Alle Aktionen" />
              </SelectTrigger>
              <SelectContent>
                {ACTIONS.map((a) => (
                  <SelectItem key={a.value} value={a.value}>
                    {a.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="min-w-52 flex-1">
            <Label className="text-xs">Filter (Name, Aktion, Details)</Label>
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="z. B. alice"
              className="mt-1"
            />
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="secondary"
              onClick={() => { setOffset(0); void load(0); }}
              disabled={loading}
            >
              <RefreshCw className={`size-4 ${loading ? "animate-spin" : ""}`} />
              Aktualisieren
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
              disabled={loading || offset === 0}
            >
              Neuere
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => setOffset((o) => o + PAGE_SIZE)}
              disabled={loading || events.length < PAGE_SIZE}
            >
              Ältere
            </Button>
          </div>
        </CardContent>
      </Card>

      {loading && events.length === 0 ? (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" /> Protokoll wird geladen …
        </p>
      ) : visible.length === 0 ? (
        <p className="flex items-center gap-2 text-sm text-muted-foreground">
          <ShieldCheck className="size-4" /> Keine Einträge. Erst seit dem Upgrade protokolliert.
        </p>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border-base">
          <table className="w-full text-sm">
            <thead className="bg-surface-muted text-left text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="px-3 py-2">Zeit</th>
                <th className="px-3 py-2">Konto</th>
                <th className="px-3 py-2">Aktion</th>
                <th className="px-3 py-2">Objekt</th>
                <th className="px-3 py-2">IP</th>
                <th className="px-3 py-2">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-base">
              {visible.map((e) => (
                <tr key={e.id} className="align-top">
                  <td className="whitespace-nowrap px-3 py-2 text-muted-foreground">{e.created_at}</td>
                  <td className="whitespace-nowrap px-3 py-2 font-medium">{actor(e)}</td>
                  <td className="whitespace-nowrap px-3 py-2">{e.action}</td>
                  <td className="whitespace-nowrap px-3 py-2 text-muted-foreground">{objectLabel(e)}</td>
                  <td className="whitespace-nowrap px-3 py-2 text-muted-foreground">{e.ip ?? ""}</td>
                  <td className="px-3 py-2 text-muted-foreground">{detail(e)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}