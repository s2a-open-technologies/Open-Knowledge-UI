import { useCallback, useEffect, useState } from "react";
import { Loader2, Mic, Play, Search, ShieldCheck } from "lucide-react";
import type { ReactNode } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Field } from "@/components/ui/field";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { CompactionCard } from "@/components/CompactionCard";
import { ToolWorkerCard } from "@/components/ToolWorkerCard";
import { api, errorMessage } from "@/lib/api";

interface ToolsOverview {
  stt: { base_url: string; model: string; mode: string; has_key: boolean };
  search: { url: string; categories: string; allow_unmasked: boolean };
  pii: { url: string; enabled: boolean; has_key: boolean; timeout_s?: number; max_chars?: number };
  web_search_available: boolean;
  web_search_masked: boolean;
}

type Kind = "stt" | "search" | "pii";

function ToolCard({
  icon,
  title,
  hint,
  badge,
  onTest,
  testing,
  children,
}: {
  icon: ReactNode;
  title: string;
  hint: string;
  badge?: ReactNode;
  onTest: () => void;
  testing: boolean;
  children: ReactNode;
}) {
  return (
    <Card>
      <CardContent className="space-y-3 p-4">
        <div className="flex items-start justify-between gap-2">
          <div>
            <h2 className="flex items-center gap-2 font-semibold">
              {icon}
              {title}
            </h2>
            <p className="text-sm text-muted-foreground">{hint}</p>
          </div>
          {badge}
        </div>
        <div className="grid gap-3 sm:grid-cols-2">{children}</div>
        <Button variant="outline" size="sm" onClick={onTest} disabled={testing}>
          {testing ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
          Testen
        </Button>
      </CardContent>
    </Card>
  );
}

export function AdminToolsPage() {
  const [data, setData] = useState<ToolsOverview | null>(null);
  const [stt, setStt] = useState({ base_url: "", model: "", mode: "auto", api_key: "" });
  const [search, setSearch] = useState({ url: "", categories: "general", allow_unmasked: false });
  const [pii, setPii] = useState({ url: "", enabled: true, api_key: "", timeout_s: 10, max_chars: 12000 });
  const [testing, setTesting] = useState<Kind | null>(null);
  const [saving, setSaving] = useState(false);

  const apply = (d: ToolsOverview) => {
    setData(d);
    setStt({ base_url: d.stt.base_url, model: d.stt.model, mode: d.stt.mode || "auto", api_key: "" });
    setSearch({ url: d.search.url, categories: d.search.categories, allow_unmasked: d.search.allow_unmasked });
    setPii({ url: d.pii.url, enabled: d.pii.enabled, api_key: "", timeout_s: d.pii.timeout_s ?? 10, max_chars: d.pii.max_chars ?? 12000 });
  };

  const load = useCallback(async () => {
    try {
      apply(await api.get<ToolsOverview>("/api/admin/tools"));
    } catch (e) {
      toast.error(errorMessage(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const save = async () => {
    setSaving(true);
    try {
      apply(await api.put<ToolsOverview>("/api/admin/tools", { stt, search, pii }));
      toast.success("Gespeichert");
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setSaving(false);
    }
  };

  const test = async (kind: Kind) => {
    const values = kind === "stt" ? stt : kind === "search" ? search : pii;
    setTesting(kind);
    try {
      const r = await api.post<{ message?: string; details?: string }>("/api/admin/tools/test", {
        kind,
        values,
      });
      toast.success((r.message || "Test erfolgreich") + (r.details ? ` – ${r.details}` : ""));
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setTesting(null);
    }
  };

  if (!data) return <div className="p-6 text-sm text-muted-foreground">Lädt …</div>;

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-6 p-6">
        <div>
          <h1 className="text-xl font-semibold">Tools</h1>
          <p className="text-sm text-muted-foreground">
            Zusatzdienste für Spracheingabe und Websuche sowie die automatische Compaction des Chat-Kontexts. Die Websuche schaltest du je Modell
            zu (Modelle › Konfigurieren). Sie läuft mit aktivem PII-Filter maskiert, ohne Filter nur mit deiner ausdrücklichen Freigabe.
          </p>
        </div>

        <ToolCard
          icon={<Mic className="size-4" />}
          title="Spracheingabe"
          hint="Speech-to-Text für den Mikrofon-Knopf im Chat. Parakeet schreibt live ins Feld; Whisper und jeder andere OpenAI-kompatible Dienst (vLLM, speaches, OpenAI, Groq …) funktionieren über Aufnehmen & senden."
          badge={
            <Badge variant={data.stt.base_url ? "secondary" : "outline"}>
              {data.stt.base_url ? "eingerichtet" : "nicht eingerichtet"}
            </Badge>
          }
          onTest={() => void test("stt")}
          testing={testing === "stt"}
        >
          <Field label="Adresse (Base-URL)">
            <Input
              value={stt.base_url}
              onChange={(e) => setStt({ ...stt, base_url: e.target.value })}
              placeholder="http://parakeet:5001/v1 oder http://whisper:8000/v1"
            />
          </Field>
          <Field label="Modell">
            <Input
              value={stt.model}
              onChange={(e) => setStt({ ...stt, model: e.target.value })}
              placeholder="nemo-parakeet-tdt-0.6b-v3 oder whisper-large-v3"
            />
          </Field>
          <Field
            label="Betriebsart"
            hint="Automatisch: Live-Streaming, wenn der Dienst es anbietet, sonst Aufnehmen & senden."
          >
            <Select value={stt.mode} onValueChange={(v) => setStt({ ...stt, mode: v })}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Automatisch</SelectItem>
                <SelectItem value="stream">Live-Streaming (Parakeet)</SelectItem>
                <SelectItem value="upload">Aufnehmen & senden (Whisper & Co.)</SelectItem>
              </SelectContent>
            </Select>
          </Field>
          <Field
            label="API-Key"
            hint={data.stt.has_key ? "Ein Key ist gespeichert. Leer lassen, um ihn zu behalten." : undefined}
          >
            <Input
              type="password"
              value={stt.api_key}
              onChange={(e) => setStt({ ...stt, api_key: e.target.value })}
              placeholder={data.stt.has_key ? "••••••••" : "optional"}
            />
          </Field>
        </ToolCard>

        <ToolCard
          icon={<ShieldCheck className="size-4" />}
          title="PII-Filter"
          hint="Maskiert personenbezogene Daten in Anfragen, bevor sie das System verlassen. Jeder Dienst mit der unten beschriebenen Schnittstelle funktioniert (entityguard ist ein Beispiel)."
          badge={
            <Badge variant={data.pii.url && data.pii.enabled ? "secondary" : "outline"}>
              {data.pii.url && data.pii.enabled ? "aktiv" : "aus"}
            </Badge>
          }
          onTest={() => void test("pii")}
          testing={testing === "pii"}
        >
          <Field label="Adresse (vollständiger Endpoint)">
            <Input
              value={pii.url}
              onChange={(e) => setPii({ ...pii, url: e.target.value })}
              placeholder="http://entityguard:9500/api/v1/sanitize"
            />
          </Field>
          <Field
            label="API-Key"
            hint={data.pii.has_key ? "Ein Key ist gespeichert. Leer lassen, um ihn zu behalten." : undefined}
          >
            <Input
              type="password"
              value={pii.api_key}
              onChange={(e) => setPii({ ...pii, api_key: e.target.value })}
              placeholder={data.pii.has_key ? "••••••••" : "optional"}
            />
          </Field>
          <div className="flex items-center justify-between rounded-lg border border-border-base px-3 py-2 sm:col-span-2">
            <div>
              <Label className="text-sm">PII-Filter verwenden</Label>
              <p className="text-xs text-muted-foreground">
                Aus: Nachrichten gehen ungefiltert ans Modell; die Websuche braucht dann die Freigabe „Auch ohne PII-Filter erlauben“.
              </p>
            </div>
            <Switch
              checked={pii.enabled}
              onCheckedChange={(v) => setPii({ ...pii, enabled: v })}
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Zeitlimit (Sekunden)</Label>
            <Input
              type="number"
              min={2}
              max={120}
              value={pii.timeout_s}
              onChange={(e) => setPii({ ...pii, timeout_s: Number(e.target.value) })}
            />
            <p className="text-xs text-muted-foreground">
              Grundwert je Aufruf, wächst mit der Textlänge; ein zweiter Versuch bekommt das doppelte Limit.
            </p>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Höchstlänge je Aufruf (Zeichen)</Label>
            <Input
              type="number"
              min={2000}
              max={100000}
              value={pii.max_chars}
              onChange={(e) => setPii({ ...pii, max_chars: Number(e.target.value) })}
            />
            <p className="text-xs text-muted-foreground">
              Längere Texte (langer Verlauf, große Ergebnisse) gehen in Stücken an den Filter. Sehr große Texte können
              manche Filter überlasten.
            </p>
          </div>
          <details className="rounded-lg border border-border-base px-3 py-2 text-xs text-muted-foreground sm:col-span-2">
            <summary className="cursor-pointer text-sm font-medium text-foreground">
              Anforderungen an den Dienst
            </summary>
            <div className="mt-2 space-y-2">
              <p>
                Die App sendet <code>POST &lt;Adresse&gt;</code> mit JSON und erwartet <code>200</code> mit dem
                maskierten Text und, für das Zurückübersetzen der Antwort, einem Mapping. Optional kommt der Key als
                <code> Authorization: Bearer …</code>.
              </p>
              <pre className="overflow-x-auto rounded bg-surface-muted p-2">{`Anfrage:  {"text": "Schreibe Max Mustermann"}
Antwort:  {"filtered": "Schreibe [NAME_1]",
           "mapping": {"[NAME_1]": "Max Mustermann"}}`}</pre>
              <p>
                Platzhalter müssen im Text eindeutig sein. Ein Dienst ohne <code>mapping</code> funktioniert für die
                Websuche, die Antworten des Modells lassen sich dann aber nicht zurückübersetzen. Nicht-200 gilt als
                Fehler. Die ganze Beschreibung steht in <code>docs/sanitizer-api.md</code>. „Testen“ prüft all das
                mit erfundenen Daten.
              </p>
            </div>
          </details>
        </ToolCard>

        <ToolCard
          icon={<Search className="size-4" />}
          title="Websuche (SearXNG)"
          hint="Metasuchmaschine, die ein Modell als Werkzeug nutzen darf. Standard ist aus; zuschaltbar je Modell."
          badge={
            <Badge variant={data.web_search_available ? "secondary" : "outline"}>
              {data.web_search_available
                ? data.web_search_masked
                  ? "bereit (maskiert)"
                  : "bereit, ohne Maskierung"
                : data.search.url
                  ? "braucht PII-Filter oder Freigabe"
                  : "nicht eingerichtet"}
            </Badge>
          }
          onTest={() => void test("search")}
          testing={testing === "search"}
        >
          <Field label="Adresse">
            <Input
              value={search.url}
              onChange={(e) => setSearch({ ...search, url: e.target.value })}
              placeholder="http://searxng:8080"
            />
          </Field>
          <Field label="Kategorien" hint="Komma-getrennt, z. B. general,news,science">
            <Input
              value={search.categories}
              onChange={(e) => setSearch({ ...search, categories: e.target.value })}
            />
          </Field>
          <div className="flex items-center justify-between gap-3 rounded-lg border border-border-base px-3 py-2 sm:col-span-2">
            <div>
              <Label className="text-sm">Auch ohne PII-Filter erlauben</Label>
              <p className="text-xs text-muted-foreground">
                Ohne aktiven PII-Filter gehen Suchanfragen unmaskiert an SearXNG und von dort an Suchmaschinen.
                Zuschalten musst du die Websuche weiterhin je Modell.
              </p>
            </div>
            <Switch
              checked={search.allow_unmasked}
              onCheckedChange={(v) => setSearch({ ...search, allow_unmasked: v })}
            />
          </div>
        </ToolCard>

        <div className="flex justify-end">
          <Button onClick={() => void save()} disabled={saving}>
            {saving ? <Loader2 className="size-4 animate-spin" /> : null}
            Speichern
          </Button>
        </div>

        <CompactionCard />
        <ToolWorkerCard />
      </div>
    </div>
  );
}
