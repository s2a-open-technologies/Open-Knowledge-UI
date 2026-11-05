import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, Loader2, Play } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
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
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { ReindexProgress } from "@/components/ReindexProgress";
import { api, errorMessage } from "@/lib/api";
import { useReindex } from "@/lib/useReindex";
import type {
  EmbeddingOverview,
  ModelEntry,
  ModelsResponse,
  IngestSettings,
  TaskSettings,
  ModelSpec,
  ModelStatus,
} from "@/lib/types";

interface Connection {
  id: number;
  name: string;
  enabled: boolean;
}

// Von fastembed unterstützte Modelle (Auswahl)
const BM25_LABELS: Record<string, string> = {
  german: "Deutsch",
  english: "Englisch",
  french: "Französisch",
  spanish: "Spanisch",
  italian: "Italienisch",
  dutch: "Niederländisch",
  portuguese: "Portugiesisch",
};

const FASTEMBED_MODELS: Record<"embedding" | "reranker", string[]> = {
  embedding: ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
  reranker: ["BAAI/bge-reranker-base", "jinaai/jina-reranker-v2-base-multilingual"],
};

function StatusBadge({ status }: { status?: ModelStatus }) {
  if (!status) return null;
  if (status.state === "ready")
    return (
      <Badge variant="secondary">
        geladen{status.dimension ? ` · ${status.dimension} Dim.` : ""}
        {status.seconds ? ` · ${status.seconds}s` : ""}
      </Badge>
    );
  if (status.state === "loading")
    return (
      <Badge variant="secondary">
        <Loader2 className="mr-1 size-3 animate-spin" />
        lädt …
      </Badge>
    );
  if (status.state === "error")
    return (
      <Badge variant="destructive" title={status.error}>
        Fehler
      </Badge>
    );
  return <Badge variant="outline">noch nicht geladen</Badge>;
}

function ModelEditor({
  kind,
  title,
  hint,
  spec,
  status,
  suggestions,
  connections,
  onChange,
}: {
  kind: "embedding" | "reranker";
  title: string;
  hint: string;
  spec: ModelSpec;
  status?: ModelStatus;
  suggestions: string[];
  connections: Connection[];
  onChange: (s: ModelSpec) => void;
}) {
  const [testing, setTesting] = useState(false);
  const [remote, setRemote] = useState<string[]>([]);
  const listId = `models-${kind}`;

  useEffect(() => {
    if (spec.source !== "endpoint" || !spec.connection_id) {
      setRemote([]);
      return;
    }
    api
      .get<string[]>(`/api/admin/embedding/remote-models?connection_id=${spec.connection_id}&capability=${kind}`)
      .then(setRemote)
      .catch(() => setRemote([]));
  }, [spec.source, spec.connection_id, kind]);

  const test = async () => {
    setTesting(true);
    try {
      const r = await api.post<{ dimension?: number; ranking_ok?: boolean; seconds: number }>(
        "/api/admin/embedding/test",
        { kind, spec },
      );
      toast.success(
        kind === "embedding"
          ? `Funktioniert: ${r.dimension} Dimensionen, ${r.seconds}s`
          : `Funktioniert (${r.seconds}s)${r.ranking_ok === false ? ", aber die Testrangfolge war falsch" : ""}`,
      );
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setTesting(false);
    }
  };

  return (
    <Card>
      <CardContent className="space-y-3 p-4">
        <div className="flex items-center justify-between gap-2">
          <div>
            <h2 className="font-semibold">{title}</h2>
            <p className="text-sm text-muted-foreground">{hint}</p>
          </div>
          <StatusBadge status={status} />
        </div>
        {status?.state === "error" ? (
          <p className="text-sm text-destructive">{status.error}</p>
        ) : null}
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="space-y-1">
            <Label className="text-xs">Quelle</Label>
            <Select
              value={spec.source}
              onValueChange={(v) =>
                onChange({
                  source: v as ModelSpec["source"],
                  model: spec.model,
                  connection_id: v === "endpoint" ? connections[0]?.id : undefined,
                })
              }
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="hf">HuggingFace (lokal)</SelectItem>
                <SelectItem value="endpoint">Endpoint (Connection)</SelectItem>
                <SelectItem value="fastembed">fastembed (ONNX, lokal)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {spec.source === "endpoint" ? (
            <div className="space-y-1">
              <Label className="text-xs">Connection</Label>
              <Select
                value={spec.connection_id ? String(spec.connection_id) : ""}
                onValueChange={(v) => onChange({ ...spec, connection_id: Number(v) })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="wählen" />
                </SelectTrigger>
                <SelectContent>
                  {connections.map((c) => (
                    <SelectItem key={c.id} value={String(c.id)}>
                      {c.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          ) : null}
          <div className="space-y-1 sm:col-span-full">
            <Label className="text-xs">Modell</Label>
            <Input
              list={listId}
              value={spec.model}
              onChange={(e) => onChange({ ...spec, model: e.target.value })}
              placeholder={spec.source === "hf" ? "z. B. BAAI/bge-m3" : spec.source === "fastembed" ? "Modellname laut fastembed" : "Modellname am Endpoint"}
            />
            <datalist id={listId}>
              {(spec.source === "endpoint"
                ? remote
                : spec.source === "fastembed"
                  ? FASTEMBED_MODELS[kind]
                  : suggestions
              ).map((m) => (
                <option key={m} value={m} />
              ))}
            </datalist>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => void test()} disabled={testing}>
            {testing ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
            Testen
          </Button>
          {spec.source === "hf" ? (
            <span className="text-xs text-muted-foreground">
              Wird beim ersten Mal heruntergeladen; große Modelle laufen auf der CPU langsam.
            </span>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}

function TaskCard({
  title,
  switchLabel,
  switchHint,
  promptLabel,
  promptHint,
  task,
  models,
  onChange,
  excerpt,
  deviation,
  defaultModelLabel = "Standardmodell",
}: {
  title: string;
  switchLabel: string;
  switchHint: string;
  promptLabel: string;
  promptHint: string;
  task: TaskSettings;
  models: ModelEntry[];
  onChange: (t: TaskSettings) => void;
  excerpt?: boolean;
  deviation?: boolean;
  defaultModelLabel?: string;
}) {
  const num = (v: string): number | "" => (v === "" ? "" : Number(v));
  return (
    <Card>
      <CardContent className="space-y-3 p-4">
        <h2 className="font-semibold">{title}</h2>
        <div className="flex items-start justify-between rounded-lg border border-border-base px-3 py-2">
          <div>
            <Label className="text-sm">{switchLabel}</Label>
            <p className="text-xs text-muted-foreground">{switchHint}</p>
          </div>
          <Switch checked={task.enabled} onCheckedChange={(v) => onChange({ ...task, enabled: v })} />
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="space-y-1 sm:col-span-3">
            <Label className="text-xs">Modell</Label>
            <Select
              value={task.model_ref || "__default__"}
              onValueChange={(v) => onChange({ ...task, model_ref: v === "__default__" ? "" : v })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__default__">{defaultModelLabel}</SelectItem>
                {models.map((m) => (
                  <SelectItem key={m.ref} value={m.ref}>
                    {m.name} · {m.connection_name}
                    {m.visible ? "" : " (im Chat ausgeblendet)"}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Temperatur</Label>
            <Input
              type="number"
              step="0.1"
              min={0}
              max={2}
              value={task.temperature ?? ""}
              placeholder="Modell"
              onChange={(e) => onChange({ ...task, temperature: num(e.target.value) })}
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Max. Tokens</Label>
            <Input
              type="number"
              min={1}
              value={task.max_tokens ?? ""}
              placeholder="Modell"
              onChange={(e) => onChange({ ...task, max_tokens: num(e.target.value) })}
            />
          </div>
          {excerpt ? (
            <div className="space-y-1">
              <Label className="text-xs">Auszug (Zeichen)</Label>
              <Input
                type="number"
                min={200}
                max={20000}
                value={task.excerpt_chars ?? 2000}
                onChange={(e) => onChange({ ...task, excerpt_chars: Number(e.target.value) })}
              />
            </div>
          ) : null}
          {deviation ? (
            <div className="space-y-1">
              <Label className="text-xs">Zeilenabweichung (%)</Label>
              <Input
                type="number"
                min={0}
                max={100}
                value={task.max_row_deviation ?? 30}
                onChange={(e) => onChange({ ...task, max_row_deviation: Number(e.target.value) })}
              />
            </div>
          ) : null}
        </div>
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <Label className="text-xs">{promptLabel}</Label>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              disabled={task.prompt === task.default_prompt}
              onClick={() => onChange({ ...task, prompt: task.default_prompt })}
            >
              Auf Standard zurücksetzen
            </Button>
          </div>
          <Textarea
            value={task.prompt}
            rows={10}
            className="font-mono text-xs"
            onChange={(e) => onChange({ ...task, prompt: e.target.value })}
          />
          <p className="text-xs text-muted-foreground">{promptHint}</p>
        </div>
      </CardContent>
    </Card>
  );
}

export function AdminEmbeddingPage() {
  const [data, setData] = useState<EmbeddingOverview | null>(null);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [embedding, setEmbedding] = useState<ModelSpec | null>(null);
  const [reranker, setReranker] = useState<ModelSpec | null>(null);
  const [ingest, setIngest] = useState<IngestSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [chatModels, setChatModels] = useState<ModelEntry[]>([]);
  const [tasks, setTasks] = useState<EmbeddingOverview["tasks"] | null>(null);

  const load = useCallback(async () => {
    try {
      const d = await api.get<EmbeddingOverview>("/api/admin/embedding");
      setData(d);
      const { status: _e, ...e } = d.embedding;
      const { status: _r, ...r } = d.reranker;
      setEmbedding((prev) => prev ?? e);
      setReranker((prev) => prev ?? r);
      setIngest((prev) => prev ?? d.ingest);
      setTasks((prev) => prev ?? d.tasks);
    } catch (err) {
      toast.error(errorMessage(err));
    }
  }, []);

  const { status: reindex, refresh } = useReindex(() => void load());

  useEffect(() => {
    void load();
    api.get<Connection[]>("/api/connections").then(setConnections).catch(() => {});
    api
      .get<ModelsResponse>("/api/models")
      .then((r) => setChatModels(r.models.filter((m) => m.available)))
      .catch(() => {});
  }, [load]);

  // Ladestatus der Modelle aktualisieren, solange eines noch lädt
  useEffect(() => {
    const loading = [data?.embedding.status, data?.reranker.status].some(
      (s) => s?.state === "loading",
    );
    if (!loading) return;
    const t = setInterval(async () => {
      try {
        setData(await api.get<EmbeddingOverview>("/api/admin/embedding"));
      } catch {
        /* nur Anzeige */
      }
    }, 2000);
    return () => clearInterval(t);
  }, [data]);

  const save = async () => {
    if (!embedding || !reranker || !ingest || !tasks) return;
    setSaving(true);
    try {
      setData(await api.put<EmbeddingOverview>("/api/admin/embedding", { embedding, reranker, ingest, tasks }));
      toast.success("Gespeichert. Neue Modelle werden im Hintergrund geladen.");
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setSaving(false);
    }
  };

  const reindexAll = async () => {
    try {
      await api.post("/api/reindex-all");
      await refresh();
    } catch (e) {
      toast.error(errorMessage(e));
    }
  };

  if (!data || !embedding || !reranker || !ingest || !tasks) {
    return <div className="p-6 text-sm text-muted-foreground">Lädt …</div>;
  }

  const setNum = (key: "chunk_size" | "chunk_overlap", v: string) =>
    setIngest({ ...ingest, [key]: Number(v) });

  return (
    <div className="h-full overflow-y-auto">
    <div className="mx-auto max-w-5xl space-y-6 p-6">
      <div>
        <h1 className="text-xl font-semibold">Einlesen & Modelle</h1>
        <p className="text-sm text-muted-foreground">
          Ein Embedding-Modell gilt für alle Knowledge-Bereiche. Die Vektor-Dimension
          wird automatisch aus dem Modell bestimmt.
          {data.device ? ` Rechengerät: ${data.device === "mps" ? "Apple Silicon (MPS)" : data.device === "cuda" ? "NVIDIA-GPU (CUDA)" : "CPU"}.` : ""}
        </p>
      </div>

      {data.stale.length > 0 ? (
        <div className="space-y-2 rounded-lg border border-amber-500/50 bg-amber-500/10 p-3 text-sm">
          <div className="flex items-center gap-2 font-medium">
            <AlertTriangle className="size-4" />
            {data.stale.length} Knowledge-Bereich(e) müssen neu indexiert werden
          </div>
          <p className="text-muted-foreground">
            {data.stale.map((k) => k.name).join(", ")}. Bis dahin werden sie weiter mit ihrem
            bisherigen Modell durchsucht, die Suche funktioniert also.
          </p>
          {data.stale.some((k) => k.error) ? (
            <p className="text-destructive">
              Letzter Fehler: {data.stale.find((k) => k.error)?.error}
            </p>
          ) : null}
          <Button size="sm" onClick={() => void reindexAll()} disabled={reindex.running}>
            Alle neu indexieren
          </Button>
        </div>
      ) : null}
      <ReindexProgress status={reindex} />

      <ModelEditor
        kind="embedding"
        title="Embedding"
        hint="Macht aus Text Vektoren für die Suche. Ein Wechsel erfordert Neu-Indexieren."
        spec={embedding}
        status={data.embedding.status}
        suggestions={data.suggestions.embedding}
        connections={connections.filter((c) => c.enabled)}
        onChange={setEmbedding}
      />
      <ModelEditor
        kind="reranker"
        title="Reranking"
        hint="Sortiert die Treffer neu. Ein Wechsel wirkt sofort, ohne Neu-Indexieren."
        spec={reranker}
        status={data.reranker.status}
        suggestions={data.suggestions.reranker}
        connections={connections.filter((c) => c.enabled)}
        onChange={setReranker}
      />

      <TaskCard
        title="Bildbeschreibung"
        switchLabel="Bilder beschreiben"
        switchHint="Aus: Bilder in Dokumenten werden übersprungen (schneller, aber ohne Bildinhalt)."
        promptLabel="Anweisung (Systemprompt) für das Bildmodell"
        promptHint="Das Modell braucht Bildverständnis. Die Antwort steht als [BILD-BESCHREIBUNG: …] im Text."
        task={tasks.vision}
        models={chatModels}
        onChange={(t) => setTasks({ ...tasks, vision: t })}
      />
      <TaskCard
        title="Tabellen prüfen"
        switchLabel="Tabellen aus PDFs per Bildmodell prüfen"
        switchHint="Das Bildmodell liest jede Tabelle noch einmal vom Bild und korrigiert Zellen, die die Layout-Erkennung falsch zugeordnet hat (ein Modellaufruf je Tabelle, nur neu eingelesene PDFs). Wirkt nur bei eingeschalteter Bildbeschreibung."
        promptLabel="Anweisung für die Tabellenprüfung"
        promptHint="Die Antwort muss eine Markdown-Tabelle sein. Weicht ihre Zeilenzahl zu stark vom Entwurf ab (Zeilenabweichung), bleibt die Tabelle der Layout-Erkennung."
        task={tasks.tables}
        models={chatModels}
        onChange={(t) => setTasks({ ...tasks, tables: t })}
        deviation
        defaultModelLabel="Modell der Bildbeschreibung"
      />
      <TaskCard
        title="Bilder im Chat"
        switchLabel="Bilder im Chat annehmen"
        switchHint="Das Bildmodell liest ein angehängtes Bild (Foto, Screenshot, Scan) und gibt den Text daraus wieder; dieser Text wird wie ein Anhang durchsuchbar. Das Bild selbst geht nicht an das Chat-Modell."
        promptLabel="Anweisung für das Bildmodell"
        promptHint="Das Modell braucht Bildverständnis. Gelesen wird einmal beim Hochladen; die Antwort wird in Abschnitte geteilt und durchsucht."
        task={tasks.images}
        models={chatModels}
        onChange={(t) => setTasks({ ...tasks, images: t })}
        defaultModelLabel="Modell der Bildbeschreibung"
      />
      <TaskCard
        title="Kontext-Anreicherung"
        switchLabel="Kontext je Chunk per LLM ergänzen"
        switchHint="Bessere Treffer, aber deutlich langsameres Einlesen (ein Modellaufruf je Abschnitt)."
        promptLabel="Prompt"
        promptHint="Platzhalter: {dokument} = Auszug aus dem Dokument, {abschnitt} = der Abschnitt (Pflicht)."
        task={tasks.context}
        models={chatModels}
        onChange={(t) => setTasks({ ...tasks, context: t })}
        excerpt
      />

      <Card>
        <CardContent className="space-y-3 p-4">
          <h2 className="font-semibold">Zerlegen</h2>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1">
              <Label className="text-xs">Chunk-Größe (Zeichen)</Label>
              <Input
                type="number"
                value={ingest.chunk_size}
                onChange={(e) => setNum("chunk_size", e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <Label className="text-xs">Überlappung</Label>
              <Input
                type="number"
                value={ingest.chunk_overlap}
                onChange={(e) => setNum("chunk_overlap", e.target.value)}
              />
            </div>
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Sprache der Stichwortsuche (BM25)</Label>
            <Select
              value={ingest.bm25_language}
              onValueChange={(v) => setIngest({ ...ingest, bm25_language: v })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(data.bm25_languages ?? [ingest.bm25_language]).map((l) => (
                  <SelectItem key={l} value={l}>
                    {BM25_LABELS[l] ?? l}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Stemmer und Stoppwörter der Stichwortsuche: Wortformen („Kündigungsfristen“ – „Kündigungsfrist“) finden sich, „der“, „und“ zählen nicht. Gilt für neue und neu
              indexierte Knowledge; bestehende bleiben in ihrer Sprache korrekt und melden „Index veraltet“, bis sie neu indexiert werden.
              Zusammengesetzte Wörter trennt der Stemmer nicht.
            </p>
          </div>
          {(
            [
              ["keep_original", "Originaldatei behalten", "Standard aus (Datensparsamkeit): nur das Markdown bleibt, Neu-Indexieren nutzt dieses"],
              ["index_enabled", "Inhaltsverzeichnis (index.md) pro Knowledge", "das Modell liest es zuerst; spart Suchschritte"],
            ] as const
          ).map(([key, label, hint]) => (
            <div
              key={key}
              className="flex items-center justify-between rounded-lg border border-border-base px-3 py-2"
            >
              <div>
                <Label className="text-sm">{label}</Label>
                <p className="text-xs text-muted-foreground">{hint}</p>
              </div>
              <Switch
                checked={ingest[key]}
                onCheckedChange={(v) => setIngest({ ...ingest, [key]: v })}
              />
            </div>
          ))}
          <p className="text-xs text-muted-foreground">
            Änderungen an Größe, Überlappung und Anreicherung gelten für neue Dokumente; bestehende
            werden beim Neu-Indexieren angepasst.
          </p>
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button onClick={() => void save()} disabled={saving}>
          {saving ? <Loader2 className="size-4 animate-spin" /> : null}
          Speichern
        </Button>
      </div>
    </div>
    </div>
  );
}
