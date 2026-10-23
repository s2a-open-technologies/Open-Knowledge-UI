import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Copy,
  RefreshCw,
  Search,
  Settings2,
  Star,
  Share2,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { AccessDialog } from "@/components/AccessDialog";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { ModelConfigDialog } from "@/components/ModelConfigDialog";
import { AdminLink } from "@/components/AdminNav";
import { BehaviorCard } from "@/components/BehaviorCard";
import { api, errorMessage } from "@/lib/api";
import { notifyModelsChanged } from "@/lib/modelChoice";
import { cn } from "@/lib/utils";
import type { AccessInfo, Knowledge, ModelEntry, ModelsResponse } from "@/lib/types";

export function AdminModelsPage() {
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [knowledge, setKnowledge] = useState<Knowledge[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [configTarget, setConfigTarget] = useState<ModelEntry | null>(null);
  const [cloneSource, setCloneSource] = useState<ModelEntry | null>(null);
  const [cloneName, setCloneName] = useState("");
  const [resetTarget, setResetTarget] = useState<ModelEntry | null>(null);
  const [accessTarget, setAccessTarget] = useState<ModelEntry | null>(null);

  const load = useCallback(async (refresh = false) => {
    setLoading(true);
    try {
      setData(
        await api.get<ModelsResponse>(`/api/models${refresh ? "?refresh=1" : ""}`),
      );
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    api
      .get<Knowledge[]>("/api/knowledge")
      .then(setKnowledge)
      .catch((e) => console.error(e));
  }, [load]);

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const map = new Map<string, ModelEntry[]>();
    for (const m of data?.models ?? []) {
      if (q && !`${m.name} ${m.model_id}`.toLowerCase().includes(q)) continue;
      map.set(m.connection_name, [...(map.get(m.connection_name) ?? []), m]);
    }
    return [...map.entries()];
  }, [data, query]);

  const setVisible = async (m: ModelEntry, visible: boolean) => {
    try {
      await api.put("/api/models/config", { ref: m.ref, visible });
      await load();
      notifyModelsChanged();
    } catch (e) {
      toast.error(errorMessage(e));
    }
  };

  const setDefault = async (ref: string) => {
    try {
      await api.put("/api/models/default", { ref });
      setData((prev) => (prev ? { ...prev, default: ref } : prev));
      notifyModelsChanged();
      toast.success("Standardmodell gesetzt");
    } catch (e) {
      toast.error(errorMessage(e));
    }
  };

  const clone = async () => {
    if (!cloneSource || !cloneName.trim()) return;
    try {
      await api.post("/api/models/clone", {
        ref: cloneSource.ref,
        name: cloneName.trim(),
      });
      toast.success("Modell geklont");
      setCloneSource(null);
      await load();
      notifyModelsChanged();
    } catch (e) {
      toast.error(errorMessage(e));
    }
  };

  const reset = async () => {
    if (!resetTarget) return;
    try {
      await api.del(`/api/models/config?ref=${encodeURIComponent(resetTarget.ref)}`);
      await load();
      notifyModelsChanged();
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setResetTarget(null);
    }
  };

  const selectable = (data?.models ?? []).filter((m) => m.visible && m.available);

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-6 p-6">
        <div className="flex flex-col items-start justify-between gap-4 border-b border-border-base pb-4 sm:flex-row sm:items-center">
          <div>
            <h1 className="text-2xl font-semibold">Modelle</h1>
            <p className="text-sm text-muted-foreground">
              Alle Modelle deiner{" "}
              <AdminLink tab="connections">
                Verbindungen
              </AdminLink>
              . Pro Modell Prompt, Parameter und Knowledge anpassen oder klonen.
            </p>
          </div>
          <Button variant="secondary" onClick={() => void load(true)} disabled={loading}>
            <RefreshCw className={loading ? "size-4 animate-spin" : "size-4"} />
            Aktualisieren
          </Button>
        </div>

        <BehaviorCard />

        {data?.errors.map((e) => (
          <div
            key={e.connection_id}
            className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm"
          >
            <AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" />
            <span>
              <strong>{e.connection}:</strong> {e.error}
            </span>
          </div>
        ))}

        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Modelle suchen…"
              className="pl-9"
            />
          </div>
          <div className="flex items-center gap-2">
            <span className="shrink-0 text-sm text-muted-foreground">
              Standardmodell
            </span>
            <Select
              value={selectable.some((m) => m.ref === data?.default) ? data?.default : ""}
              onValueChange={(v) => void setDefault(v)}
            >
              <SelectTrigger className="w-64">
                <SelectValue placeholder="Wählen…" />
              </SelectTrigger>
              <SelectContent>
                {selectable.map((m) => (
                  <SelectItem key={m.ref} value={m.ref}>
                    {m.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {loading && !data ? (
          <p className="text-muted-foreground">Lade…</p>
        ) : groups.length === 0 ? (
          <p className="text-muted-foreground">
            {query ? "Keine Treffer." : "Keine Modelle. Prüfe die Verbindungen."}
          </p>
        ) : (
          groups.map(([connection, models]) => (
            <section key={connection} className="space-y-2">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                {connection}
              </h2>
              <Card>
                <CardContent className="p-0">
                  {models.map((m) => (
                    <div
                      key={m.ref}
                      className="flex flex-wrap items-center gap-3 border-b border-border-base px-4 py-3 transition-colors last:border-0 hover:bg-hover/40"
                    >
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium">{m.name}</p>
                        {m.is_clone || m.name !== m.model_id ? (
                          <p className="truncate text-xs text-muted-foreground">
                            {m.is_clone ? "Basis" : "Original"}: {m.model_id}
                          </p>
                        ) : null}
                        <div className="mt-1 flex flex-wrap gap-1">
                          {m.is_clone ? <Badge variant="secondary">Klon</Badge> : null}
                          {m.customized ? <Badge variant="outline">angepasst</Badge> : null}
                          {m.config?.knowledge_ids?.length ? (
                            <Badge variant="outline">
                              {m.config.knowledge_ids.length} Knowledge
                            </Badge>
                          ) : null}
                          {m.config && !m.config.is_public ? (
                            <Badge variant="outline">eingeschränkt</Badge>
                          ) : null}
                          {!m.available ? (
                            <Badge variant="destructive">nicht verfügbar</Badge>
                          ) : null}
                          {m.ref === data?.default ? <Badge>Standard</Badge> : null}
                        </div>
                      </div>
                      <div className="flex items-center gap-1">
                        <Switch
                          checked={m.visible}
                          onCheckedChange={(v) => void setVisible(m, v)}
                          aria-label="Im Chat sichtbar"
                          title="Im Chat sichtbar"
                        />
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          title={m.ref === data?.default ? "Ist das Standardmodell" : "Zum Standardmodell machen"}
                          disabled={!m.available || m.ref === data?.default}
                          onClick={() => void setDefault(m.ref)}
                        >
                          <Star
                            className={cn("size-4", m.ref === data?.default && "fill-current text-primary")}
                          />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          title="Konfigurieren"
                          onClick={() => setConfigTarget(m)}
                        >
                          <Settings2 className="size-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          title="Freigaben"
                          onClick={() => setAccessTarget(m)}
                        >
                          <Share2 className="size-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          title="Klonen"
                          onClick={() => {
                            setCloneSource(m);
                            setCloneName(`${m.name} (Kopie)`);
                          }}
                        >
                          <Copy className="size-4" />
                        </Button>
                        {m.is_clone || m.customized ? (
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            title={m.is_clone ? "Klon löschen" : "Anpassung zurücksetzen"}
                            onClick={() => setResetTarget(m)}
                          >
                            <Trash2 className="size-4" />
                          </Button>
                        ) : null}
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </section>
          ))
        )}
      </div>

      <ModelConfigDialog
        model={configTarget}
        knowledge={knowledge}
        webSearchAvailable={data?.web_search_available ?? false}
        webSearchMasked={data?.web_search_masked ?? false}
        paramDefaults={data?.param_defaults}
        compaction={data?.compaction}
        onOpenChange={(o) => !o && setConfigTarget(null)}
        onSaved={() => void load().then(notifyModelsChanged)}
      />

      <Dialog open={cloneSource !== null} onOpenChange={(o) => !o && setCloneSource(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Modell klonen</DialogTitle>
          </DialogHeader>
          <Input
            value={cloneName}
            onChange={(e) => setCloneName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && void clone()}
            autoFocus
          />
          <DialogFooter>
            <Button variant="secondary" onClick={() => setCloneSource(null)}>
              Abbrechen
            </Button>
            <Button onClick={clone} disabled={!cloneName.trim()}>
              Klonen
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AccessDialog
        open={accessTarget !== null}
        onOpenChange={(o) => !o && setAccessTarget(null)}
        title="Freigaben"
        description={accessTarget ? `Wer darf „${accessTarget.name}“ im Chat nutzen?` : undefined}
        publicLabel="Für alle angemeldeten Nutzer nutzbar"
        readLabel="Nutzen"
        allowWrite={false}
        load={() =>
          api.get<AccessInfo>(
            `/api/models/access?ref=${encodeURIComponent(accessTarget?.ref ?? "")}`,
          )
        }
        save={async (info) => {
          const saved = await api.put<AccessInfo>("/api/models/access", {
            ref: accessTarget?.ref,
            ...info,
          });
          await load();
      notifyModelsChanged();
          return saved;
        }}
      />
      <ConfirmDialog
        open={resetTarget !== null}
        onOpenChange={(o) => !o && setResetTarget(null)}
        title={resetTarget?.is_clone ? "Klon löschen?" : "Anpassung zurücksetzen?"}
        description={resetTarget?.name}
        confirmLabel={resetTarget?.is_clone ? "Löschen" : "Zurücksetzen"}
        destructive
        onConfirm={reset}
      />
    </div>
  );
}
