import { useState } from "react";
import { KeyRound, Pencil, Plug, Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import {
  ConnectionDialog,
  type ConnectionValues,
} from "@/components/ConnectionDialog";
import { AdminLink } from "@/components/AdminNav";
import { api, errorMessage } from "@/lib/api";
import { useApiQuery } from "@/lib/useApiQuery";
import type { Connection } from "@/lib/types";

export function AdminConnectionsPage() {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editing, setEditing] = useState<Connection | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Connection | null>(null);

  const {
    data: items,
    loading,
    reload: load,
    setData: setItems,
  } = useApiQuery<Connection[]>(["connections"], () => api.get("/api/connections"), []);

  const save = async (values: ConnectionValues) => {
    try {
      if (editing) {
        await api.patch(`/api/connections/${editing.id}`, values);
      } else {
        await api.post("/api/connections", values);
      }
      await load();
    } catch (e) {
      toast.error(errorMessage(e));
      throw e;
    }
  };

  const toggle = async (c: Connection, enabled: boolean) => {
    setItems((prev) => prev.map((x) => (x.id === c.id ? { ...x, enabled } : x)));
    try {
      await api.patch(`/api/connections/${c.id}`, { enabled });
    } catch (e) {
      toast.error(errorMessage(e));
      await load();
    }
  };

  const remove = async () => {
    if (!deleteTarget) return;
    try {
      await api.del(`/api/connections/${deleteTarget.id}`);
      toast.success("Connection gelöscht");
      await load();
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setDeleteTarget(null);
    }
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-6 p-6">
        <div className="flex flex-col items-start justify-between gap-4 border-b border-border-base pb-4 sm:flex-row sm:items-center">
          <div>
            <h1 className="text-2xl font-semibold">Verbindungen</h1>
            <p className="text-sm text-muted-foreground">
              LLM-Endpunkte (OpenAI-kompatibel). Ihre Modelle erscheinen unter{" "}
              <AdminLink tab="models">
                Modelle
              </AdminLink>
              .
            </p>
          </div>
          <Button
            onClick={() => {
              setEditing(null);
              setDialogOpen(true);
            }}
          >
            <Plus className="size-4" />
            Neue Connection
          </Button>
        </div>

        {loading ? (
          <p className="text-muted-foreground">Lade…</p>
        ) : items.length === 0 ? (
          <p className="text-muted-foreground">Noch keine Connections.</p>
        ) : (
          <div className="grid grid-cols-1 gap-2 md:grid-cols-2 2xl:grid-cols-3">
            {items.map((c) => (
              <Card key={c.id}>
                <CardContent className="flex items-center gap-2.5 p-3">
                  <div className="flex size-8 shrink-0 items-center justify-center rounded-md bg-surface-muted">
                    <Plug className="size-4 text-muted-foreground" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="flex items-center gap-1.5 truncate text-sm font-semibold">
                      <span className="truncate">{c.name}</span>
                      {c.kind !== "llm" ? (
                        <span className="shrink-0 rounded bg-surface-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                          {c.kind === "tei" ? "TEI" : "Infinity"}
                        </span>
                      ) : null}
                      <KeyRound
                        className={`size-3 shrink-0 ${c.has_key ? "text-muted-foreground" : "text-muted-foreground/40"}`}
                        aria-label={c.has_key ? "API-Key gesetzt" : "Kein API-Key"}
                      >
                        <title>{c.has_key ? "API-Key gesetzt" : "Kein API-Key"}</title>
                      </KeyRound>
                    </p>
                    <p className="truncate text-xs text-muted-foreground" title={c.base_url}>
                      {c.base_url}
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-0.5">
                    <Switch
                      checked={c.enabled}
                      onCheckedChange={(v) => void toggle(c, v)}
                      aria-label="Aktiv"
                    />
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      title="Bearbeiten"
                      onClick={() => {
                        setEditing(c);
                        setDialogOpen(true);
                      }}
                    >
                      <Pencil className="size-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      title="Löschen"
                      onClick={() => setDeleteTarget(c)}
                    >
                      <Trash2 className="size-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>

      <ConnectionDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        connection={editing}
        onSubmit={save}
      />
      <ConfirmDialog
        open={deleteTarget !== null}
        onOpenChange={(o) => !o && setDeleteTarget(null)}
        title="Connection löschen?"
        description={
          deleteTarget
            ? `„${deleteTarget.name}“ samt aller Modell-Anpassungen und Klone wird gelöscht.`
            : undefined
        }
        confirmLabel="Löschen"
        destructive
        onConfirm={remove}
      />
    </div>
  );
}
