import { useState } from "react";
import { FileText, FolderOpen, Loader2, MessageSquare, NotebookText, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { MarkdownDialog } from "@/components/MarkdownDialog";
import { api, errorMessage } from "@/lib/api";
import { useApiQuery } from "@/lib/useApiQuery";
import { formatBytes, formatDateTime } from "@/lib/format";
import type { MyUpload } from "@/lib/types";

export function MyUploadsPage() {
  const [view, setView] = useState<MyUpload | null>(null);
  const [remove, setRemove] = useState<MyUpload | null>(null);

  const { data: items, loading, reload: load } = useApiQuery<MyUpload[]>(
    ["my-uploads"],
    () => api.get("/api/my-uploads"),
    [],
  );

  const confirmRemove = async () => {
    if (!remove) return;
    try {
      await api.del(`/api/my-uploads/${remove.kind}/${encodeURIComponent(remove.id)}`);
      toast.success(`${remove.filename} gelöscht`);
      setRemove(null);
      await load();
    } catch (e) {
      toast.error(errorMessage(e));
    }
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="w-full space-y-4 p-6">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-semibold">
            <FolderOpen className="size-6" />
            Meine Uploads
          </h1>
          <p className="text-sm text-muted-foreground">
            Deine Chat-Anhänge als Markdown. Beim Löschen verschwindet auch ihr Inhalt aus der
            Suche. Dokumente in einem Knowledge findest du dort in der Dokumentliste.
          </p>
        </div>

        {loading ? (
          <p className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Lade…
          </p>
        ) : items.length === 0 ? (
          <Card>
            <CardContent className="p-6 text-sm text-muted-foreground">
              Noch keine Uploads. Dateien, die du im Chat anhängst, erscheinen hier.
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardContent className="p-0">
              {items.map((item) => (
                <div
                  key={`${item.kind}-${item.id}`}
                  className="flex flex-wrap items-center gap-3 border-b border-border-base px-4 py-3 transition-colors last:border-0 hover:bg-hover/40"
                >
                  <FileText className="size-5 shrink-0 text-muted-foreground" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium">{item.filename}</p>
                    <p className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                      <Badge variant="secondary" className="gap-1">
                        {item.kind === "chat" ? (
                          <MessageSquare className="size-3" />
                        ) : (
                          <NotebookText className="size-3" />
                        )}
                        {item.kind === "chat" ? "Chat" : "Knowledge"}: {item.source}
                      </Badge>
                      {item.created_at ? formatDateTime(item.created_at) : null}
                      {item.size != null ? ` · ${formatBytes(item.size)}` : null}
                    </p>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!item.has_markdown}
                      onClick={() => setView(item)}
                    >
                      Ansehen
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      title="Löschen"
                      onClick={() => setRemove(item)}
                    >
                      <Trash2 className="size-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        )}
      </div>

      <MarkdownDialog
        source={
          view
            ? {
                title: view.filename,
                url: `/api/my-uploads/${view.kind}/${encodeURIComponent(view.id)}/markdown`,
              }
            : null
        }
        onOpenChange={(o) => !o && setView(null)}
      />
      <ConfirmDialog
        open={remove !== null}
        onOpenChange={(o) => !o && setRemove(null)}
        title="Upload löschen?"
        description={
          remove
            ? `${remove.filename} wird entfernt – samt allen zugehörigen Einträgen in der Vektordatenbank. Das lässt sich nicht rückgängig machen.`
            : undefined
        }
        confirmLabel="Löschen"
        destructive
        onConfirm={() => void confirmRemove()}
      />
    </div>
  );
}
