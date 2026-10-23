import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Download, FileText, Pencil, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { api, errorMessage } from "@/lib/api";
import { formatDate } from "@/lib/format";
import type { SummaryFile } from "@/lib/types";

export function SummariesPage() {
  const [summaries, setSummaries] = useState<SummaryFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleteTarget, setDeleteTarget] = useState<SummaryFile | null>(null);

  const load = () =>
    api
      .get<SummaryFile[]>("/api/summaries")
      .then(setSummaries)
      .catch((e) => toast.error(errorMessage(e)))
      .finally(() => setLoading(false));

  useEffect(() => {
    void load();
  }, []);

  const confirmDelete = async () => {
    if (!deleteTarget) return;
    try {
      await api.del(`/api/summaries/${encodeURIComponent(deleteTarget.filename)}`);
      toast.success("Ablage gelöscht");
      setSummaries((s) => s.filter((x) => x.filename !== deleteTarget.filename));
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setDeleteTarget(null);
    }
  };

  return (
    <div className="space-y-6 p-6">
      <div className="border-b border-border-base pb-4">
        <h1 className="mb-1 text-2xl font-semibold">Meine Ablagen</h1>
        <p className="text-muted-foreground">
          Gespeicherte Markdown-Dateien aus deinen Chats
        </p>
      </div>

      {loading ? (
        <p className="text-muted-foreground">Lade…</p>
      ) : summaries.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <FileText className="mx-auto mb-4 size-16 text-muted-foreground" />
            <h2 className="mb-2 text-lg font-semibold">Noch keine Ablagen</h2>
            <p className="text-muted-foreground">
              Eine Ablage entsteht, wenn du im Chat darum bittest, etwas abzulegen,
              z. B. eine Zusammenfassung des Gesprächs.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {summaries.map((s) => (
            <Card key={s.filename} className="p-5">
              <div className="mb-4 flex items-start gap-3">
                <div className="flex size-12 shrink-0 items-center justify-center rounded-xl bg-surface-muted">
                  <FileText className="size-6 text-muted-foreground" />
                </div>
                <div className="min-w-0 flex-1">
                  <h3 className="truncate font-semibold">{s.filename}</h3>
                  <p className="text-sm text-muted-foreground">
                    {s.size_human} • {formatDate(s.created_at)}
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button variant="secondary" size="sm" asChild>
                  <Link
                    to={`/summaries/${encodeURIComponent(s.filename)}`}
                  >
                    <Pencil className="size-4" />
                    Bearbeiten
                  </Link>
                </Button>
                <Button variant="secondary" size="sm" asChild>
                  <a
                    href={`/api/summaries/${encodeURIComponent(s.filename)}/download-md`}
                  >
                    <Download className="size-4" />
                    MD
                  </a>
                </Button>
                <Button
                  size="sm"
                  className="bg-emerald-600 text-white hover:bg-emerald-700"
                  asChild
                >
                  <a
                    href={`/api/summaries/${encodeURIComponent(s.filename)}/download-docx`}
                  >
                    <Download className="size-4" />
                    DOCX
                  </a>
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => setDeleteTarget(s)}
                >
                  <Trash2 className="size-4" />
                  Löschen
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}

      <ConfirmDialog
        open={deleteTarget !== null}
        onOpenChange={(o) => !o && setDeleteTarget(null)}
        title="Ablage löschen?"
        description={deleteTarget?.filename}
        confirmLabel="Löschen"
        destructive
        onConfirm={confirmDelete}
      />
    </div>
  );
}
