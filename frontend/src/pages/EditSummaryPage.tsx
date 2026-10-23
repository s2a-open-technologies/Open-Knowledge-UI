import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Download, Save } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Markdown } from "@/components/Markdown";
import { api, errorMessage } from "@/lib/api";
import type { SummaryContent } from "@/lib/types";

export function EditSummaryPage() {
  const { filename = "" } = useParams();
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const encoded = encodeURIComponent(filename);

  useEffect(() => {
    api
      .get<SummaryContent>(`/api/summaries/${encoded}`)
      .then((res) => setContent(res.content))
      .catch((e) => setError(errorMessage(e)))
      .finally(() => setLoading(false));
  }, [encoded]);

  const save = async () => {
    setSaving(true);
    try {
      await api.put(`/api/summaries/${encoded}`, { content });
      toast.success("Ablage gespeichert");
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6 p-6">
      <div className="flex flex-col items-start justify-between gap-4 border-b border-border-base pb-4 sm:flex-row sm:items-center">
        <div>
          <Link
            to="/summaries"
            className="mb-2 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="size-4" />
            Zurück zu Meine Ablagen
          </Link>
          <h1 className="text-xl font-semibold">{filename}</h1>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" size="sm" asChild>
            <a href={`/api/summaries/${encoded}/download-md`}>
              <Download className="size-4" />
              MD Download
            </a>
          </Button>
          <Button
            size="sm"
            className="bg-emerald-600 text-white hover:bg-emerald-700"
            asChild
          >
            <a href={`/api/summaries/${encoded}/download-docx`}>
              <Download className="size-4" />
              DOCX Download
            </a>
          </Button>
          <Button size="sm" onClick={save} disabled={saving}>
            <Save className="size-4" />
            {saving ? "Speichern…" : "Speichern"}
          </Button>
        </div>
      </div>

      {error ? (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            {error}
          </CardContent>
        </Card>
      ) : loading ? (
        <p className="text-muted-foreground">Lade…</p>
      ) : (
        <Tabs defaultValue="edit">
          <TabsList>
            <TabsTrigger value="edit">Bearbeiten</TabsTrigger>
            <TabsTrigger value="preview">Vorschau</TabsTrigger>
          </TabsList>
          <TabsContent value="edit">
            <Textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              className="min-h-[580px] font-mono text-sm"
            />
          </TabsContent>
          <TabsContent value="preview">
            <div className="min-h-[580px] rounded-lg border border-border-base bg-surface p-6">
              <Markdown content={content} />
            </div>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}
