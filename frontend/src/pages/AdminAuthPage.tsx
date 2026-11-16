import { useCallback, useEffect, useState } from "react";
import { Copy, KeyRound, Loader2, Network, Play } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Field } from "@/components/ui/field";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { api, errorMessage } from "@/lib/api";
import type { Role } from "@/lib/types";

interface LdapCfg {
  enabled: boolean;
  url: string;
  starttls: boolean;
  verify_tls: boolean;
  ca_file: string;
  bind_dn: string;
  bind_password: string;
  has_bind_password?: boolean;
  base_dn: string;
  user_filter: string;
  username_attr: string;
  name_attr: string;
  group_attr: string;
  timeout: number;
}

interface OidcCfg {
  enabled: boolean;
  name: string;
  issuer: string;
  client_id: string;
  client_secret: string;
  has_client_secret?: boolean;
  scopes: string;
  username_claim: string;
  name_claim: string;
  groups_claim: string;
  timeout: number;
}

interface AuthOverview {
  ldap: LdapCfg;
  oidc: OidcCfg;
  default_role: Role;
  auto_create: boolean;
  base_url: string;
  redirect_uri: string;
}

type Kind = "ldap" | "oidc";

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 text-sm">
      <Switch checked={checked} onCheckedChange={onChange} />
      {label}
    </label>
  );
}

export function AdminAuthPage() {
  const [data, setData] = useState<AuthOverview | null>(null);
  const [ldap, setLdap] = useState<LdapCfg | null>(null);
  const [oidc, setOidc] = useState<OidcCfg | null>(null);
  const [general, setGeneral] = useState({ default_role: "user" as Role, auto_create: true, base_url: "" });
  const [probe, setProbe] = useState({ user: "", password: "" });
  const [testing, setTesting] = useState<Kind | null>(null);
  const [saving, setSaving] = useState(false);

  const apply = (d: AuthOverview) => {
    setData(d);
    setLdap({ ...d.ldap, bind_password: "" });
    setOidc({ ...d.oidc, client_secret: "" });
    setGeneral({ default_role: d.default_role, auto_create: d.auto_create, base_url: d.base_url });
  };

  const load = useCallback(async () => {
    try {
      apply(await api.get<AuthOverview>("/api/admin/auth"));
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
      apply(await api.put<AuthOverview>("/api/admin/auth", { ldap, oidc, ...general }));
      toast.success("Gespeichert");
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setSaving(false);
    }
  };

  const test = async (kind: Kind) => {
    setTesting(kind);
    try {
      const r = await api.post<Record<string, unknown>>("/api/admin/auth/test", {
        kind,
        values: kind === "ldap" ? ldap : oidc,
        ...(kind === "ldap" ? { test_user: probe.user, test_password: probe.password } : {}),
      });
      if (kind === "ldap") {
        const groups = Array.isArray(r.groups) ? (r.groups as string[]).join(", ") : "";
        const parts = [
          r.service_bind ? "Verbindung ok" : "",
          probe.user ? (r.user_found ? "Benutzer gefunden" : "Benutzer nicht gefunden") : "",
          probe.user && probe.password ? (r.login_ok ? "Anmeldung ok" : "Anmeldung abgelehnt") : "",
          groups ? `Gruppen: ${groups}` : "",
        ].filter(Boolean);
        toast.success(parts.join(" · ") || "Test erfolgreich");
      } else {
        toast.success("Anbieter erreichbar" + (r.pkce_supported ? " · PKCE unterstützt" : ""));
      }
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setTesting(null);
    }
  };

  if (!data || !ldap || !oidc) return <div className="p-6 text-sm text-muted-foreground">Lädt …</div>;

  const setL = <K extends keyof LdapCfg>(k: K, v: LdapCfg[K]) => setLdap({ ...ldap, [k]: v });
  const setO = <K extends keyof OidcCfg>(k: K, v: OidcCfg[K]) => setOidc({ ...oidc, [k]: v });

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-6 p-6">
        <div>
          <h1 className="text-xl font-semibold">Authentifizierung</h1>
          <p className="text-sm text-muted-foreground">
            Anmeldung über LDAP/Active Directory oder OIDC. Die lokale Anmeldung bleibt immer
            erhalten. Gruppen werden über ihren Namen den lokalen Gruppen zugeordnet. Fehlende
            Gruppen werden nicht angelegt, und Rollen kommen nie aus Gruppen.
          </p>
        </div>

        <Card>
          <CardContent className="grid gap-3 p-4 sm:grid-cols-2">
            <Field label="Rolle neuer externer Konten" hint="Befördern kannst du später unter Benutzer & Gruppen.">
              <Select value={general.default_role} onValueChange={(v) => setGeneral({ ...general, default_role: v as Role })}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="user">Benutzer</SelectItem>
                  <SelectItem value="editor">Editor</SelectItem>
                  <SelectItem value="admin">Admin</SelectItem>
                </SelectContent>
              </Select>
            </Field>
            <Field
              label="Öffentliche Adresse der App"
              hint="Für den OIDC-Rückruf, z. B. https://rag.firma.de (hinter einem Proxy nötig). Leer = aus der Anfrage."
            >
              <Input value={general.base_url} onChange={(e) => setGeneral({ ...general, base_url: e.target.value })} placeholder="https://rag.firma.de" />
            </Field>
            <Toggle label="Konten beim ersten Login automatisch anlegen" checked={general.auto_create} onChange={(v) => setGeneral({ ...general, auto_create: v })} />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-3 p-4">
            <div className="flex items-start justify-between gap-2">
              <div>
                <h2 className="flex items-center gap-2 font-semibold">
                  <Network className="size-4" />
                  LDAP / Active Directory
                </h2>
                <p className="text-sm text-muted-foreground">Anmeldung über das normale Formular mit Benutzername und Passwort.</p>
              </div>
              <Toggle label="Aktiv" checked={ldap.enabled} onChange={(v) => setL("enabled", v)} />
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Server-Adresse" hint="ldap://host:389 oder ldaps://host:636">
                <Input value={ldap.url} onChange={(e) => setL("url", e.target.value)} placeholder="ldaps://ad.firma.de" />
              </Field>
              <Field label="Basis-DN">
                <Input value={ldap.base_dn} onChange={(e) => setL("base_dn", e.target.value)} placeholder="dc=firma,dc=de" />
              </Field>
              <Field label="Service-Konto (Bind-DN)" hint="Zum Suchen des Benutzers. Leer = anonyme Suche.">
                <Input value={ldap.bind_dn} onChange={(e) => setL("bind_dn", e.target.value)} />
              </Field>
              <Field label="Service-Passwort" hint={ldap.has_bind_password ? "Gespeichert. Leer lassen = behalten." : undefined}>
                <Input type="password" autoComplete="new-password" value={ldap.bind_password} onChange={(e) => setL("bind_password", e.target.value)} placeholder={ldap.has_bind_password ? "••••••••" : ""} />
              </Field>
              <Field label="Benutzerfilter" hint="{username} wird maskiert eingesetzt.">
                <Input value={ldap.user_filter} onChange={(e) => setL("user_filter", e.target.value)} />
              </Field>
              <Field label="Gruppen-Attribut" hint="Meist memberOf. Aus Gruppen-DNs wird der Name (CN) genommen.">
                <Input value={ldap.group_attr} onChange={(e) => setL("group_attr", e.target.value)} />
              </Field>
              <Field label="Attribut Benutzername">
                <Input value={ldap.username_attr} onChange={(e) => setL("username_attr", e.target.value)} />
              </Field>
              <Field label="Attribut Anzeigename">
                <Input value={ldap.name_attr} onChange={(e) => setL("name_attr", e.target.value)} />
              </Field>
              <Field label="Eigene CA-Datei (optional)" hint="Pfad auf dem Server. Zertifikate werden immer geprüft.">
                <Input value={ldap.ca_file} onChange={(e) => setL("ca_file", e.target.value)} />
              </Field>
              <Field label="Zeitlimit (Sekunden)">
                <Input type="number" min={1} max={120} value={ldap.timeout} onChange={(e) => setL("timeout", Number(e.target.value) || 10)} />
              </Field>
              <Toggle label="StartTLS verwenden" checked={ldap.starttls} onChange={(v) => setL("starttls", v)} />
              <Toggle label="Zertifikat prüfen" checked={ldap.verify_tls} onChange={(v) => setL("verify_tls", v)} />
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Probe-Benutzer (optional)">
                <Input value={probe.user} onChange={(e) => setProbe({ ...probe, user: e.target.value })} autoComplete="off" />
              </Field>
              <Field label="Probe-Passwort (optional)" hint="Wird nur für den Test genutzt und nicht gespeichert.">
                <Input type="password" value={probe.password} onChange={(e) => setProbe({ ...probe, password: e.target.value })} autoComplete="new-password" />
              </Field>
            </div>
            <Button variant="outline" size="sm" onClick={() => test("ldap")} disabled={testing !== null}>
              {testing === "ldap" ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
              Testen
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-3 p-4">
            <div className="flex items-start justify-between gap-2">
              <div>
                <h2 className="flex items-center gap-2 font-semibold">
                  <KeyRound className="size-4" />
                  OIDC (Keycloak, Entra ID, Okta …)
                </h2>
                <p className="text-sm text-muted-foreground">Der Login-Bildschirm zeigt einen zusätzlichen Knopf. Verfahren: Authorization-Code mit PKCE.</p>
              </div>
              <Toggle label="Aktiv" checked={oidc.enabled} onChange={(v) => setO("enabled", v)} />
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Name auf dem Knopf">
                <Input value={oidc.name} onChange={(e) => setO("name", e.target.value)} />
              </Field>
              <Field label="Issuer-Adresse" hint="Discovery unter <Issuer>/.well-known/openid-configuration">
                <Input value={oidc.issuer} onChange={(e) => setO("issuer", e.target.value)} placeholder="https://sso.firma.de/realms/firma" />
              </Field>
              <Field label="Client-ID">
                <Input value={oidc.client_id} onChange={(e) => setO("client_id", e.target.value)} />
              </Field>
              <Field label="Client-Secret" hint={oidc.has_client_secret ? "Gespeichert. Leer lassen = behalten." : undefined}>
                <Input type="password" autoComplete="new-password" value={oidc.client_secret} onChange={(e) => setO("client_secret", e.target.value)} placeholder={oidc.has_client_secret ? "••••••••" : ""} />
              </Field>
              <Field label="Scopes">
                <Input value={oidc.scopes} onChange={(e) => setO("scopes", e.target.value)} />
              </Field>
              <Field label="Claim Benutzername">
                <Input value={oidc.username_claim} onChange={(e) => setO("username_claim", e.target.value)} />
              </Field>
              <Field label="Claim Anzeigename">
                <Input value={oidc.name_claim} onChange={(e) => setO("name_claim", e.target.value)} />
              </Field>
              <Field label="Claim Gruppen" hint="Der Anbieter muss ihn im ID-Token mitschicken.">
                <Input value={oidc.groups_claim} onChange={(e) => setO("groups_claim", e.target.value)} />
              </Field>
            </div>
            <div className="flex flex-wrap items-center gap-2 rounded-md bg-surface-muted p-3 text-xs">
              <Badge variant="secondary">Rückruf-Adresse</Badge>
              <code className="break-all">{data.redirect_uri}</code>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  void navigator.clipboard?.writeText(data.redirect_uri);
                  toast.success("Kopiert");
                }}
              >
                <Copy className="size-3.5" />
              </Button>
              <span className="text-muted-foreground">Diese Adresse beim Anbieter als Redirect-URI eintragen.</span>
            </div>
            <Button variant="outline" size="sm" onClick={() => test("oidc")} disabled={testing !== null}>
              {testing === "oidc" ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
              Testen
            </Button>
          </CardContent>
        </Card>

        <div>
          <Button onClick={save} disabled={saving}>
            {saving ? "Speichert …" : "Speichern"}
          </Button>
        </div>
      </div>
    </div>
  );
}
