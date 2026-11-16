import { useEffect, useMemo, useState } from "react";
import { Pencil, Plus, Trash2, Users as UsersIcon } from "lucide-react";
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
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { useAuth } from "@/context/AuthContext";
import { api, errorMessage } from "@/lib/api";
import { useApiQuery } from "@/lib/useApiQuery";
import type { AuthSource, Group, ManagedUser, Role } from "@/lib/types";

const SOURCE_LABEL: Record<AuthSource, string> = { local: "lokal", ldap: "LDAP", oidc: "OIDC" };

const ROLE_LABEL: Record<Role, string> = {
  admin: "Admin",
  editor: "Editor",
  user: "Nutzer",
};

const ROLE_HINT: Record<Role, string> = {
  admin: "Alles, inkl. Einstellungen, Verbindungen, Modelle, Benutzer",
  editor: "Knowledge, Dokumente und Vektordatenbank verwalten",
  user: "Chatten und freigegebenes Wissen nutzen",
};

interface UserForm {
  username: string;
  display_name: string;
  password: string;
  role: Role;
  is_active: boolean;
}

function UserDialog({
  open,
  user,
  isSelf,
  onOpenChange,
  onSaved,
}: {
  open: boolean;
  user: ManagedUser | null;
  isSelf: boolean;
  onOpenChange: (open: boolean) => void;
  onSaved: () => void;
}) {
  const [form, setForm] = useState<UserForm>({
    username: "",
    display_name: "",
    password: "",
    role: "user",
    is_active: true,
  });
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setForm({
      username: user?.username ?? "",
      display_name: user?.display_name ?? "",
      password: "",
      role: user?.role ?? "user",
      is_active: user?.is_active ?? true,
    });
  }, [open, user]);

  const set = <K extends keyof UserForm>(k: K, v: UserForm[K]) =>
    setForm((f) => ({ ...f, [k]: v }));

  const submit = async () => {
    setBusy(true);
    try {
      if (user) {
        await api.patch(`/api/users/${user.id}`, {
          display_name: form.display_name,
          // Eigene Rolle/Status sind serverseitig gesperrt und werden nicht gesendet.
          ...(isSelf ? {} : { role: form.role, is_active: form.is_active }),
          ...(form.password ? { password: form.password } : {}),
        });
      } else {
        await api.post("/api/users", form);
      }
      toast.success(user ? "Benutzer gespeichert" : "Benutzer angelegt");
      onSaved();
      onOpenChange(false);
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const valid = (user || form.username.trim().length >= 3) && (user || form.password.length >= 4);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{user ? "Benutzer bearbeiten" : "Neuer Benutzer"}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="u-name">Benutzername</Label>
            <Input
              id="u-name"
              value={form.username}
              disabled={user !== null}
              onChange={(e) => set("username", e.target.value)}
              autoComplete="off"
              autoFocus={!user}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="u-display">Anzeigename</Label>
            <Input
              id="u-display"
              value={form.display_name}
              onChange={(e) => set("display_name", e.target.value)}
              placeholder={form.username}
            />
          </div>
          {user && user.auth_source && user.auth_source !== "local" ? (
            <p className="rounded-md bg-surface-muted p-3 text-xs text-muted-foreground">
              Dieses Konto meldet sich über {SOURCE_LABEL[user.auth_source]} an. Ein Passwort
              lässt sich hier nicht setzen.
            </p>
          ) : (
            <div className="space-y-2">
              <Label htmlFor="u-pw">
                {user ? "Neues Passwort (leer = unverändert)" : "Passwort"}
              </Label>
              <Input
                id="u-pw"
                type="password"
                autoComplete="new-password"
                value={form.password}
                onChange={(e) => set("password", e.target.value)}
              />
            </div>
          )}
          <div className="space-y-2">
            <Label>Rolle</Label>
            <Select
              value={form.role}
              onValueChange={(v) => set("role", v as Role)}
              disabled={isSelf}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(Object.keys(ROLE_LABEL) as Role[]).map((r) => (
                  <SelectItem key={r} value={r}>
                    {ROLE_LABEL[r]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">{ROLE_HINT[form.role]}</p>
          </div>
          <div className="flex items-center justify-between rounded-lg border border-border-base px-3 py-2">
            <Label htmlFor="u-active" className="cursor-pointer">
              Aktiv
            </Label>
            <Switch
              id="u-active"
              checked={form.is_active}
              disabled={isSelf}
              onCheckedChange={(v) => set("is_active", v)}
            />
          </div>
          {isSelf ? (
            <p className="text-xs text-muted-foreground">
              Eigene Rolle und eigener Status sind gesperrt, damit du dich nicht aussperrst.
            </p>
          ) : null}
        </div>
        <DialogFooter>
          <Button variant="secondary" onClick={() => onOpenChange(false)}>
            Abbrechen
          </Button>
          <Button onClick={submit} disabled={busy || !valid}>
            {user ? "Speichern" : "Anlegen"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function GroupDialog({
  open,
  group,
  users,
  onOpenChange,
  onSaved,
}: {
  open: boolean;
  group: Group | null;
  users: ManagedUser[];
  onOpenChange: (open: boolean) => void;
  onSaved: () => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [members, setMembers] = useState<number[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setName(group?.name ?? "");
    setDescription(group?.description ?? "");
    setMembers(group?.member_ids ?? []);
  }, [open, group]);

  const submit = async () => {
    setBusy(true);
    try {
      const saved = group
        ? await api.patch<Group>(`/api/groups/${group.id}`, { name, description })
        : await api.post<Group>("/api/groups", { name, description });
      await api.put(`/api/groups/${saved.id}/members`, { user_ids: members });
      toast.success(group ? "Gruppe gespeichert" : "Gruppe angelegt");
      onSaved();
      onOpenChange(false);
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{group ? "Gruppe bearbeiten" : "Neue Gruppe"}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="g-name">Name</Label>
            <Input
              id="g-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoFocus
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="g-desc">Beschreibung</Label>
            <Textarea
              id="g-desc"
              rows={2}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label>Mitglieder ({members.length})</Label>
            <div className="max-h-52 space-y-1 overflow-y-auto rounded-lg border border-border-base p-2">
              {users.map((u) => (
                <label
                  key={u.id}
                  className="flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-hover"
                >
                  <input
                    type="checkbox"
                    className="size-4"
                    checked={members.includes(u.id)}
                    onChange={(e) =>
                      setMembers((prev) =>
                        e.target.checked ? [...prev, u.id] : prev.filter((x) => x !== u.id),
                      )
                    }
                  />
                  <span className="truncate">
                    {u.display_name} <span className="text-muted-foreground">({u.username})</span>
                  </span>
                </label>
              ))}
            </div>
          </div>
        </div>
        <DialogFooter>
          <Button variant="secondary" onClick={() => onOpenChange(false)}>
            Abbrechen
          </Button>
          <Button onClick={submit} disabled={busy || !name.trim()}>
            {group ? "Speichern" : "Anlegen"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function AdminUsersPage() {
  const { user: me } = useAuth();
  const [userDialog, setUserDialog] = useState<{ open: boolean; user: ManagedUser | null }>({
    open: false,
    user: null,
  });
  const [groupDialog, setGroupDialog] = useState<{ open: boolean; group: Group | null }>({
    open: false,
    group: null,
  });
  const [deleteUser, setDeleteUser] = useState<ManagedUser | null>(null);
  const [deleteGroup, setDeleteGroup] = useState<Group | null>(null);

  const { data, loading, reload: load } = useApiQuery(
    ["users-and-groups"],
    async () => {
      const [users, groups] = await Promise.all([
        api.get<ManagedUser[]>("/api/users"),
        api.get<Group[]>("/api/groups"),
      ]);
      return { users, groups };
    },
    { users: [] as ManagedUser[], groups: [] as Group[] },
  );
  const { users, groups } = data;

  const groupNames = useMemo(
    () => new Map(groups.map((g) => [g.id, g.name])),
    [groups],
  );
  const userNames = useMemo(
    () => new Map(users.map((u) => [u.id, u.display_name])),
    [users],
  );

  const removeUser = async () => {
    if (!deleteUser) return;
    try {
      await api.del(`/api/users/${deleteUser.id}`);
      toast.success("Benutzer gelöscht");
      await load();
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setDeleteUser(null);
    }
  };

  const removeGroup = async () => {
    if (!deleteGroup) return;
    try {
      await api.del(`/api/groups/${deleteGroup.id}`);
      toast.success("Gruppe gelöscht");
      await load();
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setDeleteGroup(null);
    }
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl space-y-6 p-6">
        <div className="border-b border-border-base pb-4">
          <h1 className="text-2xl font-semibold">Benutzer &amp; Gruppen</h1>
          <p className="text-sm text-muted-foreground">
            Konten, Rollen und Gruppen für Freigaben von Knowledge und Modellen.
          </p>
        </div>

        {loading ? (
          <p className="text-muted-foreground">Lade…</p>
        ) : (
          <Tabs defaultValue="users">
            <TabsList>
              <TabsTrigger value="users">Benutzer ({users.length})</TabsTrigger>
              <TabsTrigger value="groups">Gruppen ({groups.length})</TabsTrigger>
            </TabsList>

            <TabsContent value="users" className="space-y-4 pt-4">
              <div className="flex justify-end">
                <Button onClick={() => setUserDialog({ open: true, user: null })}>
                  <Plus className="size-4" />
                  Neuer Benutzer
                </Button>
              </div>
              <Card>
                <CardContent className="p-0">
                  {users.map((u) => (
                    <div
                      key={u.id}
                      className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-border-base px-3 py-1.5 transition-colors last:border-0 hover:bg-hover/40"
                    >
                      <p className="min-w-0 max-w-full shrink truncate text-sm font-medium sm:max-w-[16rem]">
                        {u.display_name}{" "}
                        <span className="font-normal text-muted-foreground">({u.username})</span>
                        {u.id === me?.id ? (
                          <span className="ml-2 text-xs text-muted-foreground">du</span>
                        ) : null}
                      </p>
                      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1">
                        {u.auth_source && u.auth_source !== "local" ? (
                          <Badge variant="outline">{SOURCE_LABEL[u.auth_source]}</Badge>
                        ) : null}
                        <Badge variant={u.role === "admin" ? "default" : "secondary"}>
                          {ROLE_LABEL[u.role]}
                        </Badge>
                        {!u.is_active ? <Badge variant="destructive">gesperrt</Badge> : null}
                        {u.group_ids.slice(0, 3).map((id) => (
                          <Badge key={id} variant="outline">
                            {groupNames.get(id) ?? "?"}
                          </Badge>
                        ))}
                        {u.group_ids.length > 3 ? (
                          <Badge
                            variant="outline"
                            title={u.group_ids
                              .slice(3)
                              .map((id) => groupNames.get(id) ?? "?")
                              .join(", ")}
                          >
                            +{u.group_ids.length - 3}
                          </Badge>
                        ) : null}
                      </div>
                      <div className="flex shrink-0 items-center">
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          title="Bearbeiten"
                          onClick={() => setUserDialog({ open: true, user: u })}
                        >
                          <Pencil className="size-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          title="Löschen"
                          disabled={u.id === me?.id}
                          onClick={() => setDeleteUser(u)}
                        >
                          <Trash2 className="size-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="groups" className="space-y-4 pt-4">
              <div className="flex justify-end">
                <Button onClick={() => setGroupDialog({ open: true, group: null })}>
                  <Plus className="size-4" />
                  Neue Gruppe
                </Button>
              </div>
              {groups.length === 0 ? (
                <p className="text-sm text-muted-foreground">Noch keine Gruppen.</p>
              ) : (
                <Card>
                  <CardContent className="p-0">
                    {groups.map((g) => (
                      <div
                        key={g.id}
                        className="flex flex-wrap items-center gap-3 border-b border-border-base px-3 py-1.5 transition-colors last:border-0 hover:bg-hover/40"
                      >
                        <UsersIcon className="size-4 shrink-0 text-muted-foreground" />
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm font-medium">{g.name}</p>
                          <p className="truncate text-xs text-muted-foreground">
                            {g.member_ids.length === 0
                              ? "Keine Mitglieder"
                              : g.member_ids.map((id) => userNames.get(id) ?? "?").join(", ")}
                          </p>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          title="Bearbeiten"
                          onClick={() => setGroupDialog({ open: true, group: g })}
                        >
                          <Pencil className="size-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          title="Löschen"
                          onClick={() => setDeleteGroup(g)}
                        >
                          <Trash2 className="size-4" />
                        </Button>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}
            </TabsContent>
          </Tabs>
        )}
      </div>

      <UserDialog
        open={userDialog.open}
        user={userDialog.user}
        isSelf={userDialog.user?.id === me?.id}
        onOpenChange={(o) => setUserDialog((d) => ({ ...d, open: o }))}
        onSaved={() => void load()}
      />
      <GroupDialog
        open={groupDialog.open}
        group={groupDialog.group}
        users={users.filter((u) => u.is_active)}
        onOpenChange={(o) => setGroupDialog((d) => ({ ...d, open: o }))}
        onSaved={() => void load()}
      />
      <ConfirmDialog
        open={deleteUser !== null}
        onOpenChange={(o) => !o && setDeleteUser(null)}
        title="Benutzer löschen?"
        description={
          deleteUser
            ? `„${deleteUser.display_name}“ wird samt aller Chats und Freigaben endgültig gelöscht. Zum Sperren stattdessen „Aktiv“ ausschalten.`
            : undefined
        }
        confirmLabel="Löschen"
        destructive
        onConfirm={removeUser}
      />
      <ConfirmDialog
        open={deleteGroup !== null}
        onOpenChange={(o) => !o && setDeleteGroup(null)}
        title="Gruppe löschen?"
        description={
          deleteGroup
            ? `„${deleteGroup.name}“ wird gelöscht. Freigaben an diese Gruppe entfallen.`
            : undefined
        }
        confirmLabel="Löschen"
        destructive
        onConfirm={removeGroup}
      />
    </div>
  );
}
