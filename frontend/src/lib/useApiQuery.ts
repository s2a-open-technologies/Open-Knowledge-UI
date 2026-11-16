import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { toast } from "sonner";
import { errorMessage } from "@/lib/api";

/**
 * Lädt Daten über die API: ``data`` (bis zur Antwort ``initial``), ``loading`` nur beim ersten
 * Laden, ``reload()`` holt neu und wartet darauf, ``setData`` ändert den Zwischenspeicher sofort
 * (für optimistische Anzeige). Fehler erscheinen als Meldung.
 *
 * Der Zwischenspeicher wird beim An- und Abmelden geleert (siehe AuthContext).
 */
export function useApiQuery<T>(key: readonly unknown[], fetcher: () => Promise<T>, initial: T) {
  const queryClient = useQueryClient();
  const query = useQuery({ queryKey: key, queryFn: fetcher });
  useEffect(() => {
    if (query.error) toast.error(errorMessage(query.error));
  }, [query.error]);
  return {
    data: query.data ?? initial,
    loading: query.isPending,
    reload: () => queryClient.invalidateQueries({ queryKey: key }),
    setData: (update: (current: T) => T) =>
      queryClient.setQueryData<T>(key, (current) => update(current ?? initial)),
  };
}
