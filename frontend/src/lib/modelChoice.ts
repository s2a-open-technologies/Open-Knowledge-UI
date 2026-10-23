// Modellwahl im Chat: die gespeicherte Auswahl des Nutzers gegen das Standardmodell des Admins.

export const MODEL_KEY = "rag.model";
const DEFAULT_SEEN_KEY = "rag.model.default";
export const MODELS_CHANGED = "rag:models-changed";

/** Meldet der geöffneten Chat-Seite, dass Modelle oder Standardmodell geändert wurden. */
export function notifyModelsChanged(): void {
  window.dispatchEvent(new Event(MODELS_CHANGED));
}

/**
 * Ändert der Admin das Standardmodell, gilt es auch für Nutzer mit gespeicherter
 * Auswahl: die Auswahl wird verworfen. Danach wählt der Nutzer wieder frei.
 * Beim allerersten Besuch (nichts gemerkt) bleibt die Auswahl bestehen.
 */
export function reconcileChosenModel(
  chosen: string,
  currentDefault: string,
  lastSeenDefault: string,
): { chosen: string; seenDefault: string } {
  if (!currentDefault) return { chosen, seenDefault: lastSeenDefault };
  const changed = lastSeenDefault !== "" && lastSeenDefault !== currentDefault;
  return { chosen: changed ? "" : chosen, seenDefault: currentDefault };
}

export function readChosenModel(): string {
  try {
    return localStorage.getItem(MODEL_KEY) ?? "";
  } catch {
    return "";
  }
}

export function readSeenDefault(): string {
  try {
    return localStorage.getItem(DEFAULT_SEEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function writeSeenDefault(ref: string): void {
  try {
    localStorage.setItem(DEFAULT_SEEN_KEY, ref);
  } catch {
    // localStorage nicht verfügbar: gilt nur für diese Sitzung.
  }
}

export function writeChosenModel(ref: string): void {
  try {
    if (ref) localStorage.setItem(MODEL_KEY, ref);
    else localStorage.removeItem(MODEL_KEY);
  } catch {
    // localStorage nicht verfügbar: Auswahl gilt nur für diese Sitzung.
  }
}
