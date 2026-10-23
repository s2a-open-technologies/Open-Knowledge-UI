# Audit-Checkliste: Open Knowledge UI (Selbstauskunft)

> Jede Zeile mit Fundstelle. Status: ✅ vorhanden · ⚠️ teilweise · ❌ fehlt. Stand: gegen den Code verifiziert am 2026-10-15 mit Unreleased-Stand (Tags bis v0.2.5). Vor Wiederverwendung gegen den Code prüfen. Orientierung, keine Zertifizierung, keine Rechtsberatung.

## Security

| Prüfpunkt | Status | Fundstelle/Beleg | Kommentar |
|---|---|---|---|
| Passwort-Hashing | ✅ | `app/auth/users.py` (`generate_password_hash`/`check_password_hash`, Werkzeug) | Konstante Laufzeit bei unbekanntem Konto über `_DUMMY_HASH` (`app/auth/users.py:54`) |
| Sitzung / Secret Key | ⚠️ | `app/__init__.py:306-312`, Cookie-Flags `app/core/csrf.py:58-62` | Schlüssel wird zufällig erzeugt und in der DB gespeichert (`SECRET_KEY` sensibel), keine Rotation. Cookie-Flags jetzt explizit: `HttpOnly`, `SameSite=Lax`; `Secure` über `SESSION_COOKIE_SECURE` (opt-in, für HTTPS-Bereitstellung) |
| Zugangsdaten verschlüsselt at rest | ✅ | `app/core/crypto.py` (`EncryptedText`, `encrypt_existing`, Format `enc:v1:<Fernet-Token>`) | API-Keys der Verbindungen, LDAP/OIDC-Secrets, Quellen-Keys und sensible Settings beim Start einmalig verschlüsselt (idempotent); nie ausgeliefert (nur `has_key`); kaputter Schlüssel bricht den Start ab (`SecretKeyError`) statt leeren Zugangsdaten |
| Key-Rotation | ✅ | `app/core/crypto.py` (`ENCRYPTION_KEY` Komma-Liste: erster verschlüsselt, alle entschlüsseln) | Rotation = alte Keys anhängen und mit dem neuen neu verschlüsseln; Standortverschlüsselung der Schlüsseldatei 0600 |
| Secrets nie im Repo | ✅ | `env.example`, `.gitignore`; Standardpasswort nur als Startwert | `admin`/`admin` beim ersten Start muss geändert werden (`README.md`, `SECURITY.md`) |
| Rollen / RBAC | ✅ | `app/auth/auth.py` (`require_role`), `app/auth/access.py` (Gruppen, Grants) | Rollen user/editor/admin, Knowledge und Modelle pro Gruppe freigebbar |
| Rollen aus Fremdgruppen | ✅ | `app/auth/auth_external.py:sync_groups` | Externe Gruppen vergeben nie Rollen und legen keine Gruppen an |
| Injection-Schutz (LDAP/SQL) | ✅ | `app/auth/auth_ldap.py` (`escape_filter_chars`), SQLAlchemy ORM; Test `tests/test_external_auth.py` | `run_pandas` mit Blockliste (`SECURITY.md`) |
| OIDC-Prüfung | ✅ | `app/auth/auth_oidc.py` (PKCE, `state`, `nonce`, Signatur über JWKS, `iss`, `aud`, `exp`) | Konto-Verknüpfung nur über `sub`, nie über den Namen (Test `test_oidc_never_links_an_existing_account_by_name_only`) |
| CSRF | ✅ | `app/core/csrf.py` (Session-Token, Cookie-Spiegel, `X-CSRF-Token` per `hmac.compare_digest`), Test `tests/test_csrf.py` | Alle POST/PUT/PATCH/DELETE; Rotation bei Login/Logout; `SameSite=Lax` zusätzlich |
| CORS | ✅ | kein CORS-Header gesetzt (gleiche Herkunft) | |
| Rate-Limiting | ⚠️ | `app/auth/users.py:40-50,114-127` (5 Fehlversuche je Konto, 20 je IP, 300 s) | Zähler **im Prozessspeicher** (für den einzelnen gunicorn-Worker beabsichtigt), weg nach Neustart. Nur Login |
| MFA | ❌ | - | Nur über OIDC-Anbieter (dessen MFA) möglich |
| Security-Header | ✅ | `app/core/security_headers.py` (in `create_app` registriert), Test `tests/test_security_headers.py` | CSP (Skripte/Verbindungen gleiche Herkunft; `style/script` mit `unsafe-inline` wegen Theme-Bootstrap), `X-Frame-Options: DENY` + `frame-ancestors`, `nosniff`, `Referrer-Policy`; HSTS nur bei `SESSION_COOKIE_SECURE`. Mit `SECURITY_HEADERS=off` abschaltbar (Proxy setzt selbst) |
| Sicherheitsrelevante Tests | ✅ | `tests/test_users.py`, `tests/test_access.py`, `tests/test_external_auth.py`, `tests/test_external_sources.py` | SSRF-/Link-Local-Sperre in `app/llm/registry.py:_reject_link_local` |
| Disclosure-Prozess | ✅ | `SECURITY.md` | Antwort innerhalb 72 h zugesagt |
| Container nicht als root | ✅ | `Dockerfile` (`APP_UID=1000`, gid 0), `entrypoint.sh` (gosu-Abgabe, chown der Volumes), Tool-Worker uid 10001 | App-Bild non-root; bestehende root-Volumes werden beim Start übernommen (Gruppe 0); Parakeet-Image als GPU-Dienst wie gehabt |
| DB/Qdrant nicht öffentlich | ⚠️ | Qdrant ohne `ports`-Block nur im Compose-Netz (`docker-compose.yml:199`), Postgres-Port auskommentiert (`:186-187`) | Exponiert am Host: App 8083 (`:12`), SearXNG 8081 (`:133`), EntityGuard/PII-Filter 8082 (`:147`, unauthentifiziert — betriebsabhängig); `QDRANT_API_KEY` optional |
| Endpunkte mit Geheimnissen nur für Admins | ✅ | `/api/admin/*` mit `@require_role("admin")`; Geheimnisse nur als `has_*` | |
| Protokollierung sicherheitsrelevanter Ereignisse | ✅ | `audit_log`-Tabelle (Migration `0036`), `app/auth/audit.py` (best-effort), Admin-API `GET /api/admin/audit` + Reiter *Administration › Protokoll* | Anmeldung ok/fehler/throttled, externe Anmeldungen, Abmeldung, Passwortwechsel, Kontoverwaltung, Gruppen, Freigaben, Datenexport; keine Rotation/Aufbewahrung |

## Datenschutz / DSGVO

| Prüfpunkt | Status | Fundstelle/Beleg | Kommentar |
|---|---|---|---|
| Datenflussdiagramm / VVT-Start | ✅ | `docs/data-mapping.md` | |
| Art.-32-Maßnahmen | ⚠️ | `SECURITY.md`, diese Liste | Verschlüsselung at rest fehlt (App-Ebene) |
| Pseudonymisierung | ⚠️ | `app/utils/pii_filter_client.py`, `docs/sanitizer-api.md` | Herstellerneutrale Schnittstelle. Nutzereingaben werden vor dem LLM maskiert, wenn ein Filter eingerichtet ist; bei Filterfehler bricht der Chat mit Fehlermeldung ab (fail-closed), der ganze Kontext inkl. Verlauf und Compaction wird maskiert, die Antwort live zurückübersetzt (`app/chat/pii_session.py`). Lücke: Dokument-Treffer gehen unmaskiert ans Modell. Websuche bricht bei Filterfehler ab und läuft ohne Filter nur nach ausdrücklicher Freigabe |
| Datenminimierung | ⚠️ | Chat-Anhänge werden mit dem Chat gelöscht (`app/chat/attachments.py:cleanup_session_attachments`) | Keine automatische Aufbewahrungsfrist für Chats |
| Berichtigung | ✅ | Anzeigename/Passwort (`app/web/user_routes.py`, `app/auth/user_admin.py:119`) | |
| Löschung | ✅ | Chat (`app/web/routes.py:210`), eigene Dokumente (`app/web/user_routes.py:127`), Benutzer samt Chats (`app/auth/user_admin.py:161`) | Benutzer löschen entfernt Chats, Mitgliedschaften, Freigaben; Uploads/Ablagen des Benutzers nicht ausdrücklich (siehe Data Mapping, Lücke 3). Backups nicht betroffen |
| Auskunft / Übertragbarkeit | ✅ | `GET /api/me/export` (`app/web/api_routes.py`), Logik `app/chat/data_export.py` (`tests/test_data_export.py`), UI *Mein Konto › Daten exportieren* | JSON mit Profil, eigenen Chats/Ablagen/Dokument- und Anhang-Referenzen; fremde Inhalte nie enthalten |
| Widerspruch | ⚠️ | Betreiberprozess | |
| TLS zu externen Endpunkten | ⚠️ | LDAP prüft Zertifikate (`app/auth/auth_ldap.py`); LLM-/Quellen-URLs vom Admin wählbar | `http://` ist erlaubt, Betreiber verantwortlich |
| Auftragsverarbeiter dokumentiert | ⚠️ | `docs/data-mapping.md` ("Externe Stellen") | Hängt von der Konfiguration ab |
| Aufbewahrungsfristen | ❌ | keine Retention-Logik im Code (Code-Sicht 2026-10-15) | Keine automatische Löschung; Betreiber legt fest |
| Bekannte Lücken benannt | ✅ | `docs/data-mapping.md` | |

## Lizenz / Compliance

| Prüfpunkt | Status | Fundstelle/Beleg | Kommentar |
|---|---|---|---|
| `LICENSE` vollständig | ✅ | `LICENSE` (AGPL-3.0) | |
| Copyright konsistent | ✅ | Dateiköpfe "SPDX-FileCopyrightText: 2026 S2A Open Technologies gGmbH" (+ Copyright-Zeile), generiert per Skript | Rechteinhaber GmbH, Urheber Christopher Abanilla (`AUTHORS.md`) |
| SPDX-Header | ✅ | 105 Quelldateien mit `SPDX-License-Identifier: AGPL-3.0-or-later` + `SPDX-FileCopyrightText`; Rest per `REUSE.toml` (Override) | Fonts (`OFL-1.1` unter `LICENSES/`) sind ausdrücklich als Fremdcode annotiert |
| `AUTHORS` / `THIRD_PARTY_NOTICES` | ✅ | `AUTHORS.md` (Rechteinhaber/Urheber), `THIRD_PARTY_NOTICES.md` (generiert über `scripts/third_party.py`: 197 Python- + 261 npm-Pakete, Stand 2026-10-15) | Generierte Snapshots; nach Dependency-Updates neu erzeugen |
| Contributor CLA | ✅ | [`CLA.md`](../CLA.md) (Teil A Einzelpersonen, Teil B Unternehmen), Verweise in `CONTRIBUTING.md`, `AUTHORS.md`, `README.md` und PR-Template | Übertragung der übertragbaren Verwertungsrechte; Urheberrecht verbleibt beim Contributor (§ 29 UrhG) |
| Lockfiles | ✅ | `uv.lock`, `frontend/package-lock.json` | CI nutzt `uv sync --frozen` |
| Automatische Lizenzprüfung / `reuse lint` | ✅ | `REUSE.toml`, `LICENSES/OFL-1.1.txt`, `uvx reuse lint` läuft durch (`reuse` nicht als Projekt-Dep; ad hoc via uvx) | CI-Integration optional nachholen |
| SBOM pro Image | ✅ | `.github/workflows/images.yml` (`sbom`-Job, `anchore/sbom-action`, CycloneDX je Image als Artefakt) | Läuft erst mit dem nächsten Image-Build (GitHub-Runner) |

## Projekt-Health / Governance

| Prüfpunkt | Status | Fundstelle/Beleg | Kommentar |
|---|---|---|---|
| CI/CD | ✅ | `.github/workflows/ci.yml`, `release.yml`, `images.yml` (Multi-Arch-Images, Push by Digest) | Actions auf SHAs gepinnt |
| Dependabot | ⚠️ | `.github/dependabot.yml` | GitHub-Actions, npm (frontend) und Docker (3 Dockerfiles); `pip`/`uv` bewusst nicht (uv.lock gepinnt, Aktualisierung manuell über `uv`) |
| Lint in CI | ✅ | `ci.yml`: hartes Ruff-Gate (`--select E4,E7,E9/F`-Basisregeln) + Statistik-Schritt ohne Exit-Code für Altlasten | Weite Regelsätze (BLE/UP/DTZ …) bewusst informativ; Basis-Regelwerk grün gehalten |
| CONTRIBUTING / CODE_OF_CONDUCT / SECURITY | ✅ | `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` | |
| CHANGELOG | ✅ | `CHANGELOG.md` (Keep a Changelog) | |
| Issue-/PR-Vorlagen | ✅ | `.github/ISSUE_TEMPLATE/`, `.github/pull_request_template.md` | |
| Versionierung und Tags | ✅ | `scripts/release.py`, Tags v0.1.0 bis v0.2.5 | Container-Images via Sigstore keyless signiert (`images.yml`, `cosign sign`, GitHub-OIDC, verifizierbar per `cosign verify`) |
| Testabdeckung | ⚠️ | Backend: 964 Tests (`pytest tests`: 958 bestanden + 6 übersprungen, Lauf 2026-10-15, SQLite; Postgres-Varianten per `TEST_DATABASE_URL`); Frontend: 64 vitest-Tests (node + jsdom + Testing-Library-Komponententests: Markdown-Sanitizer-Regression, CodeBlock-Falten/Kopieren, TableBlock-CSV, SSE-Parser, Message-Block-Extraktion, API-Fehlerbehandlung, Theme/Draft), Typprüfung + Build im CI | Frontend: keine E2E-/Browser-Interaktionstests |
| Bus-Faktor | ⚠️ | ein Maintainer | Mitigiert: Codex-/Agentenwissen in `AGENTS.md`/`CLAUDE.md`, automatisierte Releases (`scripts/release.py` + Workflows), dokumentierte Runbooks |
| Health-Endpunkt | ✅ | `app/web/routes.py:226` (`/health`) | |
| Reproduzierbare Builds | ✅ | Lockfiles (`uv.lock`, `package-lock.json`), CI nutzt `--frozen`; alle Basis-Images auf Manifest-Digest gepinnt (multi-arch; Dependabot zieht Digest + Tag) | Compose-Runtime-Images (qdrant, postgres, searxng) bleiben Tags — Betreiber-Konfiguration |

## Zusammenfassung

**Stärken:** saubere Rollen- und Gruppenprüfung, Geheimnisse werden nie ausgeliefert und liegen seit 0.2.5 verschlüsselt at rest (Fernet, Rotation über `ENCRYPTION_KEY`-Liste, Start bricht bei kaputtem Schlüssel ab), Login-Härtung inkl. OIDC/LDAP mit Tests, CSRF-Token und explizite Cookie-Flags, SSRF-Sperre, Disclosure-Prozess, CI mit gepinnten Actions, Löschwege für Chats, Dokumente und Benutzer.

**Lücken (Reihenfolge nach Wirkung):**
1. Keine **Aufbewahrungsregeln** (Audit-Log wächst ohne Rotation/Anonymisierung; Chats ohne automatische Löschung) — Betreiber legt Fristen fest.
2. Keine lokale **MFA** — nur über den OIDC-Anbieter.
3. Login-Sperre nur **im Prozessspeicher** (für `-w 1` beabsichtigt, weg bei Neustart).
4. Release-Supply-Chain: Releases **ungesigniert**, Basis-Images ohne **Digest-Pin**, Python-Abhängigkeiten ohne automatisches Bot-Update (uv manuell).
5. PII-Lücke: Dokument-Treffer (Werkzeug-Ergebnisse) gehen **unmaskiert** ans Modell (`docs/data-mapping.md`, Lücke 4).
6. CSP toleriert Inline-Skripte/Stile (Theme-Bootstrap) — kein Schutz gegen Inline-Injektion, nur gegen fremde Skriptquellen.
