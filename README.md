# Open Knowledge UI

Ein Open-Source-RAG für eigene Dokumente, gebaut auf **Qualität** und volle Kontrolle über Einlesen und Suche. Die Oberfläche ist ein Chat im Stil von ChatGPT bzw. Langdock: Fragen stellen, Antworten mit Quellen bekommen, Dateien direkt anhängen. Alles läuft lokal, Modelle und Daten bleiben bei dir.

> **Lizenz:** GNU Affero General Public License v3 oder später (AGPL-3.0-or-later), siehe [`LICENSE`](./LICENSE).

## Was es kann

- **Chat mit Quellen** (SSE-Streaming, Spracheingabe per Parakeet oder Whisper). Dateien lassen sich anhängen und sind nur in diesem Chat durchsuchbar.
- **Knowledge-Bereiche**: pro Bereich eine eigene Qdrant-Collection mit Upload, Dokumentliste, Vektoransicht und eigenem Retrieval-Profil.
- **Markdown-first-Einlesen**: PDF, DOCX, DOC, ODT und MD werden mit Docling zu Markdown (Überschriften, Tabellen, Bildbeschreibungen per Vision-LLM), gespeichert und an den Überschriften zerlegt. Das gespeicherte Markdown ist in der Oberfläche einsehbar und die Grundlage zum Neu-Indexieren.
- **Tabellen als CSV**: Tabellen aus Dokumenten und CSV/Excel-Dateien werden nicht in den Vektorindex gepresst, sondern vom Modell per Pandas ausgewertet (richtige Summen statt geratener Zahlen).
- **Hybride Suche**: Bedeutung (Dense, HNSW) plus Stichwort (BM25 mit IDF), Neusortierung mit einem Cross-Encoder und Nachbar-Chunks für vollständige Absätze. Das Standardprofil ist die empfohlene Strategie, Alternativen gibt es unter „Erweitert“.
- **Einstellbare Modelle**: Embedding und Reranking wählst du unter *Administration › Einlesen & Modelle*, entweder als HuggingFace-Modell (lokal, z. B. `BAAI/bge-m3`) oder über einen Endpoint: OpenAI-kompatibel, **Text Embeddings Inference** oder **Infinity**. „Verbindung testen“ prüft die Embedding- und Rerank-API wirklich. Die Vektor-Dimension ergibt sich aus dem Modell.
- **Neu indexieren ohne Ausfall**: Nach einem Modellwechsel baut ein Hintergrundjob den Index aus dem Markdown neu auf und schaltet erst um, wenn er fertig ist.
- **Mehrbenutzer**: Rollen (`admin`, `editor`, `user`), Gruppen und Freigaben für Knowledge und Modelle. Jeder legt eigenes Knowledge an und verwaltet es; Editoren bearbeiten zusätzlich alles, was einer ihrer Gruppen freigegeben ist. Anmeldung lokal, per LDAP oder OIDC. Chats sind pro Nutzer privat.
- **Beliebige LLMs**: OpenAI-kompatible Verbindungen (LiteLLM, vLLM, Ollama, OpenAI) mit Modellliste, eigenem Prompt und Parametern pro Modell. Das Knowledge eines Modells ist die Basis jeder Suche; Nutzer können im Chat eigenes Knowledge dazuwählen (pro Modell abschaltbar).
- **Externe Quellen**: Ein fremder Qdrant lässt sich als Knowledge anbinden (nur lesen, live abgefragt, mit dem Embedding-Modell der Quelle) und wie jedes Knowledge freigeben und an Modelle hängen.
- **SQL-Datenbanken als Knowledge**: PostgreSQL, MySQL/MariaDB, SQL Server und SQLite werden live und nur lesend abgefragt (nichts wird kopiert). Ein Wertverzeichnis hilft dem Modell, Schreibweisen wie „Stühle“ zu `Stuhl` und `Bürostuhl` zu finden. Details unten.
- **Datenauswertung in einem eigenen Container**: Pandas-Code auf CSV/Excel läuft im isolierten `tool-worker`, nicht in der App.
- **PII-Filter** (optional): Ist ein Sanitizer eingerichtet, geht der ganze Kontext maskiert an das Modell, fail-closed; die Antwort wird live zurückübersetzt.
- Optional: Websuche über SearXNG, die du pro Modell zuschaltest (Standard aus). Mit PII-Filter werden Suchanfragen vorher maskiert; ohne Filter nur nach ausdrücklicher Freigabe.

Details zu Änderungen: [`CHANGELOG.md`](./CHANGELOG.md). Architektur und Konventionen für Entwickler: [`AGENTS.md`](./AGENTS.md).

## Schnellstart

### Docker (empfohlen)

```bash
cp env.example .env
echo "TOOL_WORKER_TOKEN=$(openssl rand -hex 32)" >> .env   # Pflicht: Zugang zu den Workern
echo "ENCRYPTION_KEY=$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')" >> .env   # empfohlen: Schlüssel für Zugangsdaten in der Datenbank
docker compose up -d --build
```

Öffnet die App auf http://localhost:8083. Postgres, Qdrant, SearXNG, entityguard, Parakeet und die beiden Worker (`tool-worker` für Pandas, `tool-worker-sql` für SQL) starten als eigene Container. Das Standard-Setup nutzt eine NVIDIA-GPU; ohne GPU oder auf ARM siehe *Plattformen* unten.

### Datenbank (Postgres)

Die App speichert Benutzer, Chats, Einstellungen und Modelle in **Postgres**. `docker compose up` startet den Container `postgres` samt Volume `pgdata` und migriert beim Start automatisch. Zugangsdaten stehen in `.env` (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, siehe `env.example`); für den Betrieb ein eigenes Passwort setzen. Der Postgres-Port ist in `docker-compose.yml` auskommentiert (nur im Docker-Netz erreichbar); für Import, pgAdmin und Backups die Zeilen unter `ports:` einkommentieren (gebunden an `127.0.0.1`, `POSTGRES_PORT`, Standard 5432). Ohne `DATABASE_URL` nutzt die App eine lokale SQLite-Datei, das ist nur für Entwicklung und Tests gedacht.

**Daten einer älteren Version übernehmen** (frühere Versionen nutzten eine SQLite-Datei):

```bash
docker compose up -d postgres
DATABASE_URL=postgresql+psycopg://rag:rag@localhost:5432/rag alembic upgrade head
python scripts/migrate_sqlite_to_postgres.py --sqlite data/app.db --database-url postgresql+psycopg://rag:rag@localhost:5432/rag
```

Das Skript arbeitet an einer Kopie der SQLite-Datei (die Originaldatei bleibt unverändert), bringt sie auf den aktuellen Stand, leert die Zieltabellen, kopiert alles, setzt die ID-Zähler und vergleicht die Zeilenzahlen. Mit `--dry-run` zählt es nur. Backup: `docker compose exec postgres pg_dump -U rag rag > backup.sql`.

### Ohne Docker

Voraussetzungen: Python 3.12+, [uv](https://docs.astral.sh/uv/), Node.js 20+, ein laufender Qdrant-Server. Für PDF-OCR der Engine `tesseract` (Default, siehe `OCR_ENGINE`): Tesseract (`tesseract-ocr-deu`) und Poppler; `rapidocr` und `off` brauchen kein zusätzliches Binary.

```bash
uv sync
(cd frontend && npm install && npm run build)   # React-Oberfläche nach app/static/
cp env.example .env                             # Qdrant-Host, OCR-Engine, Worker-Token
alembic upgrade head                            # Datenbank anlegen bzw. aktualisieren
python run_linux.py                             # http://localhost:5000
```

**macOS (Apple Silicon):** Nur die App läuft nativ, alles andere bleibt in Docker, denn Docker auf dem Mac hat keinen Zugriff auf die GPU. `python run_mac.py` startet per `docker-compose.mac.yml` Postgres, Qdrant, SearXNG, entityguard, Parakeet (CPU) und die beiden Worker auf localhost, migriert die Datenbank und startet die App auf http://127.0.0.1:5000. Embedding, Reranking und Docling rechnen dann über Metal (MPS), ONNX-Modelle über CoreML; mit `RAG_DEVICE=cpu` lässt sich das abschalten. Voraussetzungen: Docker Desktop, `uv sync`, Frontend-Build und `brew install tesseract tesseract-lang poppler`. Mit `--no-docker` startet nur die App. Der Mac-Pfad ist bisher nicht auf echter Hardware getestet.

Unter Windows genügt ein Doppelklick auf `start.bat` (bzw. `python run_windows.py`). Das lädt bei Bedarf `qdrant.exe` herunter und startet es lokal, ohne Administratorrechte. Mit `--no-qdrant` wird ein vorhandener Qdrant genutzt.

**Sicherheit:** Zugangsdaten in der Datenbank (API-Keys, Quellen-Zugänge, sensible Einstellungen) sind mit Fernet verschlüsselt; der Schlüssel steht in `ENCRYPTION_KEY` (ohne ihn legt die App eine Schlüsseldatei im Datenverzeichnis an, siehe `env.example`), und er darf nicht verloren gehen. Änderungen über die Oberfläche sind gegen CSRF geschützt (Token im Header), der Sitzungs-Cookie ist `SameSite=Lax`; hinter HTTPS `SESSION_COOKIE_SECURE=true` setzen.

**Bilder im Chat:** Fotos, Screenshots und Scans lassen sich anhängen oder einfügen. Das Bildmodell liest sie einmal zu Text (durchsuchbar, PII-gefiltert, Karte „Bilder im Chat“ unter *Einlesen & Modelle*); Modelle mit dem Schalter „Versteht Bilder“ sehen das Bild zusätzlich selbst. Der PII-Filter kann Bilder nicht schwärzen.

**Ohne Docker** (Linux/Windows nativ) gibt es die Tool-Worker nicht: Pandas-Auswertungen und SQL-Knowledge melden dann „nicht verfügbar“, alles andere funktioniert. Wer sie braucht, startet die Worker per Docker (`docker compose up -d tool-worker tool-worker-sql`) und stellt die Adressen unter *Administration › Tools* ein.

### Anmeldung mit LDAP / OIDC (optional)

Unter *Administration › Authentifizierung* lässt sich die Anmeldung an ein Verzeichnis oder einen SSO-Anbieter anbinden. Die lokale Anmeldung (z. B. `admin`) bleibt immer erhalten.

- **LDAP / Active Directory**: Adresse (`ldaps://ad.firma.de`), Basis-DN, Service-Konto zum Suchen. Der Standardfilter passt für AD (`sAMAccountName`), für OpenLDAP z. B. `(&(objectClass=inetOrgPerson)(uid={username}))` mit Benutzernamen-Attribut `uid`. Mit dem Probe-Benutzer prüft „Testen“ die ganze Kette.
- **OIDC** (Keycloak, Entra ID, Okta …): Issuer-Adresse, Client-ID und -Secret eintragen und die angezeigte **Rückruf-Adresse** beim Anbieter als Redirect-URI hinterlegen (z. B. `https://rag.firma.de/api/auth/oidc/callback`). Läuft die App hinter einem Proxy, trage die öffentliche Adresse als „Öffentliche Adresse der App“ ein. Für Gruppen muss der Anbieter den Gruppen-Claim (`groups`) ins ID-Token legen (Keycloak: Mapper „Group Membership“, Entra: Gruppenansprüche).
- **Gruppen**: Es wird nichts importiert. Legst du unter *Benutzer & Gruppen* eine Gruppe an, die genauso heißt wie eine Gruppe im Verzeichnis (z. B. `opendox-admin`), kommt jeder Benutzer mit dieser Gruppe beim Login automatisch hinein. Rollen vergibst du weiter selbst; neue Konten erhalten die eingestellte Standardrolle.

### Erster Start

Beim ersten Start gibt es den Benutzer **`admin`** mit Passwort **`admin`** (bei frischer Datenbank über `ADMIN_PASSWORD` in `.env` änderbar). Bitte sofort das Passwort ändern (Kontomenü unten links).

Danach in dieser Reihenfolge:

1. **Administration › Verbindungen**: LLM-Endpunkt eintragen (Base-URL und Key). Für Embedding/Reranking über TEI oder Infinity den passenden Typ wählen und „Verbindung testen“ drücken.
2. **Administration › Modelle**: Standardmodell wählen.
3. **Administration › Einlesen & Modelle**: Embedding- und Reranking-Modell prüfen (der Standard funktioniert ohne Eingriff, beim ersten Start werden die Modelle heruntergeladen).
4. **Knowledge**: Dokumente hochladen und einlesen, dann im Chat fragen.

### SQL-Datenbanken als Knowledge

Admins binden PostgreSQL, MySQL/MariaDB, SQL Server oder SQLite an (*Knowledge › SQL-Datenbank*). Das Modell fragt Schema und Daten live und nur lesend ab, nichts wird kopiert; mit PII-Filter werden Ergebnisse maskiert. Dafür läuft der zusätzliche Dienst `tool-worker-sql`. Einrichtung und Sicherheit: [`docs/sql-knowledge.md`](./docs/sql-knowledge.md).

### Plattformen (amd64, arm64, Mac)

| Wo | Wie |
|---|---|
| Linux/Windows mit NVIDIA-GPU | `docker compose up -d --build` (CUDA, wie bisher) |
| Ohne GPU oder **ARM-Server** (arm64) | `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build` (CPU-Torch, keine GPU-Reservierung) |
| **Mac (Apple Silicon)** | `python run_mac.py`: die App läuft nativ (Metal), Hilfsdienste inkl. Tool-Worker in Docker; das Token für die Worker legt das Skript in der `.env` an |
| Fertige Images | Bei jedem Release gibt es Images in `ghcr.io/<owner>/<repo>`: `:X.Y.Z` und `:latest` (CUDA, amd64), `:X.Y.Z-cpu` und `:latest-cpu` (CPU, **amd64 und arm64**) sowie `…-tool-worker` (amd64 und arm64) |

Embedding und Reranking über **Text Embeddings Inference** laufen auch auf ARM; Infinity gibt es nur für amd64. Wie TEI/Infinity als Connection eingetragen werden, steht im Changelog (0.2.0).

### Tool-Worker (Pandas-Auswertungen)

Auswertungen von CSV/Excel (`run_pandas`) laufen nicht in der App, sondern im eigenen Container `tool-worker` (nicht als root, nur lesender Datenzugriff, kein Ausgang nach außen). Das Token (`TOOL_WORKER_TOKEN`) steht in der `.env`, siehe Schnellstart. Ohne laufenden Worker funktioniert der Chat weiter, nur die Pandas-Auswertung meldet „Codeausführung nicht verfügbar“. Details: [`docs/tool-worker-api.md`](./docs/tool-worker-api.md).

## Modelle und Hardware

- **Tabellen aus PDFs:** Nach der Layout-Erkennung liest das Bildmodell jede Tabelle noch einmal vom Bild und korrigiert falsch zugeordnete Zellen (Karte „Tabellen prüfen“ unter *Einlesen & Modelle*; ein Modellaufruf je Tabelle, abschaltbar).
- **Gerät:** Die App wählt automatisch CUDA (NVIDIA), MPS (Apple Silicon) oder CPU; `RAG_DEVICE` erzwingt eines davon. Die Wahl steht auf der Seite *Einlesen & Modelle*.
- **GPU (Docker):** Das Docker-Image bringt PyTorch mit CUDA mit; `docker-compose.yml` reserviert die GPU für die App (NVIDIA-Treiber und Container Toolkit nötig). Docling, Embedding und Reranker nutzen sie automatisch. Ohne GPU: `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build` (siehe *Plattformen*). Die lokale Entwicklung (`uv sync`) installiert bewusst das CPU-Wheel; wer lokal die GPU will, installiert CUDA-torch in die venv (`uv pip install --reinstall-package torch --reinstall-package torchvision torch torchvision`).

- Das Standard-Embedding ist `paraphrase-multilingual-MiniLM-L12-v2`, der Standard-Reranker `BAAI/bge-reranker-base` (ca. 1 GB). Beides läuft auf der CPU.
- Für höhere Qualität: `BAAI/bge-m3` (Embedding, ca. 2,2 GB) und `BAAI/bge-reranker-v2-m3`. Auf der CPU ist das Einlesen damit deutlich langsamer; mit GPU (siehe oben) läuft es direkt in der App (gemessen auf einer RTX 3060: ca. 4,5 GB GPU-Speicher für beide Modelle). Alternativ betreibst du die Modelle an einem Endpoint (vLLM, Text Embeddings Inference, Infinity) und trägst ihn als Verbindung mit dem passenden Typ ein.
- Ein anderes Embedding-Modell erfordert **Neu indexieren** (Knopf im Knowledge bzw. „Alle neu indexieren“ in der Admin-Seite). Bis dahin wird jedes Knowledge weiter mit seinem alten Modell durchsucht.
- Ein anderes Reranking-Modell wirkt sofort.

## Einlesen im Überblick

```
Datei ─► Docling ─► Markdown (gespeichert) ─► nach Überschriften zerlegen
      ─► [Kontext je Chunk per LLM] ─► Embedding (Dense + BM25) ─► Qdrant
```

- Das Markdown liegt unter `data/markdown/<knowledge_id>/`; das Original bleibt standardmäßig archiviert (abschaltbar).
- Chunk-Größe, Überlappung, Kontextanreicherung und das Inhaltsverzeichnis (`index.md` pro Knowledge) stellst du unter *Einlesen & Modelle* ein.
- Das Einlesen läuft Datei für Datei im Hintergrund, der Fortschritt steht in der Oberfläche.

## Entwicklung

```bash
uv run --extra worker --with pytest pytest tests -q   # Tests (inkl. Worker-Pakete)
uv run ruff check app tests                 # Lint
cd frontend && npm run dev                  # Vite mit HMR auf :5173, Proxy auf Flask :5000
cd frontend && npm run build                # Produktions-Build nach app/static/
```

- Backend: Flask, SQLAlchemy + Alembic (Postgres), Qdrant, LangChain. Die JSON-API liegt unter `/api/*`, der Chat unter `POST /chat` (SSE).
- Frontend: React 19, TypeScript, Vite, Tailwind v4, shadcn/ui.
- Migrationen: `alembic revision --autogenerate -m "..."`, `alembic upgrade head`. Eine bestehende Datenbank ohne Migrationsstand markierst du mit `alembic stamp head`.
- Releases: `scripts/release.py` (Version, Changelog, Tag). Beiträge siehe [`CONTRIBUTING.md`](./CONTRIBUTING.md), Sicherheitsmeldungen siehe [`SECURITY.md`](./SECURITY.md).

## Governance und Nachweise

Selbstauskunft mit Fundstellen und ehrlich benannten Lücken (keine Zertifizierung, keine Rechtsberatung):

- [`docs/audit-checklist-open-knowledge-ui.md`](./docs/audit-checklist-open-knowledge-ui.md): Security, Datenschutz, Lizenz, Projekt-Health
- [`docs/data-mapping.md`](./docs/data-mapping.md): Datenflüsse, Aufbewahrung, Betroffenenrechte (Start für VVT/DSFA)
- [`docs/rahmenwerke.md`](./docs/rahmenwerke.md): Einordnung für SOC 2, NIS2, ISO 27001, CRA, DSGVO
- [`docs/sanitizer-api.md`](./docs/sanitizer-api.md): Schnittstelle des PII-Filters (mit OpenAPI-Datei und Beispiel-Adapter)
- [`docs/tool-worker-api.md`](./docs/tool-worker-api.md): Tool-Worker (isolierte Codeausführung), Sicherheitsmodell und Grenzen
- [`docs/sql-knowledge.md`](./docs/sql-knowledge.md): SQL-Datenbanken als Knowledge (Einrichtung, Sicherheit, PII, Grenzen)
- [`SECURITY.md`](./SECURITY.md), [`CONTRIBUTING.md`](./CONTRIBUTING.md), [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md)

## Betrieb

- **Qdrant**: Server-Modus, `QDRANT_HOST`/`QDRANT_PORT` (Standard `localhost:6333`). Zum Umziehen auf einen anderen Server gibt es `scripts/manage_qdrant.py`.
- **Websuche**: `searxng/settings.yml` muss das Format `json` zulassen und `server.limiter: false` setzen. Den mitgelieferten `secret_key` für den Betrieb ersetzen (`openssl rand -hex 32`). SearXNG-URL und PII-Filter stellst du unter *Administration › Tools* ein; die Websuche selbst schaltest du je Modell unter *Modelle › Konfigurieren › Werkzeuge* zu. Sie lässt sich nur aktivieren, wenn der PII-Filter eingerichtet ist oder du sie ausdrücklich ohne Filter erlaubst (Suchanfragen gehen dann unmaskiert hinaus).
- **PII-Filter**: Jeder Dienst, der die [Sanitizer-API](./docs/sanitizer-api.md) spricht, funktioniert (entityguard ist die Referenz, ein Adapter für eigene Erkenner liegt in `docs/examples/`). `PII_FILTER_URL` leer oder `PII_FILTER_ENABLED=false` schaltet ihn ab; die Websuche läuft dann nur mit der Freigabe *Tools › Websuche › Auch ohne PII-Filter erlauben* (Suchanfragen gehen unmaskiert hinaus). Ist ein Filter eingerichtet und nicht erreichbar, wird nichts unmaskiert gesendet: der Chat zeigt eine Fehlermeldung, die Websuche bricht ab. Das Modell für Bildbeschreibungen wählst du unter *Einlesen & Modelle*.
- **Sprache**: Der Mic-Button kennt zwei Betriebsarten, die unter *Administration › Tools › Spracheingabe* eingestellt werden (Standard *Automatisch*): **Live-Streaming** (PCM per WebSocket `/ws/stt-stream` an Parakeet, der Text erscheint beim Sprechen; `STT_BASEURL` ist in Docker vorbelegt) und **Aufnehmen & senden** für jeden OpenAI-kompatiblen Dienst (`POST <Adresse>/audio/transcriptions`), z. B. Whisper über speaches/faster-whisper-server, vLLM, OpenAI oder Groq: Adresse, Modell (z. B. `whisper-large-v3`) und Key eintragen, der Text kommt nach dem Stoppen. *Automatisch* nimmt Streaming, wenn der Dienst `/audio/stream` anbietet, sonst Aufnehmen & senden. „Testen“ schickt eine Sekunde Stille und zeigt, was der Dienst kann. Das Mikrofon braucht HTTPS oder localhost.

### Fehlersuche

| Problem | Ursache und Lösung |
|---|---|
| Datenbank nicht erreichbar | `docker compose ps`: läuft `postgres` und ist er `healthy`? `DATABASE_URL` bzw. `POSTGRES_*` in `.env` prüfen. |
| Einstellungen weg nach Neustart | Das Postgres-Volume `pgdata` darf nicht gelöscht werden (`docker compose down -v` löscht es). |
| Qdrant nicht erreichbar | `docker compose ps` bzw. `QDRANT_HOST`/`QDRANT_PORT` prüfen. |
| `docker compose` bricht mit „TOOL_WORKER_TOKEN“ ab | Token in der `.env` setzen (Schnellstart); `run_mac.py` macht das selbst. |
| „Codeausführung nicht verfügbar“ / „SQL-Dienst nicht verfügbar“ | Läuft `tool-worker` bzw. `tool-worker-sql`? Status und Adresse stehen unter *Administration › Tools*; das Token muss in App und Worker gleich sein. |
| Alembic scheitert an vorhandenen Tabellen | `alembic stamp head`, wenn die Datenbank schon aktuell ist. |
| Modell lädt nicht | Status und Fehlertext stehen unter *Einlesen & Modelle*; ohne Internet können HuggingFace-Modelle nicht geladen werden. |
| Reranking fehlt in der Suche | Der Reranker lädt beim Start im Hintergrund; schlägt das fehl, läuft die Suche ohne Neusortierung weiter. |

## Autor und Rechteinhaber

Rechteinhaber ist die **S2A Open Technologies gGmbH**; Urheber ist Christopher Abanilla.
Siehe [`AUTHORS.md`](./AUTHORS.md). Beiträge setzen die Annahme der
[Contributor License Agreement (CLA)](./CLA.md) voraus.
