# Contributing to Open Knowledge UI

Vielen Dank für dein Interesse, zu diesem Projekt beizutragen!

## Entwicklungsumgebung einrichten

```bash
git clone https://github.com/s2a-open-technologies/Open-Knowledge-UI.git
cd open-knowledge-ui
uv sync
cp env.example .env
alembic upgrade head
python run_linux.py   # oder run_windows.py
```

## Code Style

- **Linting/Formatting:** [Ruff](https://docs.astral.sh/ruff/) (keine eigene Konfiguration, Defaults)
  ```bash
  uv run ruff check .
  uv run ruff format .
  ```
- **Imports:** stdlib → third party → local, getrennt durch Leerzeilen.
- **Sprache:** UI, Tool-Beschreibungen, System-Prompt und user-facing Errors sind auf Deutsch.
- **Kommentare:** Englisch oder Deutsch, aber sparsam — nur wo nötig.

## Commits & Pull Requests

1. Erstelle einen Branch: `feat/...`, `fix/...`, `docs/...`
2. Schreibe klare Commit-Messages nach [Conventional Commits](https://www.conventionalcommits.org/de/): `feat: …`, `fix(chat): …`, `docs: …`, `refactor: …`, `test: …`, `chore: …` (`feat!:` bzw. `BREAKING CHANGE:` für inkompatible Änderungen). Daraus entstehen die Changelog-Einträge.
3. Beschreibe im PR **was** geändert wurde und **warum**.
4. Bei Schema-Änderungen: erstelle eine Alembic-Migration:
   ```bash
   alembic revision --autogenerate -m "description"
   ```
5. Führe `uv run ruff check .` aus, bevor du den PR öffnest, sowie `uv run --with pytest pytest tests`.
6. Trage nutzerrelevante Änderungen unter `## [Unreleased]` in [`CHANGELOG.md`](./CHANGELOG.md) ein.

## Releases

Versionen folgen [Semantic Versioning](https://semver.org/lang/de/). Einzige Quelle ist `pyproject.toml`;
`scripts/release.py` zieht `frontend/package.json`, `frontend/package-lock.json` und `uv.lock` mit.

```bash
python3 scripts/release.py current                 # aktuelle Version
python3 scripts/release.py bump minor --dry-run    # Vorschau (patch | minor | major | X.Y.Z)
python3 scripts/release.py bump minor              # Version, CHANGELOG, Commit, Tag v0.2.0 (lokal)
python3 scripts/release.py publish --yes           # Branch + Tag pushen
```

- `bump` verlangt einen sauberen Arbeitsbaum auf `master` (Override: `--branch`) und verschiebt `[Unreleased]` im Changelog
  nach `[X.Y.Z]`; ist der Abschnitt leer, wird er aus den Commits seit dem letzten Tag erzeugt.
- Der Tag-Push startet die Action [`release.yml`](./.github/workflows/release.yml): Sie prüft, dass Tag, Version und
  Changelog-Abschnitt zusammenpassen, führt Tests und Frontend-Build aus und erstellt das GitHub-Release mit den Notizen
  aus dem Changelog. `ci.yml` testet jeden Push auf `master` und jeden Pull Request.
- Das erste Release `0.1.0` entsteht mit `bump 0.1.0` (gleiche Version, nur Changelog, Commit und Tag).

## Sicherheitsrelevante Änderungen

- Die `run_pandas`-Tool-Guardrails dürfen **nicht** gelockert werden (blockt `os.`, `sys.`, `subprocess`, `__import__`, `open(`, `eval(`, `exec(`, write-Methoden). Nur mit expliziter Zustimmung.
- Keine Secrets, API-Keys oder privaten IPs in Code committen.
- Default-Passwörter nur als Fallback, nie hartkodiert für Production.

## Lizenz / CLA

Bevor ein Beitrag übernommen wird, muss die [Contributor License Agreement (CLA)](./CLA.md) angenommen werden. Sie überträgt der S2A Open Technologies gGmbH alle übertragbaren ausschließlichen Nutzungs- und Verwertungsrechte am Beitrag; das Urheberrecht selbst bleibt nach § 29 UrhG beim Contributor. Die Beiträge stehen der Allgemeinheit weiterhin unter der [AGPL-3.0-or-later](./LICENSE) zur Verfügung.

- **Einzelpersonen:** die CLA (Teil A) unterschreiben und einreichen oder ihr im Pull Request ausdrücklich zustimmen („Ich habe die CLA gelesen und stimme ihr zu.“).
- **Unternehmen / Beschäftigte:** die Corporate CLA (Teil B) unterschreiben und die beitragenden Personen im Anhang benennen.
- Ohne angenommene CLA kann ein Beitrag nicht übernommen werden. Bereits erteilte Lizenzen (AGPL-3.0-or-later) bleiben unberührt.
