# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Open Knowledge UI is a German-language RAG app (Flask backend, React SPA, Qdrant, Postgres). [`AGENTS.md`](./AGENTS.md) is the detailed source of truth (architecture facts, gotchas, per-feature sections); this file keeps the commands, the big picture and the rules you need first.

## Commands

```bash
# tests (the `worker` extra is required: sqlglot, pymysql, pymssql). Full suite takes ~7 min, run it in the background with a log.
uv run --extra worker --with pytest pytest tests -q -p no:cacheprovider
uv run --extra worker --with pytest pytest tests/test_sql_values.py::test_full_flow_suggest_save_build_and_search -q   # one test
SQL_TEST_PG=host:port:db:user:pw SQL_TEST_MYSQL=... SQL_TEST_MSSQL=... # also run the real-server tests in tests/test_tool_worker_sqlengine.py
TEST_DATABASE_URL=postgresql+psycopg://... # run the suite against Postgres instead of SQLite

uvx ruff check --select F app tests tool_worker alembic    # no [tool.ruff] config; CI only prints statistics
cd frontend && npx tsc -b --noEmit && npm run build         # builds into app/static (not committed)
alembic upgrade head                                        # latest revision: see alembic/versions (0035 at time of writing)

docker compose up -d --build        # needs TOOL_WORKER_TOKEN in .env; -f docker-compose.cpu.yml for no-GPU/ARM
python run_mac.py                   # macOS: app native, helpers + workers in Docker
```

## Big picture

- **Chat request** (`app/api/routes.py`, `POST /chat`, SSE): resolves the model (`app/llm/registry.py`), builds a per-request `RetrievalScope` (ContextVar in `app/rag/retrieval.py`: document collections, foreign Qdrants, `sql_sources`), picks tools with `select_tools` (`app/agent/document_agent.py`), masks the whole context through the PII filter (`app/chat/pii_session.py`, fail-closed) and streams the LangChain agent. Tools read the scope, never the request.
- **Knowledge has three kinds** (`knowledge.kind`): `local` (own Qdrant collection, ingestion pipeline in `app/ingest/`), `external` (foreign Qdrant, read-only), `sql` (`app/knowledge/sql/`: live read-only SQL, value directory for spellings). `app/knowledge/service.py:is_remote` marks the read-only kinds; access is grants + `is_public` in `app/auth/access.py` (editors may write everything shared with their groups).
- **Code and database access never runs in the app.** `tool_worker/` is a separate Flask service (same image, `TOOL_WORKER_ROLE=pandas|sql`): sandboxed Pandas and read-only SQL (`sqlcheck.py` AST validation + read-only sessions). The app only talks to it via `app/agent/toolworker/client.py`; there is no in-app fallback.
- **Embedding/rerank backends** are a registry (`app/rag/models_runtime.py`); remote ones go through `app/rag/endpoints.py` per connection kind (`llm` OpenAI-style, `tei`, `infinity`).
- **Settings live in the DB** (`settings` table, seeded by Alembic migrations and `SETTINGS` lists such as `app/agent/toolworker/config.py`), not in env vars; a DB value silently beats `.env`. Infra keys (`DATA_DIR`, `QDRANT_*`) are env-only.

## Rules

- Never commit, tag or push without an explicit yes for that action. No `Co-Authored-By` trailer in commits. Force-pushes are blocked by the sandbox: prepare a script and let the user run it with `!`.
- Add every user-visible change to `CHANGELOG.md` under `[Unreleased]` (German); the release is cut with `scripts/release.py`.
- The project is called **Open Knowledge UI** (slug `open-knowledge-ui`, GitHub `s2a-open-technologies/Open-Knowledge-UI`); the old names are gone everywhere, including the default Qdrant collection `open_knowledge_ui`.
- UI text, error messages and docs are German. Name no function in `app/` starting with `test_`. Migrations must be Postgres-safe and idempotent.
- Measure before choosing a retrieval or matching method (see `app/knowledge/sql/values.py`, `app/rag/` profiles); do not swap defaults without re-measuring.
