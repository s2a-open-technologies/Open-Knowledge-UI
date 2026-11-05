# Basis-Images auf Manifest-Digest gepinnt (multi-arch, amd64+arm64); Dependabot aktualisiert den Digest zusammen mit dem Tag.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv

# Ein Image fuer beide Rollen (TOOL_WORKER_ROLE=pandas|sql). MSSQL ueber pymssql (FreeTDS im Wheel):
# kein Microsoft-ODBC-Treiber noetig.
RUN pip install "pandas>=2.2" "numpy>=1.26" "openpyxl>=3.1.5" "charset-normalizer>=3.3" "fastapi>=0.162" "uvicorn>=0.54" "pydantic>=2.12" \
    "sqlalchemy>=2.0" "psycopg[binary]>=3.2" "pymysql>=1.1" "pymssql>=2.3" "sqlglot>=25.0"

COPY tool_worker /srv/tool_worker
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin worker
USER 10001

EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090/health')"
CMD ["uvicorn", "tool_worker.wsgi:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "2", "--timeout-keep-alive", "180"]
