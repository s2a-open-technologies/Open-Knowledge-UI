# Basis-Images auf Manifest-Digest gepinnt (multi-arch, amd64+arm64); Dependabot aktualisiert den Digest zusammen mit dem Tag.
FROM python:3.14-slim-bookworm@sha256:48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv

# Ein Image fuer beide Rollen (TOOL_WORKER_ROLE=pandas|sql). MSSQL ueber pymssql (FreeTDS im Wheel):
# kein Microsoft-ODBC-Treiber noetig.
RUN pip install "pandas>=2.2" "numpy>=1.26" "openpyxl>=3.1.5" "charset-normalizer>=3.3" "flask>=3.0" "gunicorn>=22" \
    "sqlalchemy>=2.0" "psycopg[binary]>=3.2" "pymysql>=1.1" "pymssql>=2.3" "sqlglot>=25.0"

COPY tool_worker /srv/tool_worker
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin worker
USER 10001

EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090/health')"
CMD ["gunicorn", "-b", "0.0.0.0:8090", "-w", "2", "--threads", "4", "--timeout", "180", "tool_worker.wsgi:app"]
