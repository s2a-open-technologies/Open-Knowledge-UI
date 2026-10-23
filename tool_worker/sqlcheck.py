"""SQL-Pruefung: nur lesende Einzelabfragen auf erlaubten Tabellen/Spalten (zweite Schicht neben read-only Sitzung)."""

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

DIALECTS = {
    "postgresql": "postgres",
    "mysql": "mysql",
    "mssql": "tsql",
    "sqlite": "sqlite",
}

DENY_FUNCS = {
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_stat_file",
    "lo_import",
    "lo_export",
    "lo_get",
    "lo_put",
    "pg_sleep",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "dblink",
    "dblink_exec",
    "set_config",
    "load_file",
    "sleep",
    "benchmark",
    "get_lock",
    "sys_exec",
    "sys_eval",
    "xp_cmdshell",
    "openrowset",
    "opendatasource",
    "openquery",
    "load_extension",
    "readfile",
    "writefile",
    "current_setting",
    "copy",
}
DENY_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Merge,
    exp.Command,
    exp.Into,
    exp.Lock,
    exp.Set,
    exp.Use,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Grant,
    exp.Pragma,
    exp.Copy,
    exp.Analyze,
    exp.TruncateTable,
)


class SqlRejected(ValueError):
    """Die Abfrage darf nicht ausgefuehrt werden (Text ist fuer das Modell gedacht)."""


def _norm(name: str) -> str:
    return (name or "").strip('"`[] ').lower()


def validate_select(
    sql: str,
    dialect: str,
    allowed_tables: set[str] | None = None,
    blocked_columns: dict[str, set[str]] | None = None,
    row_limit: int = 50,
) -> str:
    """Prueft ``sql`` und gibt die Abfrage mit gesetztem Zeilenlimit zurueck (sonst ``SqlRejected``)."""
    gdialect = DIALECTS.get(dialect)
    if gdialect is None:
        raise SqlRejected("Unbekannter Datenbanktyp")
    if not sql or not sql.strip():
        raise SqlRejected("Leere Abfrage")
    try:
        statements = sqlglot.parse(sql, read=gdialect)
    except (ParseError, TokenError) as e:
        raise SqlRejected(f"SQL nicht lesbar: {str(e)[:200]}") from None
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SqlRejected("Genau eine SELECT-Abfrage erlaubt")
    root = statements[0]
    if not isinstance(root, (exp.Select, exp.Union, exp.Subquery)):
        raise SqlRejected("Nur SELECT-Abfragen sind erlaubt")
    allowed = {_norm(t) for t in allowed_tables} if allowed_tables else None
    blocked = {
        _norm(t): {_norm(c) for c in cols}
        for t, cols in (blocked_columns or {}).items()
        if cols
    }
    cte_names = {_norm(c.alias) for c in root.find_all(exp.CTE)}
    tables_in_query: set[str] = set()
    for node in root.walk():
        if isinstance(node, DENY_NODES):
            raise SqlRejected("Nur lesende SELECT-Abfragen sind erlaubt")
        if isinstance(node, (exp.Func, exp.Anonymous)):
            name = _norm(
                node.name if isinstance(node, exp.Anonymous) else node.sql_name()
            )
            if name in DENY_FUNCS:
                raise SqlRejected(f"Die Funktion {name} ist nicht erlaubt")
        if isinstance(node, exp.Table):
            tname = _norm(node.name)
            if not tname:  # Tabellenfunktion o. ae.
                raise SqlRejected("Tabellenfunktionen sind nicht erlaubt")
            if tname in cte_names and not node.args.get("db"):
                continue
            tables_in_query.add(tname)
            if allowed is not None and tname not in allowed:
                raise SqlRejected(f"Die Tabelle {node.name} ist nicht freigegeben")
    for node in root.find_all(exp.Column):
        col = _norm(node.name)
        owner = _norm(node.table)
        for tname, cols in blocked.items():
            if col in cols and (
                not owner or owner == tname or tname in tables_in_query
            ):
                raise SqlRejected(f"Die Spalte {node.name} ist gesperrt")
    hit = [t for t in blocked if t in tables_in_query]
    if hit:
        # Ganze Zeile ohne Spaltennamen (PostgreSQL: "SELECT items FROM items", row_to_json(i), json_agg(i)):
        # Tabellenname bzw. Alias allein als Wert wuerde die gesperrte Spalte mitliefern.
        row_names = set(hit)
        for node in root.find_all(exp.Table):
            if _norm(node.name) in blocked and node.alias:
                row_names.add(_norm(node.alias))
        for node in root.find_all(exp.Column):
            if _norm(node.name) in row_names and _norm(node.table) not in row_names:
                raise SqlRejected(
                    "Die ganze Zeile einer Tabelle mit gesperrten Spalten ist nicht abfragbar, nenne die Spalten einzeln"
                )
        # COUNT(*) liefert keine Spaltenwerte und bleibt erlaubt
        if any(
            isinstance(n, exp.Star) and not isinstance(n.parent, exp.Count)
            for n in root.walk()
        ):
            raise SqlRejected(
                "SELECT * ist hier nicht erlaubt (gesperrte Spalten), nenne die Spalten einzeln"
            )
    if isinstance(root, exp.Select):
        current = root.args.get("limit")
        if (
            current is not None
            and isinstance(current.expression, exp.Literal)
            and current.expression.is_int
        ):
            row_limit = min(int(current.expression.this), row_limit)
        root = root.limit(row_limit)
    return root.sql(dialect=gdialect)
