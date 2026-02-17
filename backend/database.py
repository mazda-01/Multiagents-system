"""
Работа с базой данных PostgreSQL.
"""

import json
import logging
import hashlib
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qs, urlparse

from langchain_core.tools import tool
from psycopg2.pool import ThreadedConnectionPool

from config import (
    DB_CONNECT_TIMEOUT,
    DB_DSN,
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_POOL_MAX_CONN,
    DB_POOL_MIN_CONN,
    DB_PORT,
    DB_USER,
)

logger = logging.getLogger(__name__)

db_pool = None
DB_SCHEMA = None
_db_config: dict[str, Any] = {}
_last_good_db_status: dict[str, Any] | None = None
_last_good_db_status_ts: float | None = None


def _is_connection_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        token in text
        for token in [
            "server closed the connection",
            "connection unexpectedly",
            "connection not open",
            "ssl connection has been closed",
            "connection refused",
            "terminating connection",
            "connection timed out",
            "timeout expired",
        ]
    )


def _get_healthy_connection(max_retries: int = 1):
    global db_pool
    if db_pool is None:
        # Auto-heal pool on accidental drop/reload using last known config (or env defaults).
        init_database_pool(_db_config or None)

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        conn = None
        try:
            conn = db_pool.getconn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return conn
        except Exception as e:
            last_error = e
            logger.warning(f"DB connection is unhealthy (attempt {attempt + 1}/{max_retries + 1}): {e}")
            if conn is not None:
                try:
                    db_pool.putconn(conn, close=True)
                except Exception:
                    pass
            if not _is_connection_error(e):
                break

    raise RuntimeError(str(last_error) if last_error else "Не удалось получить соединение с БД")


def _normalize_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(_db_config)
    if config:
        merged.update({k: v for k, v in config.items() if v is not None})

    dsn = merged.get("dsn") or DB_DSN
    normalized: dict[str, Any] = {
        "dsn": dsn,
        "connect_timeout": int(merged.get("connect_timeout") or DB_CONNECT_TIMEOUT),
        "minconn": int(merged.get("minconn") or DB_POOL_MIN_CONN),
        "maxconn": int(merged.get("maxconn") or DB_POOL_MAX_CONN),
    }

    if not dsn:
        normalized.update(
            {
                "host": merged.get("host") or DB_HOST,
                "port": int(merged.get("port") or DB_PORT),
                "database": merged.get("database") or DB_NAME,
                "user": merged.get("user") or DB_USER,
                "password": merged.get("password") if "password" in merged else DB_PASSWORD,
            }
        )

    return normalized


def _pool_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("dsn"):
        return {
            "minconn": config["minconn"],
            "maxconn": config["maxconn"],
            "dsn": config["dsn"],
            "connect_timeout": config["connect_timeout"],
        }

    return {
        "minconn": config["minconn"],
        "maxconn": config["maxconn"],
        "host": config["host"],
        "port": config["port"],
        "database": config["database"],
        "user": config["user"],
        "password": config["password"],
        "connect_timeout": config["connect_timeout"],
    }


def get_cache_namespace() -> str:
    """Возвращает стабильный namespace кэша для текущей БД."""
    cfg = _normalize_config(_db_config or None)

    host = "db"
    port = str(cfg.get("port") or "")
    database = cfg.get("database") or "default"

    dsn = cfg.get("dsn")
    if dsn:
        try:
            parsed = urlparse(str(dsn))
            if parsed.hostname:
                host = parsed.hostname
            if parsed.port:
                port = str(parsed.port)
            elif not port:
                port = "5432"
            path_db = (parsed.path or "").lstrip("/")
            qs = parse_qs(parsed.query or "")
            query_db = (qs.get("dbname") or qs.get("database") or [""])[0]
            database = path_db or query_db or database
        except Exception:
            # Если DSN нестандартный, fallback на общий namespace.
            pass
    else:
        host = str(cfg.get("host") or host)
        port = str(cfg.get("port") or port)
        database = str(cfg.get("database") or database)

    # Одна физическая БД = одна коллекция (host:port:database), независимо от профиля/пользователя.
    raw = f"{host}:{port}/{database}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    unique_number = int(digest[:12], 16) % 1_000_000_000
    safe_db = "".join(ch if ch.isalnum() else "_" for ch in database.lower())[:40] or "default"
    return f"cache_{safe_db}_{unique_number:09d}"


def init_database_pool(config: dict[str, Any] | None = None):
    global db_pool, _db_config
    normalized = _normalize_config(config)

    if db_pool:
        try:
            db_pool.closeall()
        except Exception:
            pass

    db_pool = ThreadedConnectionPool(**_pool_kwargs(normalized))
    _db_config = normalized
    logger.info("DB pool initialized")
    return db_pool


def reconfigure_database(config: dict[str, Any] | None = None):
    init_database_pool(config)
    load_schema()


def shutdown_database_pool():
    global db_pool, _db_config, _last_good_db_status, _last_good_db_status_ts
    if db_pool:
        db_pool.closeall()
        db_pool = None
    # Explicit disconnect should not be auto-recovered unless user reconnects.
    _db_config = {}
    _last_good_db_status = None
    _last_good_db_status_ts = None


def get_db_status() -> dict[str, Any]:
    global _last_good_db_status, _last_good_db_status_ts
    if db_pool is None:
        if _last_good_db_status is not None and _last_good_db_status_ts is not None:
            age_sec = max(0, int(datetime.now().timestamp() - _last_good_db_status_ts))
            if age_sec <= 120:
                stale = dict(_last_good_db_status)
                stale["connected"] = True
                stale["stale"] = True
                stale["stale_age_sec"] = age_sec
                stale["warning"] = "DB pool is not initialized, returning recent status snapshot"
                return stale
        return {"connected": False, "error": "DB pool is not initialized"}

    conn = None
    try:
        conn = _get_healthy_connection(max_retries=1)
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user")
            database, user = cur.fetchone()
        conn.rollback()
        status = {"connected": True, "database": database, "user": user}
        _last_good_db_status = status
        _last_good_db_status_ts = datetime.now().timestamp()
        return status
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        if _last_good_db_status is not None and _last_good_db_status_ts is not None and _is_connection_error(e):
            age_sec = max(0, int(datetime.now().timestamp() - _last_good_db_status_ts))
            if age_sec <= 120:
                stale = dict(_last_good_db_status)
                stale["connected"] = True
                stale["stale"] = True
                stale["stale_age_sec"] = age_sec
                stale["warning"] = f"Transient DB error, using recent status snapshot: {e}"
                return stale
        return {"connected": False, "error": str(e)}
    finally:
        if conn and db_pool:
            try:
                db_pool.putconn(conn)
            except Exception:
                pass


def _safe_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def get_schema_structured() -> dict[str, Any]:
    if db_pool is None:
        raise RuntimeError("DB pool is not initialized")

    conn = None
    try:
        conn = _get_healthy_connection(max_retries=1)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                  AND table_type = 'BASE TABLE'
                ORDER BY table_schema, table_name;
                """
            )
            table_rows = cursor.fetchall()
            tables = [f"{schema}.{table}" for schema, table in table_rows]

            columns: dict[str, list[dict[str, Any]]] = {}
            for schema, table in table_rows:
                key = f"{schema}.{table}"
                cursor.execute(
                    """
                    SELECT column_name, data_type, is_nullable, ordinal_position
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position;
                    """,
                    (schema, table),
                )
                columns[key] = [
                    {
                        "name": name,
                        "type": ctype,
                        "nullable": nullable,
                        "position": pos,
                    }
                    for name, ctype, nullable, pos in cursor.fetchall()
                ]

            cursor.execute(
                """
                SELECT
                    tc.table_schema || '.' || tc.table_name AS table_name,
                    kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
                ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position;
                """
            )
            primary_keys: dict[str, list[str]] = {}
            for table_name, column_name in cursor.fetchall():
                primary_keys.setdefault(table_name, []).append(column_name)

            cursor.execute(
                """
                SELECT
                    tc.table_schema || '.' || tc.table_name AS from_table,
                    kcu.column_name AS from_column,
                    ccu.table_schema || '.' || ccu.table_name AS to_table,
                    ccu.column_name AS to_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage ccu
                  ON ccu.constraint_name = tc.constraint_name
                 AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                ORDER BY from_table, from_column;
                """
            )
            foreign_keys = [
                {
                    "from_table": from_table,
                    "from_column": from_column,
                    "to_table": to_table,
                    "to_column": to_column,
                }
                for from_table, from_column, to_table, to_column in cursor.fetchall()
            ]

            cursor.execute(
                """
                SELECT schemaname || '.' || tablename AS table_name, indexname, indexdef
                FROM pg_indexes
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY schemaname, tablename, indexname;
                """
            )
            indexes: dict[str, list[dict[str, Any]]] = {}
            for table_name, index_name, index_def in cursor.fetchall():
                indexes.setdefault(table_name, []).append(
                    {"name": index_name, "definition": index_def}
                )

        conn.rollback()
        return {
            "tables": tables,
            "columns": columns,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "indexes": indexes,
            "metadata": {
                "generated_at": int(datetime.now().timestamp()),
                "query_timeout_ms": 10000,
            },
        }
    except Exception:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        if conn is not None and db_pool:
            try:
                db_pool.putconn(conn)
            except Exception:
                pass


@tool
def get_postgres_schema() -> str:
    """Получает структуру всех пользовательских таблиц в PostgreSQL."""
    if db_pool is None:
        return "Ошибка получения схемы: пул БД не инициализирован"

    conn = None
    cursor = None
    try:
        conn = _get_healthy_connection(max_retries=1)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
              AND table_type = 'BASE TABLE'
            ORDER BY table_schema, table_name;
            """
        )
        tables = cursor.fetchall()

        if not tables:
            return "В базе данных нет пользовательских таблиц."

        schema_lines = ["Структура базы данных:\n"]
        for schema, table in tables:
            schema_lines.append(f"Таблица: {schema}.{table}")
            cursor.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position;
                """,
                (schema, table),
            )
            columns_info = cursor.fetchall()
            json_columns: list[str] = []

            for col_name, data_type, is_nullable in columns_info:
                nullable = "" if is_nullable == "YES" else " NOT NULL"
                schema_lines.append(f"  - {col_name} ({data_type.upper()}{nullable})")
                if data_type.lower() in ("json", "jsonb", "user-defined"):
                    json_columns.append(col_name)

            for jcol in json_columns:
                try:
                    cursor.execute(
                        f'SELECT "{jcol}" FROM {schema}.{table} WHERE "{jcol}" IS NOT NULL LIMIT 1'
                    )
                    row = cursor.fetchone()
                    if row and row[0]:
                        sample = str(row[0])
                        if len(sample) > 200:
                            sample = sample[:200] + "..."
                        schema_lines.append(f"    sample {jcol}: {sample}")
                        schema_lines.append(
                            f"    filter hint: WHERE {jcol}->>'key' = 'value'"
                        )
                except Exception:
                    pass

            schema_lines.append("")

        conn.rollback()
        return "\n".join(schema_lines)
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return f"Ошибка получения схемы: {str(e)}"
    finally:
        if cursor:
            try:
                cursor.close()
            except Exception:
                pass
        if conn and db_pool:
            try:
                db_pool.putconn(conn)
            except Exception:
                pass


@tool
def run_sql(query: str) -> str:
    """Выполняет SELECT-запрос к базе данных с защитой от утечек соединений."""
    logger.info(f"SQL query: {query[:100]}...")

    if db_pool is None and not _db_config:
        return json.dumps(
            {"success": False, "error": "Пул БД не инициализирован", "data": None},
            ensure_ascii=False,
        )

    stripped = query.strip()
    if not stripped.upper().startswith(("SELECT", "WITH")):
        return json.dumps(
            {"success": False, "error": "Разрешены только SELECT-запросы", "data": None},
            ensure_ascii=False,
        )

    if "limit" not in stripped.lower():
        query = f"{stripped.rstrip(';')} LIMIT 50"

    conn = None
    cur = None
    try:
        conn = _get_healthy_connection(max_retries=1)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 10000")
        cur.execute(query)

        if cur.description is None:
            conn.rollback()
            result = {
                "success": True,
                "data": [],
                "row_count": 0,
                "message": "Запрос выполнен, но данных нет",
            }
        else:
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
            conn.rollback()
            result_data = [
                {col: _safe_value(val) for col, val in zip(columns, row)} for row in rows
            ]
            result = {
                "success": True,
                "data": result_data,
                "row_count": len(rows),
                "columns": columns,
            }

        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        error_msg = str(e)
        logger.error(f"SQL error: {error_msg}")
        return json.dumps(
            {
                "success": False,
                "error": error_msg,
                "is_connection_error": _is_connection_error(e),
                "data": None,
            },
            ensure_ascii=False,
        )
    finally:
        if cur:
            try:
                cur.close()
            except Exception:
                pass
        if conn and db_pool:
            try:
                db_pool.putconn(conn)
            except Exception:
                pass


def load_schema():
    global DB_SCHEMA
    try:
        DB_SCHEMA = get_postgres_schema.invoke({})
    except Exception as e:
        return f"Ошибка загрузки схемы БД: {e}"
