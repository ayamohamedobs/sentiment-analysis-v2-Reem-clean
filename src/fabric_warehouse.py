"""
Fabric Data Warehouse connector.

Reads survey response data from a Fabric warehouse using
pyodbc + Service Principal (client secret) token authentication.
"""

from __future__ import annotations

import os
import struct
from typing import Any

import pandas as pd
import pyodbc
from azure.identity import ClientSecretCredential, DefaultAzureCredential


def _get_access_token() -> str:
    """Get an AAD access token scoped to Azure SQL / Fabric.
    
    Uses Service Principal (ClientSecretCredential) if FABRIC_CLIENT_ID,
    FABRIC_CLIENT_SECRET, and FABRIC_TENANT_ID are set.
    Falls back to DefaultAzureCredential otherwise.
    """
    tenant = os.environ.get("FABRIC_TENANT_ID")
    client_id = os.environ.get("FABRIC_CLIENT_ID")
    client_secret = os.environ.get("FABRIC_CLIENT_SECRET")

    if tenant and client_id and client_secret:
        credential = ClientSecretCredential(
            tenant_id=tenant,
            client_id=client_id,
            client_secret=client_secret,
        )
    else:
        credential = DefaultAzureCredential()

    token = credential.get_token("https://database.windows.net/.default")
    return token.token


def _build_token_bytes(token: str) -> bytes:
    """Encode the access token into the binary format pyodbc expects."""
    token_bytes = token.encode("UTF-16-LE")
    return struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)


def _detect_driver() -> str:
    """Find the best available SQL Server ODBC driver."""
    for name in ["ODBC Driver 18 for SQL Server",
                 "ODBC Driver 17 for SQL Server",
                 "SQL Server"]:
        for d in pyodbc.drivers():
            if d == name:
                return name
    raise RuntimeError(
        "No SQL Server ODBC driver found. Install 'ODBC Driver 18 for SQL Server' from "
        "https://go.microsoft.com/fwlink/?linkid=2266337"
    )


def get_connection() -> pyodbc.Connection:
    """Open an authenticated connection to the Fabric Data Warehouse."""
    endpoint = os.environ["FABRIC_SQL_ENDPOINT"]
    database = os.environ["FABRIC_DATABASE"]
    driver = _detect_driver()

    token = _get_access_token()
    token_struct = _build_token_bytes(token)

    conn_str = (
        f"Driver={{{driver}}};"
        f"Server={endpoint};"
        f"Database={database};"
        "Encrypt=Yes;"
        "TrustServerCertificate=No;"
    )
    # SQL_COPT_SS_ACCESS_TOKEN = 1256
    conn = pyodbc.connect(conn_str, attrs_before={1256: token_struct})
    return conn


def list_views() -> list[str]:
    """List all tables and views in the warehouse (for the UI dropdown)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT TABLE_SCHEMA + '.' + TABLE_NAME "
            "FROM INFORMATION_SCHEMA.TABLES "
            "ORDER BY TABLE_SCHEMA, TABLE_NAME"
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def list_columns(view_name: str) -> list[str]:
    """List column names for a given view."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        # Use parameterised schema/table split
        parts = view_name.split(".", 1)
        schema = parts[0] if len(parts) == 2 else "dbo"
        table = parts[-1]
        cursor.execute(
            "SELECT COLUMN_NAME "
            "FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? "
            "ORDER BY ORDINAL_POSITION",
            schema,
            table,
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def load_distinct_values(view_name: str, column: str, limit: int = 500) -> list[str]:
    """Return sorted distinct non-null values for *column* in *view_name*.

    Used to populate filter dropdowns in the UI.
    """
    if not _is_safe_identifier(view_name):
        raise ValueError(f"Invalid view name: {view_name}")
    if not _is_safe_column(column):
        raise ValueError(f"Invalid column name: {column}")

    conn = get_connection()
    try:
        query = (
            f"SELECT DISTINCT TOP {int(limit)} [{column}] "
            f"FROM {view_name} "
            f"WHERE [{column}] IS NOT NULL "
            f"ORDER BY [{column}]"
        )
        cursor = conn.cursor()
        cursor.execute(query)
        return [str(row[0]) for row in cursor.fetchall()]
    finally:
        conn.close()


def count_rows(view_name: str, filters: dict[str, Any] | None = None) -> int:
    """Return the number of rows matching *filters* (or total if None)."""
    if not _is_safe_identifier(view_name):
        raise ValueError(f"Invalid view name: {view_name}")

    conn = get_connection()
    try:
        where_parts: list[str] = []
        params: list[Any] = []
        if filters:
            for col, val in filters.items():
                if not _is_safe_column(col):
                    raise ValueError(f"Invalid column name: {col}")
                if isinstance(val, (list, set)):
                    placeholders = ", ".join("?" for _ in val)
                    where_parts.append(f"[{col}] IN ({placeholders})")
                    params.extend(val)
                elif isinstance(val, tuple) and len(val) == 2:
                    where_parts.append(f"[{col}] >= ? AND [{col}] <= ?")
                    params.extend(val)
                else:
                    where_parts.append(f"[{col}] = ?")
                    params.append(val)
        where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        query = f"SELECT COUNT(*) FROM {view_name}{where_clause}"  # noqa: S608
        cursor = conn.cursor()
        cursor.execute(query, params or [])
        return cursor.fetchone()[0]
    finally:
        conn.close()


def load_view(
    view_name: str,
    limit: int | None = None,
    filters: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a warehouse view into a DataFrame with optional SQL filters.

    Parameters
    ----------
    view_name : str
        Fully-qualified view name, e.g. 'dbo.vw_survey_responses'.
    limit : int | None
        Optional row limit (for previews). None = all rows.
    filters : dict | None
        Column-name → value mapping used to build a WHERE clause.
        Supported value types:
        - tuple of two dates ``(start, end)`` → ``BETWEEN ? AND ?``
        - list/set of strings → ``IN (?, ?, ...)``
        - single scalar → ``= ?``
    """
    # Validate view_name to prevent SQL injection — must be schema.name format
    if not _is_safe_identifier(view_name):
        raise ValueError(f"Invalid view name: {view_name}")

    conn = get_connection()
    try:
        top_clause = f"TOP {int(limit)}" if limit else ""
        where_parts: list[str] = []
        params: list[Any] = []

        if filters:
            for col, val in filters.items():
                if not _is_safe_column(col):
                    raise ValueError(f"Invalid column name: {col}")
                if isinstance(val, (list, set)):
                    placeholders = ", ".join("?" for _ in val)
                    where_parts.append(f"[{col}] IN ({placeholders})")
                    params.extend(val)
                elif isinstance(val, tuple) and len(val) == 2:
                    where_parts.append(f"[{col}] >= ? AND [{col}] <= ?")
                    params.extend(val)
                else:
                    where_parts.append(f"[{col}] = ?")
                    params.append(val)

        where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        query = f"SELECT {top_clause} * FROM {view_name}{where_clause}"  # noqa: S608
        return pd.read_sql(query, conn, params=params or None)
    finally:
        conn.close()


def _is_safe_column(name: str) -> bool:
    """Allow column names with alphanumeric, spaces, slashes, parens, colons."""
    import re
    return bool(re.match(r"^[A-Za-z0-9 _/():\-]+$", name))


def _is_safe_identifier(name: str) -> bool:
    """Allow only schema.table format with alphanumeric + underscores."""
    import re
    return bool(re.match(r"^[A-Za-z_]\w*\.[A-Za-z_]\w*$", name))
