from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_FILE_EXTENSIONS: tuple[str, ...] = (".csv", ".xlsx", ".xls", ".txt", ".parquet")


def load_table_from_file(path: str | Path, **read_kwargs) -> pd.DataFrame:
    """Read a tabular file into a DataFrame.

    Supports CSV, Excel, plain text (auto-detected delimiter), and Parquet.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path, **read_kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, **read_kwargs)
    if suffix == ".txt":
        # sep=None with engine="python" auto-detects the delimiter.
        kwargs = {"sep": None, "engine": "python"}
        kwargs.update(read_kwargs)
        return pd.read_csv(file_path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(file_path, **read_kwargs)
    raise ValueError(
        "Unsupported file type. Supported extensions: " + ", ".join(SUPPORTED_FILE_EXTENSIONS)
    )


def _create_engine(connection: str):
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise ImportError(
            "SQLAlchemy is required to load tables from a database. Install it via 'pip install SQLAlchemy'."
        ) from exc
    return create_engine(connection)


def list_database_tables(connection: str) -> list[str]:
    """Return the list of tables for a SQLAlchemy-compatible connection string."""
    if not connection:
        raise ValueError("connection must be a non-empty connection string")
    from sqlalchemy import inspect

    engine = _create_engine(connection)
    try:
        inspector = inspect(engine)
        return [str(table) for table in inspector.get_table_names()]
    finally:
        engine.dispose()


def load_database_table(connection: str, table_name: str) -> pd.DataFrame:
    """Load a single table from a SQLAlchemy-compatible connection string."""
    if not connection:
        raise ValueError("connection must be a non-empty connection string")
    if not table_name:
        raise ValueError("table_name must be a non-empty string")

    engine = _create_engine(connection)
    try:
        return pd.read_sql_table(table_name, engine)
    finally:
        engine.dispose()
