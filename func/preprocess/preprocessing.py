from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


MISSING_STRATEGIES: tuple[str, ...] = ("drop", "mean", "median", "knn")
JOIN_METHODS: tuple[str, ...] = ("inner", "left", "right", "outer", "cross")
TYPE_CONVERSION_OPTIONS: tuple[str, ...] = (
    "string",
    "integer",
    "float",
    "boolean",
    "datetime",
    "category",
)


def _require_frame(frame: pd.DataFrame, name: str = "frame") -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.empty:
        raise ValueError(f"{name} is empty")


def _resolve_columns(frame: pd.DataFrame, columns: Iterable[str] | None) -> list[str]:
    if not columns:
        return list(frame.columns.astype(str))
    selected = [str(column) for column in columns]
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {', '.join(missing)}")
    return selected


def drop_missing(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Drop rows with missing values in the selected columns (or all)."""
    _require_frame(frame)
    subset = _resolve_columns(frame, columns)
    return frame.dropna(subset=subset).reset_index(drop=True)


def impute_missing(
    frame: pd.DataFrame,
    strategy: str,
    columns: Iterable[str] | None = None,
    knn_neighbors: int = 5,
) -> pd.DataFrame:
    """Fill missing values using the requested strategy.

    Strategies:
    - "drop": delegate to :func:`drop_missing`
    - "mean": fill numeric columns with the column mean
    - "median": fill numeric columns with the column median
    - "knn": fill numeric columns with sklearn KNNImputer
    """
    _require_frame(frame)
    if strategy not in MISSING_STRATEGIES:
        raise ValueError(
            "strategy must be one of: " + ", ".join(MISSING_STRATEGIES)
        )
    if strategy == "drop":
        return drop_missing(frame, columns)

    selected = _resolve_columns(frame, columns)
    numeric_columns = [
        column for column in selected if pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not numeric_columns:
        raise ValueError("No numeric columns available for imputation.")

    result = frame.copy()
    if strategy == "mean":
        for column in numeric_columns:
            result[column] = result[column].fillna(result[column].mean())
        return result
    if strategy == "median":
        for column in numeric_columns:
            result[column] = result[column].fillna(result[column].median())
        return result

    if strategy == "knn":
        if knn_neighbors < 1:
            raise ValueError("knn_neighbors must be at least 1")
        try:
            from sklearn.impute import KNNImputer
        except ImportError as exc:
            raise ImportError("scikit-learn is required for KNN imputation.") from exc
        neighbors = min(knn_neighbors, max(1, len(result) - 1))
        imputer = KNNImputer(n_neighbors=neighbors)
        imputed = imputer.fit_transform(result[numeric_columns])
        result.loc[:, numeric_columns] = imputed
        return result

    raise ValueError(f"Unsupported strategy: {strategy}")


def convert_column_types(frame: pd.DataFrame, conversions: dict[str, str]) -> pd.DataFrame:
    """Convert each requested column to its selected pandas dtype.

    Conversion errors are coerced to missing values where pandas provides a
    nullable dtype. This keeps the operation usable from the GUI when users
    need to clean mixed text/numeric columns before analysis or ML.
    """
    _require_frame(frame)
    if not conversions:
        raise ValueError("Choose at least one column type conversion.")

    missing = [column for column in conversions if column not in frame.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {', '.join(missing)}")

    invalid = [
        dtype for dtype in conversions.values() if dtype not in TYPE_CONVERSION_OPTIONS
    ]
    if invalid:
        raise ValueError(
            "dtype must be one of: " + ", ".join(TYPE_CONVERSION_OPTIONS)
        )

    result = frame.copy()
    for column, dtype in conversions.items():
        result[column] = _convert_series(result[column], dtype)
    return result


def _convert_series(series: pd.Series, dtype: str) -> pd.Series:
    if dtype == "string":
        return series.astype("string")
    if dtype == "integer":
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if dtype == "float":
        return pd.to_numeric(series, errors="coerce").astype("Float64")
    if dtype == "boolean":
        return _to_boolean(series)
    if dtype == "datetime":
        return pd.to_datetime(series, errors="coerce")
    if dtype == "category":
        return series.astype("category")
    raise ValueError(f"Unsupported dtype: {dtype}")


def _to_boolean(series: pd.Series) -> pd.Series:
    truthy = {"1", "true", "t", "yes", "y", "да", "д", "истина"}
    falsy = {"0", "false", "f", "no", "n", "нет", "н", "ложь"}

    def normalize(value: object) -> bool | pd.NA:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in truthy:
            return True
        if normalized in falsy:
            return False
        return pd.NA

    return series.map(normalize).astype("boolean")


def pivot_table(
    frame: pd.DataFrame,
    index: Iterable[str],
    columns: Iterable[str] | None = None,
    values: Iterable[str] | None = None,
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """Wrap :func:`pandas.pivot_table` with friendlier validation."""
    _require_frame(frame)
    index_columns = _resolve_columns(frame, index)
    if not index_columns:
        raise ValueError("Choose at least one index column for the pivot table.")
    column_columns = _resolve_columns(frame, columns) if columns else None
    value_columns = _resolve_columns(frame, values) if values else None

    table = pd.pivot_table(
        frame,
        index=index_columns,
        columns=column_columns,
        values=value_columns,
        aggfunc=aggfunc,
    )
    return table.reset_index()


def join_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str = "inner",
    on: Iterable[str] | str | None = None,
    left_on: Iterable[str] | str | None = None,
    right_on: Iterable[str] | str | None = None,
    suffixes: tuple[str, str] = ("_left", "_right"),
) -> pd.DataFrame:
    """Merge two tables using a pandas join method."""
    _require_frame(left, name="left")
    _require_frame(right, name="right")
    if how not in JOIN_METHODS:
        raise ValueError("how must be one of: " + ", ".join(JOIN_METHODS))

    if how == "cross":
        return left.merge(right, how="cross", suffixes=suffixes)

    if on:
        return left.merge(right, how=how, on=list(on) if isinstance(on, (list, tuple)) else on, suffixes=suffixes)
    if left_on and right_on:
        return left.merge(
            right,
            how=how,
            left_on=list(left_on) if isinstance(left_on, (list, tuple)) else left_on,
            right_on=list(right_on) if isinstance(right_on, (list, tuple)) else right_on,
            suffixes=suffixes,
        )
    raise ValueError("Specify the join keys via 'on' or both 'left_on' and 'right_on'.")


def group_table(
    frame: pd.DataFrame,
    by: Iterable[str],
    aggregations: dict[str, str] | None = None,
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """Group a table by columns and aggregate numeric columns.

    If ``aggregations`` is omitted, every numeric column not in ``by`` is
    aggregated using ``aggfunc``.
    """
    _require_frame(frame)
    by_columns = _resolve_columns(frame, by)
    if not by_columns:
        raise ValueError("Choose at least one column to group by.")
    if aggregations is None:
        numeric = [
            column
            for column in frame.columns
            if column not in by_columns and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if not numeric:
            raise ValueError("No numeric columns available for aggregation.")
        aggregations = {column: aggfunc for column in numeric}

    grouped = frame.groupby(by_columns, dropna=False).agg(aggregations)
    return grouped.reset_index()
