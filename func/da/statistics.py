from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetOverview:
    row_count: int
    column_count: int
    columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    missing_by_column: dict[str, int]
    preview: list[dict[str, Any]]


def selectable_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns in the order users see them in the loaded data."""
    _require_frame(frame)
    return [str(column) for column in frame.columns]


def dataset_overview(frame: pd.DataFrame, preview_rows: int = 5) -> DatasetOverview:
    _require_frame(frame)
    if preview_rows < 1:
        raise ValueError("preview_rows must be at least 1")

    return DatasetOverview(
        row_count=int(frame.shape[0]),
        column_count=int(frame.shape[1]),
        columns=selectable_columns(frame),
        numeric_columns=_typed_columns(frame, include=[np.number]),
        categorical_columns=_non_numeric_columns(frame),
        missing_by_column={str(column): int(count) for column, count in frame.isna().sum().items()},
        preview=_records(frame.head(preview_rows)),
    )


def describe_columns(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, float | None]]:
    _require_frame(frame)
    selected = _select_columns(frame, columns)
    numeric = selected.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("Choose at least one numeric column for descriptive statistics.")

    description = numeric.describe().replace({np.nan: None})
    return {
        str(column): {str(metric): _to_optional_float(value) for metric, value in values.items()}
        for column, values in description.to_dict().items()
    }


def _require_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("The dataset is empty.")


def _select_columns(frame: pd.DataFrame, columns: list[str] | tuple[str, ...] | None) -> pd.DataFrame:
    if not columns:
        return frame

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {', '.join(map(str, missing))}")
    return frame.loc[:, list(columns)]


def _typed_columns(frame: pd.DataFrame, include: list[Any]) -> list[str]:
    return [str(column) for column in frame.select_dtypes(include=include).columns]


def _non_numeric_columns(frame: pd.DataFrame) -> list[str]:
    numeric = set(frame.select_dtypes(include=[np.number]).columns)
    return [str(column) for column in frame.columns if column not in numeric]


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = frame.replace({np.nan: None})
    return normalized.to_dict(orient="records")


def _to_optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
