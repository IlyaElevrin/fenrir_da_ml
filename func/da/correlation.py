from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_matrix(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("The dataset is empty.")
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be one of: pearson, spearman, kendall")

    if columns:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Unknown column(s): {', '.join(map(str, missing))}")
        selected = frame.loc[:, list(columns)]
    else:
        selected = frame
    numeric = selected.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        raise ValueError("Choose at least two numeric columns for correlation analysis.")

    return numeric.corr(method=method)
