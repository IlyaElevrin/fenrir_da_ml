from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ABTestResult:
    group_column: str
    value_column: str
    control_value: str
    treatment_value: str
    control_count: int
    treatment_count: int
    control_mean: float
    treatment_mean: float
    mean_delta: float
    statistic: float
    p_value: float
    effect_size: float

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05


def run_ab_test(
    frame: pd.DataFrame,
    group_column: str,
    value_column: str,
    control_value: str,
    treatment_value: str,
) -> ABTestResult:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    _require_columns(frame, [group_column, value_column])
    if control_value == treatment_value:
        raise ValueError("Control and treatment groups must be different.")

    numeric_values = pd.to_numeric(frame[value_column], errors="coerce")
    grouped = pd.DataFrame({"group": frame[group_column].astype(str), "value": numeric_values}).dropna()
    control = grouped.loc[grouped["group"] == str(control_value), "value"]
    treatment = grouped.loc[grouped["group"] == str(treatment_value), "value"]

    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("A/B testing requires at least two numeric rows in each selected group.")

    statistic, p_value = stats.ttest_ind(control, treatment, equal_var=False, nan_policy="omit")
    if np.isnan(p_value):
        p_value = 1.0
    if np.isnan(statistic):
        statistic = 0.0

    control_mean = float(control.mean())
    treatment_mean = float(treatment.mean())
    return ABTestResult(
        group_column=str(group_column),
        value_column=str(value_column),
        control_value=str(control_value),
        treatment_value=str(treatment_value),
        control_count=int(len(control)),
        treatment_count=int(len(treatment)),
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        mean_delta=treatment_mean - control_mean,
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size=_cohens_d(control.to_numpy(), treatment.to_numpy()),
    )


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {', '.join(map(str, missing))}")


def _cohens_d(control: np.ndarray, treatment: np.ndarray) -> float:
    control_var = np.var(control, ddof=1)
    treatment_var = np.var(treatment, ddof=1)
    pooled = np.sqrt((control_var + treatment_var) / 2)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((np.mean(treatment) - np.mean(control)) / pooled)
