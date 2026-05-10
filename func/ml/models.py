from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    model: object
    score: float
    metric: str
    target_column: str
    feature_names: list[str]
    train_rows: int
    test_rows: int


ModelBuilder = Callable[[int | None], object]


def model_options() -> dict[str, str]:
    return {
        "decision_tree": "Decision Tree Classifier",
        "random_forest": "Random Forest Classifier",
        "logistic_regression": "Logistic Regression",
        "linear_regression": "Linear Regression",
    }


def create_model(model_name: str, random_state: int | None = None) -> object:
    builders: dict[str, ModelBuilder] = {
        "decision_tree": lambda seed: DecisionTreeClassifier(random_state=seed),
        "random_forest": lambda seed: RandomForestClassifier(n_estimators=150, random_state=seed),
        "logistic_regression": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
        "linear_regression": lambda seed: LinearRegression(),
    }
    try:
        return builders[model_name](random_state)
    except KeyError as exc:
        available = ", ".join(model_options())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}") from exc


def prepare_training_data(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not feature_columns:
        raise ValueError("Choose at least one feature column.")
    missing = [column for column in [*feature_columns, target_column] if column not in frame.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {', '.join(map(str, missing))}")
    if target_column in feature_columns:
        raise ValueError("The target column cannot also be used as a feature.")

    selected = frame.loc[:, [*feature_columns, target_column]].dropna()
    if selected.shape[0] < 4:
        raise ValueError("Training requires at least four complete rows.")

    features = pd.get_dummies(selected.loc[:, feature_columns], drop_first=False)
    features = features.astype(float)
    target = selected.loc[:, target_column]
    return features, target


def train_model(
    frame: pd.DataFrame,
    model_name: str,
    feature_columns: list[str],
    target_column: str,
    random_state: int | None = 42,
    test_size: float = 0.25,
) -> TrainingResult:
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")

    features, target = prepare_training_data(frame, feature_columns, target_column)
    model = create_model(model_name, random_state=random_state)
    stratify = _stratify_target(target, model_name, test_size)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    model.fit(x_train, y_train)
    score = float(model.score(x_test, y_test))
    return TrainingResult(
        model_name=model_name,
        model=model,
        score=score,
        metric="R2" if model_name == "linear_regression" else "accuracy",
        target_column=target_column,
        feature_names=[str(column) for column in features.columns],
        train_rows=int(len(x_train)),
        test_rows=int(len(x_test)),
    )


def _stratify_target(target: pd.Series, model_name: str, test_size: float) -> pd.Series | None:
    if model_name == "linear_regression":
        return None

    counts = target.value_counts()
    if counts.empty or counts.min() < 2:
        return None

    expected_test_rows = round(len(target) * test_size)
    if expected_test_rows < len(counts):
        return None
    return target
