from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
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


@dataclass(frozen=True)
class ValidationResult:
    model_name: str
    metric: str
    target_column: str
    feature_names: list[str]
    fold_count: int
    fold_scores: list[float]
    mean_score: float
    std_score: float


ModelParameterSpec = dict[str, Any]
ModelBuilder = Callable[[int | None, dict[str, Any]], object]


def model_options() -> dict[str, str]:
    return {
        "decision_tree": "Decision Tree Classifier",
        "random_forest": "Random Forest Classifier",
        "logistic_regression": "Logistic Regression",
        "linear_regression": "Linear Regression",
    }


def model_parameter_specs(model_name: str) -> list[ModelParameterSpec]:
    specs: dict[str, list[ModelParameterSpec]] = {
        "decision_tree": [
            _choice_spec("criterion", "Критерий", ["gini", "entropy", "log_loss"], "gini"),
            _optional_int_spec("max_depth", "Максимальная глубина", 0, 100, None),
            _int_spec("min_samples_split", "Мин. объектов для разделения", 2, 100, 2),
            _int_spec("min_samples_leaf", "Мин. объектов в листе", 1, 100, 1),
            _choice_spec("max_features", "Максимум признаков", [None, "sqrt", "log2"], None),
            _choice_spec("class_weight", "Вес классов", [None, "balanced"], None),
        ],
        "random_forest": [
            _int_spec("n_estimators", "Количество деревьев", 10, 1000, 150),
            _choice_spec("criterion", "Критерий", ["gini", "entropy", "log_loss"], "gini"),
            _optional_int_spec("max_depth", "Максимальная глубина", 0, 100, None),
            _int_spec("min_samples_split", "Мин. объектов для разделения", 2, 100, 2),
            _int_spec("min_samples_leaf", "Мин. объектов в листе", 1, 100, 1),
            _choice_spec("max_features", "Максимум признаков", ["sqrt", "log2", None], "sqrt"),
            _bool_spec("bootstrap", "Bootstrap", True),
            _choice_spec("class_weight", "Вес классов", [None, "balanced", "balanced_subsample"], None),
        ],
        "logistic_regression": [
            _float_spec("C", "Сила регуляризации C", 0.01, 100.0, 1.0, 0.1),
            _int_spec("max_iter", "Максимум итераций", 100, 10000, 1000),
            _choice_spec("solver", "Solver", ["lbfgs", "liblinear", "newton-cg", "saga"], "lbfgs"),
            _bool_spec("fit_intercept", "Свободный член", True),
            _choice_spec("class_weight", "Вес классов", [None, "balanced"], None),
        ],
        "linear_regression": [
            _bool_spec("fit_intercept", "Свободный член", True),
            _bool_spec("copy_X", "Копировать X", True),
            _bool_spec("positive", "Положительные коэффициенты", False),
        ],
    }
    try:
        return specs[model_name]
    except KeyError as exc:
        available = ", ".join(model_options())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}") from exc


def create_model(
    model_name: str,
    random_state: int | None = None,
    model_params: dict[str, Any] | None = None,
) -> object:
    params = _validated_model_params(model_name, model_params)
    builders: dict[str, ModelBuilder] = {
        "decision_tree": lambda seed, options: DecisionTreeClassifier(random_state=seed, **options),
        "random_forest": lambda seed, options: RandomForestClassifier(random_state=seed, **options),
        "logistic_regression": lambda seed, options: LogisticRegression(random_state=seed, **options),
        "linear_regression": lambda _seed, options: LinearRegression(**options),
    }
    try:
        return builders[model_name](random_state, params)
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
    model_params: dict[str, Any] | None = None,
) -> TrainingResult:
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")

    features, target = prepare_training_data(frame, feature_columns, target_column)
    model = create_model(model_name, random_state=random_state, model_params=model_params)
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


def cross_validate_model(
    frame: pd.DataFrame,
    model_name: str,
    feature_columns: list[str],
    target_column: str,
    cv_folds: int = 5,
    random_state: int | None = 42,
    model_params: dict[str, Any] | None = None,
) -> ValidationResult:
    if not 2 <= cv_folds <= 20:
        raise ValueError("cv_folds must be between 2 and 20")

    features, target = prepare_training_data(frame, feature_columns, target_column)
    if len(features) < cv_folds:
        raise ValueError("Cross validation requires at least as many complete rows as folds.")

    model = create_model(model_name, random_state=random_state, model_params=model_params)
    cv = _cross_validation_splitter(target, model_name, cv_folds, random_state)
    metric = "r2" if model_name == "linear_regression" else "accuracy"
    scores = cross_val_score(model, features, target, cv=cv, scoring=metric)
    return ValidationResult(
        model_name=model_name,
        metric="R2" if model_name == "linear_regression" else "accuracy",
        target_column=target_column,
        feature_names=[str(column) for column in features.columns],
        fold_count=cv_folds,
        fold_scores=[float(score) for score in scores],
        mean_score=float(np.mean(scores)),
        std_score=float(np.std(scores)),
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


def _cross_validation_splitter(
    target: pd.Series,
    model_name: str,
    cv_folds: int,
    random_state: int | None,
) -> KFold | StratifiedKFold:
    if model_name != "linear_regression":
        counts = target.value_counts()
        if not counts.empty and counts.min() >= cv_folds:
            return StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=random_state,
            )
    return KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)


def _validated_model_params(
    model_name: str,
    model_params: dict[str, Any] | None,
) -> dict[str, Any]:
    specs = model_parameter_specs(model_name)
    defaults = {spec["name"]: spec["default"] for spec in specs}
    if not model_params:
        return defaults

    allowed = set(defaults)
    unknown = sorted(set(model_params) - allowed)
    if unknown:
        raise ValueError(f"Unsupported parameter(s) for {model_name}: {', '.join(unknown)}")

    params = defaults | model_params
    if model_name == "logistic_regression" and params.get("solver") != "liblinear":
        params.pop("dual", None)
    return params


def _choice_spec(
    name: str,
    label: str,
    options: list[str | None],
    default: str | None,
) -> ModelParameterSpec:
    return {"name": name, "label": label, "type": "choice", "options": options, "default": default}


def _int_spec(
    name: str,
    label: str,
    minimum: int,
    maximum: int,
    default: int,
) -> ModelParameterSpec:
    return {
        "name": name,
        "label": label,
        "type": "int",
        "min": minimum,
        "max": maximum,
        "default": default,
    }


def _optional_int_spec(
    name: str,
    label: str,
    minimum: int,
    maximum: int,
    default: int | None,
) -> ModelParameterSpec:
    return {
        "name": name,
        "label": label,
        "type": "optional_int",
        "min": minimum,
        "max": maximum,
        "default": default,
    }


def _float_spec(
    name: str,
    label: str,
    minimum: float,
    maximum: float,
    default: float,
    step: float,
) -> ModelParameterSpec:
    return {
        "name": name,
        "label": label,
        "type": "float",
        "min": minimum,
        "max": maximum,
        "default": default,
        "step": step,
    }


def _bool_spec(name: str, label: str, default: bool) -> ModelParameterSpec:
    return {"name": name, "label": label, "type": "bool", "default": default}
