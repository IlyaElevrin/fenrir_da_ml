from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


VISUALIZATION_METHODS: dict[str, str] = {
    "line": "Линейный график",
    "bar": "Столбчатая диаграмма",
    "pie": "Круговая диаграмма",
    "heatmap": "Тепловая карта",
    "scatter": "Scatter plot",
    "histogram": "Гистограмма",
    "box": "Box with mustache",
}


@dataclass(frozen=True)
class VisualizationConfig:
    chart_type: str
    x_column: str | None = None
    y_column: str | None = None
    title: str | None = None


def validate_chart_config(
    frame: pd.DataFrame,
    config: VisualizationConfig,
) -> VisualizationConfig:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("The dataset is empty.")
    if config.chart_type not in VISUALIZATION_METHODS:
        raise ValueError(
            "chart_type must be one of: " + ", ".join(VISUALIZATION_METHODS)
        )

    columns = set(frame.columns.astype(str))
    selected = [column for column in (config.x_column, config.y_column) if column]
    missing = [column for column in selected if column not in columns]
    if missing:
        raise ValueError(f"Unknown column(s): {', '.join(missing)}")

    if config.chart_type in {"line", "bar", "scatter"} and not (
        config.x_column and config.y_column
    ):
        raise ValueError("Choose X and Y columns for this chart type.")
    if config.chart_type in {"pie", "histogram", "box"} and not (
        config.x_column or config.y_column
    ):
        raise ValueError("Choose at least one column for this chart type.")
    if config.chart_type == "heatmap" and len(frame.select_dtypes(include="number").columns) < 2:
        raise ValueError("Heat map requires at least two numeric columns.")
    return config


def render_matplotlib_figure(
    frame: pd.DataFrame,
    config: VisualizationConfig,
    figure: Figure | None = None,
) -> Figure:
    validate_chart_config(frame, config)
    target = figure or Figure(figsize=(7, 4))
    target.clear()
    axes = target.add_subplot(111)
    title = config.title or VISUALIZATION_METHODS[config.chart_type]

    if config.chart_type == "line":
        frame.plot(kind="line", x=config.x_column, y=config.y_column, ax=axes)
    elif config.chart_type == "bar":
        frame.plot(kind="bar", x=config.x_column, y=config.y_column, ax=axes)
    elif config.chart_type == "pie":
        if config.x_column and config.y_column:
            data = frame.groupby(config.x_column, dropna=False)[config.y_column].sum()
        else:
            column = config.x_column or config.y_column
            data = frame[column].value_counts(dropna=False)
        data.plot(kind="pie", autopct="%1.1f%%", ax=axes)
        axes.set_ylabel("")
    elif config.chart_type == "heatmap":
        matrix = frame.select_dtypes(include="number").corr()
        sns.heatmap(matrix, annot=True, cmap="Greys", center=0, ax=axes)
    elif config.chart_type == "scatter":
        axes.scatter(frame[config.x_column], frame[config.y_column], color="#000000")
        axes.set_xlabel(config.x_column)
        axes.set_ylabel(config.y_column)
    elif config.chart_type == "histogram":
        column = config.x_column or config.y_column
        frame[column].plot(kind="hist", bins=20, color="#000000", ax=axes)
        axes.set_xlabel(column)
    elif config.chart_type == "box":
        column = config.y_column or config.x_column
        frame[[column]].plot(kind="box", ax=axes)
    else:
        raise ValueError(f"Unsupported chart type: {config.chart_type}")

    axes.set_title(title)
    axes.tick_params(axis="x", rotation=25)
    target.tight_layout()
    return target


def save_visualization_png(
    frame: pd.DataFrame,
    config: VisualizationConfig,
    output_path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    path = Path(output_path).expanduser()
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = render_matplotlib_figure(frame, config)
    figure.savefig(path, format="png", dpi=dpi, bbox_inches="tight")
    return path
