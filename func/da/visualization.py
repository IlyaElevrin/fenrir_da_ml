from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


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


def create_dash_app(
    frame: pd.DataFrame,
    configs: list[VisualizationConfig],
    title: str = "Анализ Данных Dashboard",
) -> Any:
    if not configs:
        raise ValueError("Add at least one visualization to the dashboard.")
    for config in configs:
        validate_chart_config(frame, config)

    try:
        from dash import Dash, dcc, html
    except ImportError as exc:
        raise ImportError("dash is required to create dashboards.") from exc

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H1(title),
            *[
                html.Section(
                    [
                        html.H2(config.title or VISUALIZATION_METHODS[config.chart_type]),
                        dcc.Graph(figure=_plotly_figure(frame, config)),
                    ]
                )
                for config in configs
            ],
        ],
        style={"fontFamily": "Arial, sans-serif", "padding": "24px"},
    )
    return app


def _plotly_figure(frame: pd.DataFrame, config: VisualizationConfig) -> Any:
    import plotly.express as px

    title = config.title or VISUALIZATION_METHODS[config.chart_type]
    if config.chart_type == "line":
        return px.line(frame, x=config.x_column, y=config.y_column, title=title)
    if config.chart_type == "bar":
        return px.bar(frame, x=config.x_column, y=config.y_column, title=title)
    if config.chart_type == "pie":
        if config.x_column and config.y_column:
            data = frame.groupby(config.x_column, dropna=False, as_index=False)[config.y_column].sum()
            return px.pie(data, names=config.x_column, values=config.y_column, title=title)
        column = config.x_column or config.y_column
        counts = frame[column].value_counts(dropna=False).reset_index()
        counts.columns = [column, "count"]
        return px.pie(counts, names=column, values="count", title=title)
    if config.chart_type == "heatmap":
        corr = frame.select_dtypes(include="number").corr()
        return px.imshow(corr, text_auto=True, aspect="auto", title=title)
    if config.chart_type == "scatter":
        return px.scatter(frame, x=config.x_column, y=config.y_column, title=title)
    if config.chart_type == "histogram":
        column = config.x_column or config.y_column
        return px.histogram(frame, x=column, title=title)
    if config.chart_type == "box":
        column = config.y_column or config.x_column
        return px.box(frame, y=column, title=title)
    raise ValueError(f"Unsupported chart type: {config.chart_type}")
