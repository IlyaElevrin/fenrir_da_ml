from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from func.da.ab_testing import run_ab_test
from func.da.correlation import correlation_matrix
from func.da.statistics import dataset_overview, describe_columns
from func.ml.models import model_options, train_model


APP_NAME = "Fenrir Mining"
ASSET_DIR = Path(__file__).resolve().parent / "assets" / "svg"


def asset_path(name: str) -> Path:
    return ASSET_DIR / f"{name}.svg"


def svg_icon(name: str) -> QIcon:
    return QIcon(str(asset_path(name)))


class FenrirMiningWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1280, 820)
        self.data_frame: pd.DataFrame | None = None
        self.is_dark_theme = True

        self.home_summary = QLabel("Данные еще не загружены.")
        self.column_summary = QLabel("Выберите CSV или Excel файл, чтобы увидеть структуру таблицы.")
        self.preview_table = QTableWidget()
        self.analysis_columns = QListWidget()
        self.analysis_output = QTextEdit()
        self.analysis_plot = FigureCanvas(Figure(figsize=(7, 4)))
        self.correlation_method = QComboBox()
        self.ab_group_column = QComboBox()
        self.ab_value_column = QComboBox()
        self.ab_control_value = QComboBox()
        self.ab_treatment_value = QComboBox()
        self.ml_feature_columns = QListWidget()
        self.ml_target_column = QComboBox()
        self.ml_model = QComboBox()
        self.ml_output = QTextEdit()

        self.setCentralWidget(self._create_tabs())
        self._create_menu()
        self.apply_theme()

    def _create_menu(self) -> None:
        view_menu = self.menuBar().addMenu("Вид")
        theme_action = view_menu.addAction(svg_icon("theme"), "Переключить тему")
        theme_action.triggered.connect(self.toggle_theme)

    def _create_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._create_home_tab(), svg_icon("home"), "Главная")
        tabs.addTab(self._create_analysis_tab(), svg_icon("analysis"), "Анализ данных")
        tabs.addTab(self._create_ml_tab(), svg_icon("ml"), "Машинное обучение")
        return tabs

    def _create_home_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        hero = QFrame()
        hero.setObjectName("Hero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 24, 24, 24)
        hero_layout.setSpacing(18)

        logo = QSvgWidget(str(asset_path("fenrir-logo")))
        logo.setFixedSize(112, 112)
        hero_layout.addWidget(logo)

        text_box = QVBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("Title")
        description = QLabel(
            "Кросс-платформенная среда интеллектуального анализа данных: загрузка таблиц, "
            "статистический анализ, корреляции, A/B тесты и обучение базовых ML моделей."
        )
        description.setWordWrap(True)
        description.setObjectName("MutedText")
        text_box.addWidget(title)
        text_box.addWidget(description)
        hero_layout.addLayout(text_box, stretch=1)

        load_button = QPushButton("Загрузить CSV / Excel")
        load_button.setIcon(svg_icon("upload"))
        load_button.clicked.connect(self.load_data)
        hero_layout.addWidget(load_button)
        layout.addWidget(hero)

        summary_panel = QFrame()
        summary_panel.setObjectName("Panel")
        summary_layout = QVBoxLayout(summary_panel)
        summary_title = QLabel("Обзор данных")
        summary_title.setObjectName("SectionTitle")
        self.home_summary.setWordWrap(True)
        self.column_summary.setWordWrap(True)
        self.column_summary.setObjectName("MutedText")
        summary_layout.addWidget(summary_title)
        summary_layout.addWidget(self.home_summary)
        summary_layout.addWidget(self.column_summary)
        layout.addWidget(summary_panel)

        preview_panel = QFrame()
        preview_panel.setObjectName("Panel")
        preview_layout = QVBoxLayout(preview_panel)
        preview_title = QLabel("Первые пять строк")
        preview_title.setObjectName("SectionTitle")
        self.preview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)
        preview_layout.addWidget(preview_title)
        preview_layout.addWidget(self.preview_table)
        layout.addWidget(preview_panel, stretch=1)
        return page

    def _create_analysis_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        controls = QFrame()
        controls.setObjectName("Panel")
        controls.setMinimumWidth(310)
        controls_layout = QVBoxLayout(controls)

        controls_layout.addWidget(_section_label("Столбцы для анализа"))
        self.analysis_columns.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        controls_layout.addWidget(self.analysis_columns)

        select_numeric = QPushButton("Выбрать числовые")
        select_numeric.setIcon(svg_icon("columns"))
        select_numeric.clicked.connect(self.select_numeric_analysis_columns)
        controls_layout.addWidget(select_numeric)

        stats_button = QPushButton("Описательная статистика")
        stats_button.setIcon(svg_icon("stats"))
        stats_button.clicked.connect(self.show_descriptive_statistics)
        controls_layout.addWidget(stats_button)

        self.correlation_method.addItems(["pearson", "spearman", "kendall"])
        controls_layout.addWidget(_field_label("Метод корреляции"))
        controls_layout.addWidget(self.correlation_method)

        corr_button = QPushButton("Матрица корреляций")
        corr_button.setIcon(svg_icon("correlation"))
        corr_button.clicked.connect(self.show_correlation)
        controls_layout.addWidget(corr_button)

        controls_layout.addSpacing(12)
        controls_layout.addWidget(_section_label("A/B тест"))
        controls_layout.addWidget(_field_label("Группирующий столбец"))
        controls_layout.addWidget(self.ab_group_column)
        controls_layout.addWidget(_field_label("Метрика"))
        controls_layout.addWidget(self.ab_value_column)
        controls_layout.addWidget(_field_label("Контроль"))
        controls_layout.addWidget(self.ab_control_value)
        controls_layout.addWidget(_field_label("Вариант"))
        controls_layout.addWidget(self.ab_treatment_value)

        ab_button = QPushButton("Запустить A/B тест")
        ab_button.setIcon(svg_icon("ab-test"))
        ab_button.clicked.connect(self.show_ab_test)
        controls_layout.addWidget(ab_button)
        controls_layout.addStretch(1)

        results = QSplitter(Qt.Orientation.Vertical)
        self.analysis_output.setReadOnly(True)
        self.analysis_output.setPlaceholderText("Загрузите данные и выберите действие.")
        results.addWidget(self.analysis_output)
        results.addWidget(self.analysis_plot)
        results.setSizes([300, 420])

        layout.addWidget(controls)
        layout.addWidget(results, stretch=1)
        self.ab_group_column.currentTextChanged.connect(self.refresh_ab_values)
        return page

    def _create_ml_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        controls = QFrame()
        controls.setObjectName("Panel")
        controls.setMinimumWidth(330)
        controls_layout = QVBoxLayout(controls)

        controls_layout.addWidget(_section_label("Конструктор модели"))
        for key, label in model_options().items():
            self.ml_model.addItem(label, key)
        controls_layout.addWidget(_field_label("Алгоритм"))
        controls_layout.addWidget(self.ml_model)

        controls_layout.addWidget(_field_label("Признаки"))
        self.ml_feature_columns.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        controls_layout.addWidget(self.ml_feature_columns)

        controls_layout.addWidget(_field_label("Целевой столбец"))
        controls_layout.addWidget(self.ml_target_column)

        train_button = QPushButton("Обучить модель")
        train_button.setIcon(svg_icon("train"))
        train_button.clicked.connect(self.train_selected_model)
        controls_layout.addWidget(train_button)
        controls_layout.addStretch(1)

        self.ml_output.setReadOnly(True)
        self.ml_output.setPlaceholderText("Загрузите данные, выберите признаки и целевой столбец.")

        layout.addWidget(controls)
        layout.addWidget(self.ml_output, stretch=1)
        return page

    def load_data(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть данные",
            "",
            "Табличные данные (*.csv *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx *.xls)",
        )
        if not file_name:
            return

        path = Path(file_name)
        try:
            frame = self._read_table(path)
            self.set_data(frame, source_name=path.name)
        except Exception as exc:  # pragma: no cover - GUI error path
            self.show_error("Не удалось загрузить файл", exc)

    def set_data(self, frame: pd.DataFrame, source_name: str = "dataset") -> None:
        try:
            overview = dataset_overview(frame)
        except Exception as exc:
            self.show_error("Данные не подходят для анализа", exc)
            return

        self.data_frame = frame
        self.home_summary.setText(
            f"{source_name}: {overview.row_count} строк, {overview.column_count} столбцов, "
            f"пропусков: {sum(overview.missing_by_column.values())}."
        )
        self.column_summary.setText(
            "Числовые столбцы: "
            + (", ".join(overview.numeric_columns) if overview.numeric_columns else "нет")
            + "\nКатегориальные столбцы: "
            + (", ".join(overview.categorical_columns) if overview.categorical_columns else "нет")
        )
        self.populate_preview_table(overview.columns, overview.preview)
        self.refresh_column_controls()
        self.analysis_output.setText("Данные загружены. Выберите анализ слева.")
        self.ml_output.setText("Данные загружены. Выберите признаки, цель и модель.")

    def populate_preview_table(self, columns: list[str], rows: list[dict[str, Any]]) -> None:
        self.preview_table.clear()
        self.preview_table.setColumnCount(len(columns))
        self.preview_table.setRowCount(len(rows))
        self.preview_table.setHorizontalHeaderLabels(columns)

        for row_index, row in enumerate(rows):
            for column_index, column in enumerate(columns):
                value = row.get(column)
                item = QTableWidgetItem("" if value is None else str(value))
                self.preview_table.setItem(row_index, column_index, item)
        self.preview_table.resizeColumnsToContents()

    def refresh_column_controls(self) -> None:
        if self.data_frame is None:
            return

        columns = [str(column) for column in self.data_frame.columns]
        numeric_columns = [
            str(column) for column in self.data_frame.select_dtypes(include="number").columns
        ]

        self.analysis_columns.clear()
        for column in columns:
            item = QListWidgetItem(column)
            self.analysis_columns.addItem(item)
            if column in numeric_columns:
                item.setSelected(True)

        self.ml_feature_columns.clear()
        for column in columns:
            item = QListWidgetItem(column)
            self.ml_feature_columns.addItem(item)
            if column != columns[-1]:
                item.setSelected(True)

        for combo in [self.ab_group_column, self.ab_value_column, self.ml_target_column]:
            combo.clear()
            combo.addItems(columns)

        if numeric_columns:
            self.ab_value_column.setCurrentText(numeric_columns[0])
        if columns:
            self.ml_target_column.setCurrentText(columns[-1])
        self.refresh_ab_values()

    def refresh_ab_values(self) -> None:
        if self.data_frame is None or not self.ab_group_column.currentText():
            return
        group_column = self.ab_group_column.currentText()
        values = sorted(self.data_frame[group_column].dropna().astype(str).unique().tolist())
        self.ab_control_value.clear()
        self.ab_treatment_value.clear()
        self.ab_control_value.addItems(values)
        self.ab_treatment_value.addItems(values)
        if len(values) > 1:
            self.ab_treatment_value.setCurrentIndex(1)

    def select_numeric_analysis_columns(self) -> None:
        if self.data_frame is None:
            return
        numeric_columns = set(self.data_frame.select_dtypes(include="number").columns.astype(str))
        for index in range(self.analysis_columns.count()):
            item = self.analysis_columns.item(index)
            item.setSelected(item.text() in numeric_columns)

    def show_descriptive_statistics(self) -> None:
        if not self._has_data():
            return
        columns = self._selected_items(self.analysis_columns)
        try:
            summary = describe_columns(self.data_frame, columns)
            table = pd.DataFrame(summary).T
            self.analysis_output.setText(table.round(4).to_string())
            self._draw_distribution(columns)
        except Exception as exc:
            self.show_error("Не удалось посчитать статистику", exc)

    def show_correlation(self) -> None:
        if not self._has_data():
            return
        columns = self._selected_items(self.analysis_columns)
        try:
            matrix = correlation_matrix(
                self.data_frame,
                columns,
                method=self.correlation_method.currentText(),
            )
            self.analysis_output.setText(matrix.round(4).to_string())
            self._draw_correlation(matrix)
        except Exception as exc:
            self.show_error("Не удалось построить корреляцию", exc)

    def show_ab_test(self) -> None:
        if not self._has_data():
            return
        try:
            result = run_ab_test(
                self.data_frame,
                group_column=self.ab_group_column.currentText(),
                value_column=self.ab_value_column.currentText(),
                control_value=self.ab_control_value.currentText(),
                treatment_value=self.ab_treatment_value.currentText(),
            )
            self.analysis_output.setText(
                "\n".join(
                    [
                        "A/B тест Welch t-test",
                        f"Группа контроля: {result.control_value} (n={result.control_count})",
                        f"Группа варианта: {result.treatment_value} (n={result.treatment_count})",
                        f"Среднее контроля: {result.control_mean:.4f}",
                        f"Среднее варианта: {result.treatment_mean:.4f}",
                        f"Разница средних: {result.mean_delta:.4f}",
                        f"t-статистика: {result.statistic:.4f}",
                        f"p-value: {result.p_value:.6f}",
                        f"Cohen d: {result.effect_size:.4f}",
                        "Статистически значимо: да" if result.is_significant else "Статистически значимо: нет",
                    ]
                )
            )
            self._draw_ab_test(result)
        except Exception as exc:
            self.show_error("Не удалось выполнить A/B тест", exc)

    def train_selected_model(self) -> None:
        if not self._has_data():
            return
        target = self.ml_target_column.currentText()
        features = [column for column in self._selected_items(self.ml_feature_columns) if column != target]
        try:
            result = train_model(
                self.data_frame,
                model_name=self.ml_model.currentData(),
                feature_columns=features,
                target_column=target,
            )
            self.ml_output.setText(
                "\n".join(
                    [
                        f"Модель: {model_options()[result.model_name]}",
                        f"Целевой столбец: {result.target_column}",
                        f"Метрика: {result.metric}",
                        f"Score: {result.score:.4f}",
                        f"Обучающих строк: {result.train_rows}",
                        f"Тестовых строк: {result.test_rows}",
                        "",
                        "Использованные признаки:",
                        ", ".join(result.feature_names),
                    ]
                )
            )
        except Exception as exc:
            self.show_error("Не удалось обучить модель", exc)

    def _draw_distribution(self, columns: list[str]) -> None:
        assert self.data_frame is not None
        numeric = self.data_frame.loc[:, columns].select_dtypes(include="number")
        self.analysis_plot.figure.clear()
        axes = self.analysis_plot.figure.add_subplot(111)
        if numeric.empty:
            axes.text(0.5, 0.5, "Нет числовых столбцов", ha="center", va="center")
        else:
            numeric.plot(kind="box", ax=axes)
            axes.set_title("Распределение выбранных числовых столбцов")
            axes.tick_params(axis="x", rotation=25)
        self.analysis_plot.figure.tight_layout()
        self.analysis_plot.draw()

    def _draw_correlation(self, matrix: pd.DataFrame) -> None:
        self.analysis_plot.figure.clear()
        axes = self.analysis_plot.figure.add_subplot(111)
        sns.heatmap(matrix, annot=True, cmap="vlag", center=0, ax=axes)
        axes.set_title("Матрица корреляций")
        self.analysis_plot.figure.tight_layout()
        self.analysis_plot.draw()

    def _draw_ab_test(self, result: Any) -> None:
        self.analysis_plot.figure.clear()
        axes = self.analysis_plot.figure.add_subplot(111)
        axes.bar(
            [result.control_value, result.treatment_value],
            [result.control_mean, result.treatment_mean],
            color=["#2f9e9e", "#f0b429"],
        )
        axes.set_ylabel(result.value_column)
        axes.set_title("Средние значения групп")
        self.analysis_plot.figure.tight_layout()
        self.analysis_plot.draw()

    def _selected_items(self, list_widget: QListWidget) -> list[str]:
        return [item.text() for item in list_widget.selectedItems()]

    def _has_data(self) -> bool:
        if self.data_frame is not None:
            return True
        self.show_error("Нет данных", RuntimeError("Сначала загрузите CSV или Excel файл."))
        return False

    def _read_table(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError("Поддерживаются только CSV, XLSX и XLS файлы.")

    def show_error(self, title: str, error: Exception) -> None:
        QMessageBox.warning(self, title, str(error))

    def toggle_theme(self) -> None:
        self.is_dark_theme = not self.is_dark_theme
        self.apply_theme()

    def apply_theme(self) -> None:
        if self.is_dark_theme:
            self.setStyleSheet(
                """
                QMainWindow, QWidget {
                    background: #15181c;
                    color: #edf2f7;
                    font-size: 14px;
                }
                QLabel {
                    background: transparent;
                    border: 0;
                }
                QMenuBar, QMenu {
                    background: #20252b;
                    color: #edf2f7;
                }
                QFrame#Hero, QFrame#Panel, QTextEdit, QTableWidget, QListWidget, QComboBox {
                    background: #20252b;
                    border: 1px solid #39424e;
                    border-radius: 8px;
                }
                QFrame#Hero {
                    background: #1b2026;
                }
                QLabel#Title {
                    font-size: 34px;
                    font-weight: 700;
                    color: #ffffff;
                }
                QLabel#SectionTitle {
                    font-size: 18px;
                    font-weight: 700;
                }
                QLabel#MutedText {
                    color: #a9b4c0;
                }
                QPushButton {
                    background: #2f9e9e;
                    color: #ffffff;
                    border: 0;
                    border-radius: 6px;
                    padding: 9px 12px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: #268989;
                }
                QTabBar::tab {
                    background: #20252b;
                    color: #d8dee9;
                    padding: 10px 16px;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background: #2f9e9e;
                    color: #ffffff;
                }
                QHeaderView::section {
                    background: #2b323a;
                    color: #edf2f7;
                    border: 0;
                    padding: 6px;
                }
                QTableWidget {
                    gridline-color: #303843;
                    alternate-background-color: #252c34;
                    selection-background-color: #2f9e9e;
                    selection-color: #ffffff;
                }
                QTableWidget::item {
                    color: #edf2f7;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QMainWindow, QWidget {
                    background: #f6f7f9;
                    color: #1d2630;
                    font-size: 14px;
                }
                QLabel {
                    background: transparent;
                    border: 0;
                }
                QMenuBar, QMenu {
                    background: #ffffff;
                    color: #1d2630;
                }
                QFrame#Hero, QFrame#Panel, QTextEdit, QTableWidget, QListWidget, QComboBox {
                    background: #ffffff;
                    border: 1px solid #d6dde5;
                    border-radius: 8px;
                }
                QLabel#Title {
                    font-size: 34px;
                    font-weight: 700;
                    color: #0b1f24;
                }
                QLabel#SectionTitle {
                    font-size: 18px;
                    font-weight: 700;
                }
                QLabel#MutedText {
                    color: #5b6775;
                }
                QPushButton {
                    background: #006d77;
                    color: #ffffff;
                    border: 0;
                    border-radius: 6px;
                    padding: 9px 12px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: #005a63;
                }
                QTabBar::tab {
                    background: #e9edf2;
                    color: #1d2630;
                    padding: 10px 16px;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background: #006d77;
                    color: #ffffff;
                }
                QHeaderView::section {
                    background: #eef2f6;
                    color: #1d2630;
                    border: 0;
                    padding: 6px;
                }
                QTableWidget {
                    gridline-color: #d6dde5;
                    alternate-background-color: #f3f6f9;
                    selection-background-color: #006d77;
                    selection-color: #ffffff;
                }
                """
            )


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("SectionTitle")
    return label


def _field_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("MutedText")
    return label


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    window = FenrirMiningWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
