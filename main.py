from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QByteArray, QSize
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from func.da.ab_testing import run_ab_test
from func.da.correlation import correlation_matrix
from func.da.statistics import dataset_overview, describe_columns
from func.da.visualization import (
    VISUALIZATION_METHODS,
    VisualizationConfig,
    render_matplotlib_figure,
    save_visualization_png,
    validate_chart_config,
)
from func.io.loaders import (
    SUPPORTED_FILE_EXTENSIONS,
    list_database_tables,
    load_database_table,
    load_table_from_file,
)
from func.ml.models import (
    cross_validate_model,
    model_options,
    model_parameter_specs,
    train_model,
)
from func.preprocess.preprocessing import (
    JOIN_METHODS,
    MISSING_STRATEGIES,
    TYPE_CONVERSION_OPTIONS,
    convert_column_types,
    drop_missing,
    group_table,
    impute_missing,
    join_tables,
    pivot_table,
)


APP_NAME = "Анализ Данных"
HOME_TITLE = "Фенрир Анализ Данных"
ASSET_DIR = Path(__file__).resolve().parent / "assets" / "svg"
LOGO_DIR = Path(__file__).resolve().parent / "assets" / "logo"
PREVIEW_ROWS = 5


@dataclass
class LoadedTable:
    name: str
    frame: pd.DataFrame


def asset_path(name: str) -> Path:
    return ASSET_DIR / f"{name}.svg"


def logo_path() -> Path | None:
    for name in ("logo.png", "logo.svg"):
        candidate = LOGO_DIR / name
        if candidate.exists():
            return candidate
    return None


def themed_svg_icon(name: str, color: str, size: int = 24) -> QIcon:
    """Render an SVG asset using *color* for any ``currentColor`` references."""
    path = asset_path(name)
    if not path.exists():
        return QIcon()
    raw = path.read_text(encoding="utf-8")
    tinted = raw.replace("currentColor", color)
    renderer = QSvgRenderer(QByteArray(tinted.encode("utf-8")))
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)



class FenrirMiningWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1320, 860)
        self.setMinimumSize(800, 480)
        self.is_dark_theme = True
        self.tables: list[LoadedTable] = []

        self._icon_targets: list[tuple[Callable[[QIcon], None], str]] = []
        self._tab_icon_targets: list[tuple[int, str]] = []

        # Data Loading widgets
        self.tables_list = QListWidget()
        self.table_summary = QLabel("Таблицы пока не загружены.")
        self.preview_table = QTableWidget()
        self.preview_selector = QComboBox()

        # Data Preprocessing widgets
        self.prep_source_table = QComboBox()
        self.prep_strategy = QComboBox()
        self.prep_columns = QListWidget()
        self.prep_knn_neighbors = QSpinBox()
        self.prep_output_name = QLineEdit()
        self.prep_log = QTextEdit()
        self.pivot_index = QListWidget()
        self.pivot_columns = QListWidget()
        self.pivot_values = QListWidget()
        self.pivot_aggfunc = QComboBox()
        self.join_left = QComboBox()
        self.join_right = QComboBox()
        self.join_method = QComboBox()
        self.join_on = QLineEdit()
        self.group_by = QListWidget()
        self.group_aggfunc = QComboBox()
        self.dtype_table = QTableWidget()

        # Analysis widgets
        self.analysis_table = QComboBox()
        self.analysis_columns = QListWidget()
        self.analysis_output = QTextEdit()
        self.correlation_method = QComboBox()
        self.ab_group_column = QComboBox()
        self.ab_value_column = QComboBox()
        self.ab_control_value = QComboBox()
        self.ab_treatment_value = QComboBox()

        # Machine learning widgets
        self.ml_table = QComboBox()
        self.ml_feature_columns = QListWidget()
        self.ml_target_column = QComboBox()
        self.ml_model = QComboBox()
        self.ml_output = QTextEdit()
        self.ml_parameter_stack = QStackedWidget()
        self.ml_parameter_widgets: dict[str, dict[str, tuple[dict[str, Any], QWidget]]] = {}
        self.ml_test_size = QDoubleSpinBox()
        self.ml_random_state = QSpinBox()
        self.ml_cv_folds = QSpinBox()

        # Visualization widgets
        self.viz_table = QComboBox()
        self.viz_method = QComboBox()
        self.viz_x_column = QComboBox()
        self.viz_y_column = QComboBox()
        self.viz_title = QLineEdit()
        self.viz_plot = FigureCanvas(Figure(figsize=(7, 4)))
        self.saved_visualization_items = QListWidget()
        self._saved_visualizations: list[tuple[str, VisualizationConfig]] = []

        self.tabs = self._create_tabs()
        self.setCentralWidget(self.tabs)
        self._create_menu()
        self._connect_signals()
        self.apply_theme()
        self._refresh_table_dependent_widgets()

    # ----- Layout construction -----------------------------------------

    def _create_menu(self) -> None:
        view_menu = self.menuBar().addMenu("Вид")
        theme_action = view_menu.addAction("Переключить тему")
        theme_action.triggered.connect(self.toggle_theme)
        self._register_icon(theme_action.setIcon, "theme")

    def _create_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        for index, (icon_name, title, factory) in enumerate(
            [
                ("home", "Главная", self._create_home_tab),
                ("upload", "Загрузка данных", self._create_data_loading_tab),
                ("columns", "Предобработка", self._create_preprocessing_tab),
                ("analysis", "Анализ данных", self._create_analysis_tab),
                ("analysis", "Визуализация", self._create_visualization_tab),
                ("ml", "Машинное обучение", self._create_ml_tab),
            ]
        ):
            tabs.addTab(factory(), title)
            self._tab_icon_targets.append((index, icon_name))
        return tabs

    def _create_home_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        hero = QFrame()
        hero.setObjectName("Hero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(28, 28, 28, 28)
        hero_layout.setSpacing(24)

        logo_file = logo_path()
        if logo_file and logo_file.suffix.lower() == ".svg":
            logo: QWidget = QSvgWidget(str(logo_file))
        else:
            logo_label = QLabel()
            if logo_file:
                logo_label.setPixmap(
                    QPixmap(str(logo_file)).scaled(
                        132,
                        132,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo = logo_label
        logo.setFixedSize(132, 132)
        hero_layout.addWidget(logo)

        text_box = QVBoxLayout()
        title = QLabel(HOME_TITLE)
        title.setObjectName("Title")
        description = QLabel(
            "Кросс-платформенная среда для анализа данных и базового машинного "
            "обучения. Программа объединяет загрузку табличных данных из файлов и "
            "баз, предобработку, статистический анализ, визуализацию, сохранение PNG "
            "и обучение классических моделей scikit-learn в едином интерфейсе."
        )
        description.setWordWrap(True)
        description.setObjectName("BodyText")
        text_box.addWidget(title)
        text_box.addWidget(description)
        text_box.addStretch(1)
        hero_layout.addLayout(text_box, stretch=1)
        layout.addWidget(hero)
        layout.addStretch(1)
        return page

    def _create_data_loading_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        controls = QFrame()
        controls.setObjectName("Panel")
        controls.setMinimumWidth(320)
        controls_layout = QVBoxLayout(controls)

        controls_layout.addWidget(_section_label("Источники данных"))

        load_file_button = QPushButton("Загрузить из файла")
        load_file_button.clicked.connect(self.load_from_files)
        self._register_icon(load_file_button.setIcon, "upload")
        controls_layout.addWidget(load_file_button)

        load_db_button = QPushButton("Загрузить из базы данных")
        load_db_button.clicked.connect(self.load_from_database)
        self._register_icon(load_db_button.setIcon, "table")
        controls_layout.addWidget(load_db_button)

        controls_layout.addSpacing(12)
        controls_layout.addWidget(_section_label("Загруженные таблицы"))
        self.tables_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        controls_layout.addWidget(self.tables_list)

        remove_button = QPushButton("Удалить выбранную")
        remove_button.clicked.connect(self.remove_selected_table)
        controls_layout.addWidget(remove_button)

        controls_layout.addStretch(1)
        layout.addWidget(controls)

        right_panel = QFrame()
        right_panel.setObjectName("Panel")
        right_layout = QVBoxLayout(right_panel)

        summary_row = QHBoxLayout()
        summary_row.addWidget(_section_label("Сводка по таблицам"))
        summary_row.addStretch(1)
        right_layout.addLayout(summary_row)
        self.table_summary.setWordWrap(True)
        self.table_summary.setObjectName("BodyText")
        right_layout.addWidget(self.table_summary)

        preview_row = QHBoxLayout()
        preview_row.addWidget(_field_label("Таблица для просмотра"))
        preview_row.addWidget(self.preview_selector, stretch=1)
        right_layout.addLayout(preview_row)

        preview_title = QLabel(f"Первые {PREVIEW_ROWS} строк")
        preview_title.setObjectName("SectionTitle")
        right_layout.addWidget(preview_title)
        self.preview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)
        right_layout.addWidget(self.preview_table, stretch=1)

        layout.addWidget(right_panel, stretch=1)
        return page

    def _create_preprocessing_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        controls = QFrame()
        controls.setObjectName("Panel")
        controls.setMinimumWidth(360)
        controls_layout = QVBoxLayout(controls)

        controls_layout.addWidget(_section_label("Исходная таблица"))
        controls_layout.addWidget(self.prep_source_table)

        controls_layout.addWidget(_field_label("Имя результирующей таблицы"))
        controls_layout.addWidget(self.prep_output_name)

        controls_layout.addWidget(_section_label("Метод предобработки"))
        missing_group = QGroupBox("Пропущенные значения")
        missing_layout = QFormLayout(missing_group)
        self.prep_strategy.addItems(list(MISSING_STRATEGIES))
        self.prep_knn_neighbors.setRange(1, 50)
        self.prep_knn_neighbors.setValue(5)
        missing_layout.addRow("Метод:", self.prep_strategy)
        missing_layout.addRow("Соседей (KNN):", self.prep_knn_neighbors)

        self.prep_columns.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        missing_layout.addRow("Столбцы:", self.prep_columns)

        missing_button = QPushButton("Обработать пропуски")
        missing_button.clicked.connect(self.apply_missing_strategy)
        missing_layout.addRow(missing_button)

        dtype_group = QGroupBox("Преобразование типов")
        dtype_layout = QVBoxLayout(dtype_group)
        self.dtype_table.setColumnCount(3)
        self.dtype_table.setHorizontalHeaderLabels(["Столбец", "Текущий тип", "Новый тип"])
        self.dtype_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.dtype_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        dtype_layout.addWidget(self.dtype_table)
        dtype_button = QPushButton("Преобразовать типы")
        dtype_button.clicked.connect(self.apply_type_conversions)
        dtype_layout.addWidget(dtype_button)

        pivot_group = QGroupBox("Сводная таблица")
        pivot_layout = QFormLayout(pivot_group)
        for widget in (self.pivot_index, self.pivot_columns, self.pivot_values):
            widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        pivot_layout.addRow("Индекс:", self.pivot_index)
        pivot_layout.addRow("Столбцы:", self.pivot_columns)
        pivot_layout.addRow("Значения:", self.pivot_values)
        self.pivot_aggfunc.addItems(["mean", "sum", "median", "min", "max", "count"])
        pivot_layout.addRow("Агрегация:", self.pivot_aggfunc)
        pivot_button = QPushButton("Построить сводную таблицу")
        pivot_button.clicked.connect(self.apply_pivot_table)
        pivot_layout.addRow(pivot_button)
        controls_layout.addWidget(pivot_group)

        join_group = QGroupBox("Объединение таблиц (join)")
        join_layout = QFormLayout(join_group)
        join_layout.addRow("Левая таблица:", self.join_left)
        join_layout.addRow("Правая таблица:", self.join_right)
        self.join_method.addItems(list(JOIN_METHODS))
        join_layout.addRow("Метод:", self.join_method)
        self.join_on.setPlaceholderText("Например: id или id,date")
        join_layout.addRow("Ключи объединения:", self.join_on)
        join_button = QPushButton("Объединить таблицы")
        join_button.clicked.connect(self.apply_join)
        join_layout.addRow(join_button)
        controls_layout.addWidget(join_group)

        group_group = QGroupBox("Группировка")
        group_layout = QFormLayout(group_group)
        self.group_by.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        group_layout.addRow("Группировать по:", self.group_by)
        self.group_aggfunc.addItems(["mean", "sum", "median", "min", "max", "count"])
        group_layout.addRow("Агрегация:", self.group_aggfunc)
        group_button = QPushButton("Сгруппировать таблицу")
        group_button.clicked.connect(self.apply_group)
        group_layout.addRow(group_button)

        method_button, method_stack = _create_method_selector(
            [
                ("Пропущенные значения", missing_group),
                ("Преобразование типов", dtype_group),
                ("Сводная таблица", pivot_group),
                ("Объединение таблиц", join_group),
                ("Группировка", group_group),
            ]
        )
        controls_layout.addWidget(method_button)
        controls_layout.addWidget(method_stack)

        controls_layout.addStretch(1)
        layout.addWidget(controls)

        log_panel = QFrame()
        log_panel.setObjectName("Panel")
        log_layout = QVBoxLayout(log_panel)
        log_layout.addWidget(_section_label("Журнал предобработки"))
        self.prep_log.setReadOnly(True)
        self.prep_log.setPlaceholderText("Здесь появятся результаты операций над таблицами.")
        log_layout.addWidget(self.prep_log)
        layout.addWidget(log_panel, stretch=1)
        return page

    def _create_analysis_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        controls = QFrame()
        controls.setObjectName("Panel")
        controls.setMinimumWidth(320)
        controls_layout = QVBoxLayout(controls)

        controls_layout.addWidget(_section_label("Таблица для анализа"))
        controls_layout.addWidget(self.analysis_table)

        controls_layout.addWidget(_section_label("Столбцы для анализа"))
        self.analysis_columns.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        controls_layout.addWidget(self.analysis_columns)

        select_numeric = QPushButton("Выбрать числовые")
        self._register_icon(select_numeric.setIcon, "columns")
        select_numeric.clicked.connect(self.select_numeric_analysis_columns)
        controls_layout.addWidget(select_numeric)

        controls_layout.addWidget(_section_label("Метод анализа"))
        stats_group = QGroupBox("Описательная статистика")
        stats_layout = QVBoxLayout(stats_group)
        stats_button = QPushButton("Описательная статистика")
        self._register_icon(stats_button.setIcon, "stats")
        stats_button.clicked.connect(self.show_descriptive_statistics)
        stats_layout.addWidget(stats_button)

        corr_group = QGroupBox("Корреляция")
        corr_layout = QFormLayout(corr_group)
        self.correlation_method.addItems(["pearson", "spearman", "kendall"])
        corr_layout.addRow("Метод:", self.correlation_method)

        corr_button = QPushButton("Матрица корреляций")
        self._register_icon(corr_button.setIcon, "correlation")
        corr_button.clicked.connect(self.show_correlation)
        corr_layout.addRow(corr_button)

        ab_group = QGroupBox("A/B тест")
        ab_layout = QFormLayout(ab_group)
        ab_layout.addRow("Группирующий столбец:", self.ab_group_column)
        ab_layout.addRow("Метрика:", self.ab_value_column)
        ab_layout.addRow("Контроль:", self.ab_control_value)
        ab_layout.addRow("Вариант:", self.ab_treatment_value)

        ab_button = QPushButton("Запустить A/B тест")
        self._register_icon(ab_button.setIcon, "ab-test")
        ab_button.clicked.connect(self.show_ab_test)
        ab_layout.addRow(ab_button)

        method_button, method_stack = _create_method_selector(
            [
                ("Описательная статистика", stats_group),
                ("Корреляция", corr_group),
                ("A/B тест", ab_group),
            ]
        )
        controls_layout.addWidget(method_button)
        controls_layout.addWidget(method_stack)
        controls_layout.addStretch(1)

        self.analysis_output.setReadOnly(True)
        self.analysis_output.setPlaceholderText("Загрузите данные и выберите действие.")

        layout.addWidget(controls)
        layout.addWidget(self.analysis_output, stretch=1)
        return page

    def _create_visualization_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        controls = QFrame()
        controls.setObjectName("Panel")
        controls.setMinimumWidth(340)
        controls_layout = QVBoxLayout(controls)

        controls_layout.addWidget(_section_label("Таблица для визуализации"))
        controls_layout.addWidget(self.viz_table)

        controls_layout.addWidget(_section_label("График"))
        for key, label in VISUALIZATION_METHODS.items():
            self.viz_method.addItem(label, key)
        controls_layout.addWidget(_field_label("Тип графика"))
        controls_layout.addWidget(self.viz_method)
        controls_layout.addWidget(_field_label("X / категории"))
        controls_layout.addWidget(self.viz_x_column)
        controls_layout.addWidget(_field_label("Y / значения"))
        controls_layout.addWidget(self.viz_y_column)
        self.viz_title.setPlaceholderText("Заголовок")
        controls_layout.addWidget(self.viz_title)

        render_button = QPushButton("Построить график")
        self._register_icon(render_button.setIcon, "analysis")
        render_button.clicked.connect(self.render_visualization)
        controls_layout.addWidget(render_button)

        controls_layout.addSpacing(12)
        controls_layout.addWidget(_section_label("Созданные графики"))
        add_visualization_button = QPushButton("Добавить текущий график")
        add_visualization_button.clicked.connect(self.add_saved_visualization)
        controls_layout.addWidget(add_visualization_button)
        self.saved_visualization_items.setMaximumHeight(120)
        controls_layout.addWidget(self.saved_visualization_items)
        save_png_button = QPushButton("Сохранить выбранный PNG")
        save_png_button.clicked.connect(self.save_selected_visualization_png)
        controls_layout.addWidget(save_png_button)
        controls_layout.addStretch(1)

        plot_panel = QFrame()
        plot_panel.setObjectName("Panel")
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.addWidget(_section_label("Предпросмотр"))
        plot_layout.addWidget(self.viz_plot, stretch=1)

        layout.addWidget(controls)
        layout.addWidget(plot_panel, stretch=1)
        return page

    def _create_ml_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        controls = QFrame()
        controls.setObjectName("Panel")
        controls.setMinimumWidth(340)
        controls_layout = QVBoxLayout(controls)

        controls_layout.addWidget(_section_label("Таблица для обучения"))
        controls_layout.addWidget(self.ml_table)

        controls_layout.addWidget(_section_label("Конструктор модели"))
        for key, label in model_options().items():
            self.ml_model.addItem(label, key)
        controls_layout.addWidget(_field_label("Алгоритм"))
        controls_layout.addWidget(self.ml_model)
        self._build_ml_parameter_pages()
        controls_layout.addWidget(_section_label("Параметры модели"))
        controls_layout.addWidget(self.ml_parameter_stack)

        controls_layout.addWidget(_field_label("Признаки"))
        self.ml_feature_columns.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        controls_layout.addWidget(self.ml_feature_columns)

        controls_layout.addWidget(_field_label("Целевой столбец"))
        controls_layout.addWidget(self.ml_target_column)

        controls_layout.addWidget(_section_label("Метод машинного обучения"))
        training_group = QGroupBox("Обучение модели")
        training_layout = QFormLayout(training_group)
        self.ml_test_size.setRange(0.1, 0.5)
        self.ml_test_size.setSingleStep(0.05)
        self.ml_test_size.setDecimals(2)
        self.ml_test_size.setValue(0.25)
        self.ml_random_state.setRange(0, 1_000_000)
        self.ml_random_state.setValue(42)
        training_layout.addRow("Доля теста:", self.ml_test_size)
        training_layout.addRow("Random state:", self.ml_random_state)
        train_button = QPushButton("Обучить модель")
        self._register_icon(train_button.setIcon, "train")
        train_button.clicked.connect(self.train_selected_model)
        training_layout.addRow(train_button)

        validation_group = QGroupBox("Кросс-валидация")
        validation_layout = QFormLayout(validation_group)
        self.ml_cv_folds.setRange(2, 20)
        self.ml_cv_folds.setValue(5)
        validation_layout.addRow("Количество фолдов:", self.ml_cv_folds)
        validate_button = QPushButton("Запустить кросс-валидацию")
        self._register_icon(validate_button.setIcon, "stats")
        validate_button.clicked.connect(self.validate_selected_model)
        validation_layout.addRow(validate_button)

        method_button, method_stack = _create_method_selector(
            [
                ("Обучение модели", training_group),
                ("Кросс-валидация", validation_group),
            ]
        )
        controls_layout.addWidget(method_button)
        controls_layout.addWidget(method_stack)
        controls_layout.addStretch(1)

        self.ml_output.setReadOnly(True)
        self.ml_output.setPlaceholderText("Загрузите данные, выберите признаки и целевой столбец.")

        layout.addWidget(controls)
        layout.addWidget(self.ml_output, stretch=1)
        return page

    def _connect_signals(self) -> None:
        self.tables_list.currentRowChanged.connect(self._on_active_table_changed)
        self.preview_selector.currentIndexChanged.connect(self._update_preview_from_selector)
        self.prep_source_table.currentIndexChanged.connect(self._refresh_prep_columns)
        self.analysis_table.currentIndexChanged.connect(self._refresh_analysis_columns)
        self.ml_table.currentIndexChanged.connect(self._refresh_ml_columns)
        self.ml_model.currentIndexChanged.connect(self._refresh_ml_parameter_page)
        self.ab_group_column.currentTextChanged.connect(self._refresh_ab_values)
        self.viz_table.currentIndexChanged.connect(self._refresh_visualization_columns)
        self.viz_method.currentIndexChanged.connect(self._refresh_visualization_defaults)
        self.saved_visualization_items.currentRowChanged.connect(self.preview_saved_visualization)

    # ----- Theme & icons -----------------------------------------------

    def _register_icon(self, setter: Callable[[QIcon], None], asset_name: str) -> None:
        self._icon_targets.append((setter, asset_name))

    def _icon_color(self) -> str:
        return "#ffffff" if self.is_dark_theme else "#000000"

    def _apply_icons(self) -> None:
        color = self._icon_color()
        for setter, asset_name in self._icon_targets:
            setter(themed_svg_icon(asset_name, color))
        for index, asset_name in self._tab_icon_targets:
            self.tabs.setTabIcon(index, themed_svg_icon(asset_name, color))

    def toggle_theme(self) -> None:
        self.is_dark_theme = not self.is_dark_theme
        self.apply_theme()

    def apply_theme(self) -> None:
        if self.is_dark_theme:
            self.setStyleSheet(_dark_stylesheet())
        else:
            self.setStyleSheet(_light_stylesheet())
        self._apply_icons()

    # ----- Data loading -------------------------------------------------

    def load_from_files(self) -> None:
        filter_string = (
            "Табличные данные ("
            + " ".join(f"*{ext}" for ext in SUPPORTED_FILE_EXTENSIONS)
            + ");;CSV (*.csv);;Excel (*.xlsx *.xls);;Текст (*.txt);;Parquet (*.parquet)"
        )
        file_names, _ = QFileDialog.getOpenFileNames(self, "Открыть файлы", "", filter_string)
        if not file_names:
            return

        loaded: list[str] = []
        for file_name in file_names:
            path = Path(file_name)
            try:
                frame = load_table_from_file(path)
            except Exception as exc:
                self.show_error(f"Не удалось загрузить файл {path.name}", exc)
                continue
            name = self._make_unique_table_name(path.stem)
            self._add_table(LoadedTable(name=name, frame=frame))
            loaded.append(name)
        if loaded:
            self._log_preprocessing("Загружены таблицы: " + ", ".join(loaded))

    def load_from_database(self) -> None:
        connection, ok = QInputDialog.getText(
            self,
            "Подключение к базе данных",
            "Строка подключения SQLAlchemy (например, sqlite:///data.db):",
        )
        if not ok or not connection.strip():
            return

        try:
            tables = list_database_tables(connection.strip())
        except Exception as exc:
            self.show_error("Не удалось получить список таблиц", exc)
            return

        if not tables:
            QMessageBox.information(self, "База данных", "В базе данных нет таблиц.")
            return

        dialog = TableSelectionDialog(tables, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_tables()
        if not selected:
            return

        loaded: list[str] = []
        for table_name in selected:
            try:
                frame = load_database_table(connection.strip(), table_name)
            except Exception as exc:
                self.show_error(f"Не удалось загрузить таблицу {table_name}", exc)
                continue
            unique = self._make_unique_table_name(table_name)
            self._add_table(LoadedTable(name=unique, frame=frame))
            loaded.append(unique)
        if loaded:
            self._log_preprocessing("Из БД загружены таблицы: " + ", ".join(loaded))

    def remove_selected_table(self) -> None:
        index = self.tables_list.currentRow()
        if index < 0 or index >= len(self.tables):
            return
        removed = self.tables.pop(index)
        self._refresh_table_dependent_widgets()
        self._log_preprocessing(f"Удалена таблица: {removed.name}")

    def _add_table(self, table: LoadedTable) -> None:
        self.tables.append(table)
        self._refresh_table_dependent_widgets()
        self.tables_list.setCurrentRow(len(self.tables) - 1)

    def _make_unique_table_name(self, base_name: str) -> str:
        clean = base_name.strip() or "table"
        existing = {table.name for table in self.tables}
        if clean not in existing:
            return clean
        suffix = 2
        while f"{clean}_{suffix}" in existing:
            suffix += 1
        return f"{clean}_{suffix}"

    def _refresh_table_dependent_widgets(self) -> None:
        names = [table.name for table in self.tables]
        previous = self.tables_list.currentRow()
        self.tables_list.blockSignals(True)
        self.tables_list.clear()
        for table in self.tables:
            label = f"{table.name} ({table.frame.shape[0]} × {table.frame.shape[1]})"
            self.tables_list.addItem(QListWidgetItem(label))
        if names:
            self.tables_list.setCurrentRow(min(max(previous, 0), len(names) - 1))
        self.tables_list.blockSignals(False)

        self._refresh_table_summary()
        self._refresh_combo(self.preview_selector, names, preserve=True)
        self._refresh_combo(self.prep_source_table, names, preserve=True)
        self._refresh_combo(self.analysis_table, names, preserve=True)
        self._refresh_combo(self.ml_table, names, preserve=True)
        self._refresh_combo(self.viz_table, names, preserve=True)
        self._refresh_combo(self.join_left, names, preserve=True)
        self._refresh_combo(self.join_right, names, preserve=True)

        self._update_preview_from_selector()
        self._refresh_prep_columns()
        self._refresh_analysis_columns()
        self._refresh_ml_columns()
        self._refresh_visualization_columns()

    def _refresh_table_summary(self) -> None:
        if not self.tables:
            self.table_summary.setText("Таблицы пока не загружены.")
            return
        lines = [
            f"• {table.name}: {table.frame.shape[0]} строк, {table.frame.shape[1]} столбцов"
            for table in self.tables
        ]
        self.table_summary.setText("\n".join(lines))

    def _refresh_combo(self, combo: QComboBox, items: list[str], *, preserve: bool) -> None:
        current = combo.currentText() if preserve else ""
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        if preserve and current in items:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _on_active_table_changed(self, row: int) -> None:
        if 0 <= row < len(self.tables):
            self.preview_selector.setCurrentIndex(row)

    def _update_preview_from_selector(self) -> None:
        index = self.preview_selector.currentIndex()
        if index < 0 or index >= len(self.tables):
            self.preview_table.clear()
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            return
        table = self.tables[index]
        preview = table.frame.head(PREVIEW_ROWS)
        self._populate_preview_table(preview)

    def _populate_preview_table(self, frame: pd.DataFrame) -> None:
        self.preview_table.clear()
        columns = [str(column) for column in frame.columns]
        self.preview_table.setColumnCount(len(columns))
        self.preview_table.setRowCount(len(frame))
        self.preview_table.setHorizontalHeaderLabels(columns)
        for row_index, (_, row) in enumerate(frame.iterrows()):
            for column_index, column in enumerate(columns):
                value = row[column]
                text = "" if pd.isna(value) else str(value)
                self.preview_table.setItem(row_index, column_index, QTableWidgetItem(text))
        self.preview_table.resizeColumnsToContents()

    # ----- Preprocessing -----------------------------------------------

    def _selected_table(self, combo: QComboBox) -> LoadedTable | None:
        name = combo.currentText()
        return self._table_by_name(name)

    def _table_by_name(self, name: str) -> LoadedTable | None:
        for table in self.tables:
            if table.name == name:
                return table
        return None

    def _refresh_prep_columns(self) -> None:
        table = self._selected_table(self.prep_source_table)
        columns = [str(column) for column in table.frame.columns] if table else []
        self._populate_list(self.prep_columns, columns, select_all=True)
        self._populate_list(self.pivot_index, columns)
        self._populate_list(self.pivot_columns, columns)
        self._populate_list(self.pivot_values, columns)
        self._populate_list(self.group_by, columns)
        self._refresh_dtype_table()

    def _populate_list(self, widget: QListWidget, items: list[str], *, select_all: bool = False) -> None:
        widget.clear()
        for item in items:
            list_item = QListWidgetItem(item)
            widget.addItem(list_item)
            if select_all:
                list_item.setSelected(True)

    def _log_preprocessing(self, message: str) -> None:
        self.prep_log.append(message)

    def _refresh_dtype_table(self) -> None:
        table = self._selected_table(self.prep_source_table)
        if table is None:
            self.dtype_table.setRowCount(0)
            return

        columns = [str(column) for column in table.frame.columns]
        self.dtype_table.setRowCount(len(columns))
        for row, column in enumerate(columns):
            current_dtype = str(table.frame[column].dtype)
            self.dtype_table.setItem(row, 0, QTableWidgetItem(column))
            self.dtype_table.setItem(row, 1, QTableWidgetItem(current_dtype))

            dtype_selector = QComboBox()
            dtype_selector.addItems(list(TYPE_CONVERSION_OPTIONS))
            dtype_selector.setCurrentText(_dtype_option_for_series(table.frame[column]))
            self.dtype_table.setCellWidget(row, 2, dtype_selector)
        self.dtype_table.resizeRowsToContents()

    def apply_missing_strategy(self) -> None:
        table = self._selected_table(self.prep_source_table)
        if table is None:
            self.show_error("Нет данных", RuntimeError("Загрузите таблицу перед обработкой пропусков."))
            return
        strategy = self.prep_strategy.currentText()
        columns = self._list_selected_items(self.prep_columns) or None
        try:
            if strategy == "drop":
                new_frame = drop_missing(table.frame, columns)
            else:
                new_frame = impute_missing(
                    table.frame,
                    strategy=strategy,
                    columns=columns,
                    knn_neighbors=self.prep_knn_neighbors.value(),
                )
        except Exception as exc:
            self.show_error("Не удалось обработать пропуски", exc)
            return
        self._publish_result_table(f"{table.name}_{strategy}", new_frame, "пропуски")

    def apply_type_conversions(self) -> None:
        table = self._selected_table(self.prep_source_table)
        if table is None:
            self.show_error("Нет данных", RuntimeError("Выберите таблицу для преобразования типов."))
            return

        conversions: dict[str, str] = {}
        for row in range(self.dtype_table.rowCount()):
            column_item = self.dtype_table.item(row, 0)
            dtype_selector = self.dtype_table.cellWidget(row, 2)
            if column_item is None or not isinstance(dtype_selector, QComboBox):
                continue
            conversions[column_item.text()] = dtype_selector.currentText()

        try:
            new_frame = convert_column_types(table.frame, conversions)
        except Exception as exc:
            self.show_error("Не удалось преобразовать типы", exc)
            return
        self._publish_result_table(f"{table.name}_types", new_frame, "преобразование типов")

    def apply_pivot_table(self) -> None:
        table = self._selected_table(self.prep_source_table)
        if table is None:
            self.show_error("Нет данных", RuntimeError("Выберите исходную таблицу для сводной."))
            return
        index = self._list_selected_items(self.pivot_index)
        if not index:
            self.show_error("Нет данных", RuntimeError("Выберите хотя бы один столбец для индекса."))
            return
        columns = self._list_selected_items(self.pivot_columns) or None
        values = self._list_selected_items(self.pivot_values) or None
        try:
            new_frame = pivot_table(
                table.frame,
                index=index,
                columns=columns,
                values=values,
                aggfunc=self.pivot_aggfunc.currentText(),
            )
        except Exception as exc:
            self.show_error("Не удалось построить сводную таблицу", exc)
            return
        self._publish_result_table(f"{table.name}_pivot", new_frame, "сводная таблица")

    def apply_join(self) -> None:
        left = self._selected_table(self.join_left)
        right = self._selected_table(self.join_right)
        if left is None or right is None:
            self.show_error("Нет данных", RuntimeError("Загрузите минимум две таблицы для объединения."))
            return
        method = self.join_method.currentText()
        on_text = self.join_on.text().strip()
        try:
            on = [token.strip() for token in on_text.split(",") if token.strip()] if on_text else None
            if method == "cross":
                new_frame = join_tables(left.frame, right.frame, how="cross")
            else:
                if not on:
                    raise ValueError("Укажите ключи объединения (через запятую).")
                new_frame = join_tables(left.frame, right.frame, how=method, on=on)
        except Exception as exc:
            self.show_error("Не удалось объединить таблицы", exc)
            return
        self._publish_result_table(f"{left.name}_{method}_{right.name}", new_frame, "объединение")

    def apply_group(self) -> None:
        table = self._selected_table(self.prep_source_table)
        if table is None:
            self.show_error("Нет данных", RuntimeError("Выберите таблицу для группировки."))
            return
        by = self._list_selected_items(self.group_by)
        if not by:
            self.show_error("Нет данных", RuntimeError("Выберите хотя бы один столбец для группировки."))
            return
        try:
            new_frame = group_table(table.frame, by=by, aggfunc=self.group_aggfunc.currentText())
        except Exception as exc:
            self.show_error("Не удалось сгруппировать таблицу", exc)
            return
        self._publish_result_table(f"{table.name}_group", new_frame, "группировка")

    def _publish_result_table(self, default_name: str, frame: pd.DataFrame, action: str) -> None:
        requested = self.prep_output_name.text().strip()
        target_name = requested or default_name
        unique = self._make_unique_table_name(target_name)
        self._add_table(LoadedTable(name=unique, frame=frame))
        self.prep_output_name.clear()
        self._log_preprocessing(
            f"{action}: создана таблица '{unique}' ({frame.shape[0]} строк × {frame.shape[1]} столбцов)."
        )

    # ----- Analysis ----------------------------------------------------

    def _refresh_analysis_columns(self) -> None:
        table = self._selected_table(self.analysis_table)
        columns = [str(column) for column in table.frame.columns] if table else []
        self._populate_list(self.analysis_columns, columns)
        if table is not None:
            numeric = {
                str(column)
                for column in table.frame.select_dtypes(include="number").columns
            }
            for index in range(self.analysis_columns.count()):
                item = self.analysis_columns.item(index)
                item.setSelected(item.text() in numeric)
        for combo in (self.ab_group_column, self.ab_value_column):
            self._refresh_combo(combo, columns, preserve=False)
        if table is not None:
            numeric_columns = [
                str(column) for column in table.frame.select_dtypes(include="number").columns
            ]
            if numeric_columns:
                self.ab_value_column.setCurrentText(numeric_columns[0])
        self._refresh_ab_values()

    def _refresh_ab_values(self) -> None:
        table = self._selected_table(self.analysis_table)
        group_column = self.ab_group_column.currentText()
        if table is None or not group_column or group_column not in table.frame.columns:
            self.ab_control_value.clear()
            self.ab_treatment_value.clear()
            return
        values = sorted(table.frame[group_column].dropna().astype(str).unique().tolist())
        self.ab_control_value.clear()
        self.ab_treatment_value.clear()
        self.ab_control_value.addItems(values)
        self.ab_treatment_value.addItems(values)
        if len(values) > 1:
            self.ab_treatment_value.setCurrentIndex(1)

    def select_numeric_analysis_columns(self) -> None:
        table = self._selected_table(self.analysis_table)
        if table is None:
            return
        numeric = {str(column) for column in table.frame.select_dtypes(include="number").columns}
        for index in range(self.analysis_columns.count()):
            item = self.analysis_columns.item(index)
            item.setSelected(item.text() in numeric)

    def show_descriptive_statistics(self) -> None:
        table = self._require_analysis_table()
        if table is None:
            return
        columns = self._list_selected_items(self.analysis_columns)
        try:
            overview = dataset_overview(table.frame)
            summary = describe_columns(table.frame, columns)
            summary_frame = pd.DataFrame(summary).T
            header = (
                f"Таблица: {table.name}\n"
                f"Строк: {overview.row_count}, столбцов: {overview.column_count}, "
                f"пропусков: {sum(overview.missing_by_column.values())}\n\n"
            )
            self.analysis_output.setText(header + summary_frame.round(4).to_string())
        except Exception as exc:
            self.show_error("Не удалось посчитать статистику", exc)

    def show_correlation(self) -> None:
        table = self._require_analysis_table()
        if table is None:
            return
        columns = self._list_selected_items(self.analysis_columns)
        try:
            matrix = correlation_matrix(
                table.frame, columns, method=self.correlation_method.currentText()
            )
            self.analysis_output.setText(matrix.round(4).to_string())
        except Exception as exc:
            self.show_error("Не удалось построить корреляцию", exc)

    def show_ab_test(self) -> None:
        table = self._require_analysis_table()
        if table is None:
            return
        try:
            result = run_ab_test(
                table.frame,
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
        except Exception as exc:
            self.show_error("Не удалось выполнить A/B тест", exc)

    # ----- Visualization ----------------------------------------------

    def _refresh_visualization_columns(self) -> None:
        table = self._selected_table(self.viz_table)
        columns = [str(column) for column in table.frame.columns] if table else []
        self._refresh_combo(self.viz_x_column, columns, preserve=False)
        self._refresh_combo(self.viz_y_column, columns, preserve=False)
        self._refresh_visualization_defaults()

    def _refresh_visualization_defaults(self) -> None:
        table = self._selected_table(self.viz_table)
        if table is None:
            return
        columns = [str(column) for column in table.frame.columns]
        numeric = [str(column) for column in table.frame.select_dtypes(include="number").columns]
        chart_type = self.viz_method.currentData()
        if chart_type in {"line", "bar", "scatter"}:
            if columns:
                self.viz_x_column.setCurrentText(columns[0])
            if numeric:
                self.viz_y_column.setCurrentText(numeric[0])
        elif chart_type == "pie":
            categorical = [column for column in columns if column not in numeric]
            if categorical:
                self.viz_x_column.setCurrentText(categorical[0])
            if numeric:
                self.viz_y_column.setCurrentText(numeric[0])
        elif chart_type in {"histogram", "box"} and numeric:
            self.viz_x_column.setCurrentText(numeric[0])
            self.viz_y_column.setCurrentText(numeric[0])

    def _current_visualization_config(self) -> VisualizationConfig:
        chart_type = self.viz_method.currentData()
        title = self.viz_title.text().strip() or None
        if chart_type == "heatmap":
            return VisualizationConfig(chart_type=chart_type, title=title)
        if chart_type == "box":
            return VisualizationConfig(
                chart_type=chart_type,
                y_column=self.viz_y_column.currentText() or self.viz_x_column.currentText(),
                title=title,
            )
        return VisualizationConfig(
            chart_type=chart_type,
            x_column=self.viz_x_column.currentText() or None,
            y_column=self.viz_y_column.currentText() or None,
            title=title,
        )

    def render_visualization(self) -> None:
        table = self._selected_table(self.viz_table)
        if table is None:
            self.show_error("Нет данных", RuntimeError("Загрузите хотя бы одну таблицу."))
            return
        config = self._current_visualization_config()
        try:
            validate_chart_config(table.frame, config)
            self._draw_visualization(table.frame, config)
        except Exception as exc:
            self.show_error("Не удалось построить график", exc)

    def add_saved_visualization(self) -> None:
        table = self._selected_table(self.viz_table)
        if table is None:
            self.show_error("Нет данных", RuntimeError("Загрузите хотя бы одну таблицу."))
            return
        config = self._current_visualization_config()
        try:
            validate_chart_config(table.frame, config)
            self._draw_visualization(table.frame, config)
        except Exception as exc:
            self.show_error("Не удалось добавить график", exc)
            return
        self._saved_visualizations.append((table.name, config))
        label = config.title or VISUALIZATION_METHODS[config.chart_type]
        index = len(self._saved_visualizations)
        self.saved_visualization_items.addItem(f"{index}. {table.name}: {label}")
        self.saved_visualization_items.setCurrentRow(index - 1)

    def preview_saved_visualization(self, row: int) -> None:
        if row < 0 or row >= len(self._saved_visualizations):
            return
        table_name, config = self._saved_visualizations[row]
        table = self._table_by_name(table_name)
        if table is None:
            self.show_error("Нет данных", RuntimeError(f"Таблица '{table_name}' больше недоступна."))
            return
        try:
            self._draw_visualization(table.frame, config)
        except Exception as exc:
            self.show_error("Не удалось показать график", exc)

    def save_selected_visualization_png(self) -> None:
        selected = self._selected_saved_visualization()
        if selected is None:
            self.show_error("Нет графика", RuntimeError("Добавьте график и выберите его в списке."))
            return
        table, config = selected
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить график PNG",
            _png_file_name(config),
            "PNG изображение (*.png)",
        )
        if not file_name:
            return
        try:
            saved_path = save_visualization_png(table.frame, config, file_name)
        except Exception as exc:
            self.show_error("Не удалось сохранить PNG", exc)
            return
        QMessageBox.information(
            self,
            "PNG сохранён",
            f"График сохранён:\n{saved_path}",
        )

    def _selected_saved_visualization(self) -> tuple[LoadedTable, VisualizationConfig] | None:
        row = self.saved_visualization_items.currentRow()
        if row < 0 or row >= len(self._saved_visualizations):
            return None
        table_name, config = self._saved_visualizations[row]
        table = self._table_by_name(table_name)
        if table is None:
            return None
        return table, config

    def _draw_visualization(self, frame: pd.DataFrame, config: VisualizationConfig) -> None:
        render_matplotlib_figure(frame, config, self.viz_plot.figure)
        self.viz_plot.draw()

    # ----- Machine learning --------------------------------------------

    def _build_ml_parameter_pages(self) -> None:
        for key in model_options():
            page = QWidget()
            form = QFormLayout(page)
            self.ml_parameter_widgets[key] = {}
            for spec in model_parameter_specs(key):
                editor = _parameter_editor(spec)
                form.addRow(f"{spec['label']}:", editor)
                self.ml_parameter_widgets[key][spec["name"]] = (spec, editor)
            self.ml_parameter_stack.addWidget(page)
        self._refresh_ml_parameter_page()

    def _refresh_ml_parameter_page(self) -> None:
        index = max(0, self.ml_model.currentIndex())
        self.ml_parameter_stack.setCurrentIndex(index)

    def _refresh_ml_columns(self) -> None:
        table = self._selected_table(self.ml_table)
        columns = [str(column) for column in table.frame.columns] if table else []
        self._populate_list(self.ml_feature_columns, columns)
        self._refresh_combo(self.ml_target_column, columns, preserve=False)
        if columns:
            for index in range(self.ml_feature_columns.count()):
                item = self.ml_feature_columns.item(index)
                if item.text() != columns[-1]:
                    item.setSelected(True)
            self.ml_target_column.setCurrentText(columns[-1])

    def _selected_ml_model_params(self) -> dict[str, Any]:
        model_name = self.ml_model.currentData()
        widgets = self.ml_parameter_widgets.get(model_name, {})
        params: dict[str, Any] = {}
        for name, (spec, editor) in widgets.items():
            params[name] = _parameter_value(spec, editor)
        return params

    def train_selected_model(self) -> None:
        table = self._require_ml_table()
        if table is None:
            return
        target = self.ml_target_column.currentText()
        features = [column for column in self._list_selected_items(self.ml_feature_columns) if column != target]
        try:
            result = train_model(
                table.frame,
                model_name=self.ml_model.currentData(),
                feature_columns=features,
                target_column=target,
                random_state=self.ml_random_state.value(),
                test_size=self.ml_test_size.value(),
                model_params=self._selected_ml_model_params(),
            )
            self.ml_output.setText(
                "\n".join(
                    [
                        f"Модель: {model_options()[result.model_name]}",
                        f"Таблица: {table.name}",
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

    def validate_selected_model(self) -> None:
        table = self._require_ml_table()
        if table is None:
            return
        target = self.ml_target_column.currentText()
        features = [column for column in self._list_selected_items(self.ml_feature_columns) if column != target]
        try:
            result = cross_validate_model(
                table.frame,
                model_name=self.ml_model.currentData(),
                feature_columns=features,
                target_column=target,
                cv_folds=self.ml_cv_folds.value(),
                random_state=self.ml_random_state.value(),
                model_params=self._selected_ml_model_params(),
            )
            self.ml_output.setText(
                "\n".join(
                    [
                        f"Кросс-валидация: {model_options()[result.model_name]}",
                        f"Таблица: {table.name}",
                        f"Целевой столбец: {result.target_column}",
                        f"Метрика: {result.metric}",
                        f"Фолдов: {result.fold_count}",
                        f"Средний score: {result.mean_score:.4f}",
                        f"Стандартное отклонение: {result.std_score:.4f}",
                        "",
                        "Scores по фолдам:",
                        ", ".join(f"{score:.4f}" for score in result.fold_scores),
                        "",
                        "Использованные признаки:",
                        ", ".join(result.feature_names),
                    ]
                )
            )
        except Exception as exc:
            self.show_error("Не удалось выполнить кросс-валидацию", exc)

    # ----- Helpers -----------------------------------------------------

    def _require_analysis_table(self) -> LoadedTable | None:
        table = self._selected_table(self.analysis_table)
        if table is None:
            self.show_error("Нет данных", RuntimeError("Загрузите хотя бы одну таблицу."))
        return table

    def _require_ml_table(self) -> LoadedTable | None:
        table = self._selected_table(self.ml_table)
        if table is None:
            self.show_error("Нет данных", RuntimeError("Загрузите хотя бы одну таблицу."))
        return table

    def _list_selected_items(self, widget: QListWidget) -> list[str]:
        return [item.text() for item in widget.selectedItems()]

    def show_error(self, title: str, error: Exception) -> None:
        QMessageBox.warning(self, title, str(error))


class TableSelectionDialog(QDialog):
    def __init__(self, tables: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Выбор таблиц")
        self._checkboxes: list[QCheckBox] = []
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Отметьте таблицы для загрузки:"))
        for name in tables:
            box = QCheckBox(name, self)
            self._checkboxes.append(box)
            layout.addWidget(box)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_tables(self) -> list[str]:
        return [box.text() for box in self._checkboxes if box.isChecked()]


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("SectionTitle")
    return label


def _field_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("MutedText")
    return label


def _png_file_name(config: VisualizationConfig) -> str:
    label = config.title or VISUALIZATION_METHODS[config.chart_type]
    safe = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in label)
    safe = safe.strip("_")
    return f"{safe or 'visualization'}.png"


def _create_method_selector(entries: list[tuple[str, QWidget]]) -> tuple[QToolButton, QStackedWidget]:
    button = QToolButton()
    button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    button.setMinimumHeight(36)
    menu = QMenu(button)
    stack = QStackedWidget()

    def select_method(index: int, title: str) -> None:
        stack.setCurrentIndex(index)
        button.setText(title)

    for index, (title, widget) in enumerate(entries):
        stack.addWidget(widget)
        action = menu.addAction(title)
        action.triggered.connect(
            lambda _checked=False, selected=index, name=title: select_method(selected, name)
        )
    button.setMenu(menu)
    if entries:
        select_method(0, entries[0][0])
    return button, stack


def _dtype_option_for_series(series: pd.Series) -> str:
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return "category"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    return "string"


def _parameter_editor(spec: dict[str, Any]) -> QWidget:
    kind = spec["type"]
    if kind == "choice":
        editor = QComboBox()
        for option in spec["options"]:
            label = "None" if option is None else str(option)
            editor.addItem(label, option)
        default_index = editor.findData(spec["default"])
        if default_index >= 0:
            editor.setCurrentIndex(default_index)
        return editor
    if kind in {"int", "optional_int"}:
        editor = QSpinBox()
        editor.setRange(int(spec["min"]), int(spec["max"]))
        editor.setValue(0 if spec["default"] is None else int(spec["default"]))
        return editor
    if kind == "float":
        editor = QDoubleSpinBox()
        editor.setRange(float(spec["min"]), float(spec["max"]))
        editor.setSingleStep(float(spec["step"]))
        editor.setDecimals(3)
        editor.setValue(float(spec["default"]))
        return editor
    if kind == "bool":
        editor = QCheckBox()
        editor.setChecked(bool(spec["default"]))
        return editor
    raise ValueError(f"Unsupported parameter editor type: {kind}")


def _parameter_value(spec: dict[str, Any], editor: QWidget) -> Any:
    kind = spec["type"]
    if kind == "choice" and isinstance(editor, QComboBox):
        return editor.currentData()
    if kind == "optional_int" and isinstance(editor, QSpinBox):
        value = editor.value()
        return None if value == 0 else value
    if kind == "int" and isinstance(editor, QSpinBox):
        return editor.value()
    if kind == "float" and isinstance(editor, QDoubleSpinBox):
        return editor.value()
    if kind == "bool" and isinstance(editor, QCheckBox):
        return editor.isChecked()
    raise TypeError(f"Unsupported parameter widget for {spec['name']}")


def _dark_stylesheet() -> str:
    return """
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
        QFrame#Hero, QFrame#Panel, QTextEdit, QTableWidget, QListWidget, QComboBox,
        QLineEdit, QSpinBox, QGroupBox {
            background: #20252b;
            border: 1px solid #39424e;
            border-radius: 8px;
        }
        QGroupBox {
            margin-top: 18px;
            padding: 14px 10px 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
            color: #edf2f7;
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
        QLabel#BodyText {
            color: #d8dee9;
        }
        QPushButton, QToolButton {
            background: #000000;
            color: #ffffff;
            border: 0;
            border-radius: 6px;
            padding: 9px 12px;
            min-height: 18px;
            font-weight: 600;
        }
        QPushButton:hover, QToolButton:hover {
            background: #171717;
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
            background: #000000;
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
            selection-background-color: #000000;
            selection-color: #ffffff;
        }
        QTableWidget::item {
            color: #edf2f7;
        }
    """


def _light_stylesheet() -> str:
    return """
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
        QFrame#Hero, QFrame#Panel, QTextEdit, QTableWidget, QListWidget, QComboBox,
        QLineEdit, QSpinBox, QGroupBox {
            background: #ffffff;
            border: 1px solid #d6dde5;
            border-radius: 8px;
        }
        QGroupBox {
            margin-top: 18px;
            padding: 14px 10px 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
            color: #1d2630;
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
        QLabel#BodyText {
            color: #1d2630;
        }
        QPushButton, QToolButton {
            background: #000000;
            color: #ffffff;
            border: 0;
            border-radius: 6px;
            padding: 9px 12px;
            min-height: 18px;
            font-weight: 600;
        }
        QPushButton:hover, QToolButton:hover {
            background: #171717;
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
            background: #000000;
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
            selection-background-color: #000000;
            selection-color: #ffffff;
        }
    """


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    window = FenrirMiningWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
