# Анализ Данных

Анализ Данных is a cross-platform desktop application for data analysis and basic machine learning. It loads CSV and Excel tables, previews the first five rows, offers statistical/correlation/A-B analysis, visualization with PNG export, and trains starter scikit-learn models from selected columns.

## Возможности

- Главная вкладка: название "Фенрир Анализ Данных" и логотип из `assets/logo/`.
- Загрузка данных: чтение CSV, XLSX, TXT, Parquet файлов, подключение к базе данных (SQLite/SQLAlchemy URI), одновременная работа с несколькими таблицами и предпросмотр первых пяти строк выбранной таблицы.
- Предобработка данных: всплывающее меню методов, удаление пропусков или их заполнение средним, медианой или методом KNN, преобразование типов данных по каждому столбцу, построение сводных таблиц, объединение таблиц различными join-методами и группировка с агрегациями.
- Анализ данных: всплывающее меню методов, выбор столбцов, описательная статистика, корреляции Pearson/Spearman/Kendall и Welch A/B test.
- Визуализация: line graph, bar chart, pie chart, heat map, scatter plot, histogram, box with mustache, выбор созданного графика и сохранение в PNG.
- Машинное обучение: всплывающее меню методов, конструкторы Decision Tree, Random Forest, Logistic Regression и Linear Regression, расширенные параметры моделей и кросс-валидация.
- Переключение светлой и темной темы с автоматической перекраской SVG-иконок (чёрные на светлой теме, белые на тёмной) и чёрным акцентным цветом интерфейса.

## Структура

- `main.py` - PyQt6 GUI launcher.
- `func/da/` - функции анализа данных.
- `func/ml/` - функции машинного обучения.
- `func/io/` - загрузка данных из файлов и баз.
- `func/preprocess/` - предобработка данных.
- `assets/svg/` - SVG-ресурсы интерфейса.
- `tests/` - модульные тесты для функций анализа, ML, загрузки и предобработки.

## Установка и запуск

```bash
python -m pip install -r requirements.txt
python main.py
```

## Проверка

```bash
python -m unittest discover -s tests
```
