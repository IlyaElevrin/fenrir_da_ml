# Fenrir Mining

Fenrir Mining is a cross-platform desktop application for data analysis and basic machine learning. It loads CSV and Excel tables, previews the first five rows, offers statistical/correlation/A-B analysis, and trains starter scikit-learn models from selected columns.

## Возможности

- Главная вкладка: описание программы.
- Загрузка данных: чтение CSV, XLSX, TXT, Parquet файлов, подключение к базе данных (SQLite/SQLAlchemy URI), одновременная работа с несколькими таблицами и предпросмотр первых пяти строк выбранной таблицы.
- Предобработка данных: удаление пропусков или их заполнение средним, медианой или методом KNN, построение сводных таблиц, объединение таблиц различными join-методами и группировка с агрегациями.
- Анализ данных: выбор столбцов, описательная статистика, корреляции Pearson/Spearman/Kendall и Welch A/B test.
- Машинное обучение: конструкторы Decision Tree, Random Forest, Logistic Regression и Linear Regression.
- Переключение светлой и темной темы с автоматической перекраской SVG-иконок (чёрные на светлой теме, белые на тёмной).

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
