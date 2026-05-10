# Fenrir Mining

Fenrir Mining is a cross-platform desktop application for data analysis and basic machine learning. It loads CSV and Excel tables, previews the first five rows, offers statistical/correlation/A-B analysis, and trains starter scikit-learn models from selected columns.

## Возможности

- Главная вкладка: описание программы, загрузка CSV/XLSX/XLS, обзор структуры данных и первые пять строк таблицы.
- Анализ данных: выбор столбцов, описательная статистика, корреляции Pearson/Spearman/Kendall и Welch A/B test.
- Машинное обучение: конструкторы Decision Tree, Random Forest, Logistic Regression и Linear Regression.
- Переключение светлой и темной темы.
- SVG-иконки в интерфейсе.

## Структура

- `data_weka.py` - PyQt6 GUI launcher.
- `func/da/` - функции анализа данных.
- `func/ml/` - функции машинного обучения.
- `assets/svg/` - SVG-ресурсы интерфейса.
- `tests/` - модульные тесты для функций анализа и ML.

## Установка и запуск

```bash
python -m pip install -r requirements.txt
python data_weka.py
```

## Проверка

```bash
python -m unittest discover -s tests
```
