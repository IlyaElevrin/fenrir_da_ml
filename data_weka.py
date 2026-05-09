import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, 
                            QWidget, QLabel, QPushButton, QHBoxLayout, QFrame,
                            QFileDialog, QComboBox, QTextEdit, QScrollArea)
from PyQt6.QtCore import Qt, QUrl, QSize
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor
from PyQt6.QtSvgWidgets import QSvgWidget
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns

class SVGIcon(QWidget):
    def __init__(self, svg_path, size=32, parent=None):
        super().__init__(parent)
        self.svg = QSvgWidget(svg_path)
        self.svg.setFixedSize(size, size)


def create_sample_svg():
    pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeoWEKA — Data Analysis & ML Studio")
        self.resize(1400, 900)
        
        self.is_dark = True
        self.apply_theme()
        
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setStyleSheet("QTabWidget::pane { border: 0; }")
        
        tabs.addTab(self.create_home_tab(), "🏠 Главная")
        tabs.addTab(self.create_analysis_tab(), "📊 Анализ данных")
        tabs.addTab(self.create_ml_tab(), "🤖 Машинное обучение")
        
        self.setCentralWidget(tabs)
        
        menu = self.menuBar()
        theme_action = menu.addAction("🌗 Переключить тему")
        theme_action.triggered.connect(self.toggle_theme)

    def apply_theme(self):
        if self.is_dark:
            self.setStyleSheet("""
                QMainWindow, QTabWidget, QWidget {
                    background: #1e1e2e;
                    color: #cdd6f4;
                }
                QTabBar::tab { background: #313244; padding: 12px; border-radius: 8px; }
                QTabBar::tab:selected { background: #89b4fa; color: #11111b; }
                QPushButton { background: #89b4fa; color: #11111b; border-radius: 8px; padding: 10px; }
                /* Blur эффект (Qt GraphicsEffect) */
            """)
        else:
            self.setStyleSheet("""
                QMainWindow, QTabWidget, QWidget { background: #f0f0f5; color: #222; }
                QTabBar::tab { background: #e0e0e8; }
                QTabBar::tab:selected { background: #1e3a8a; color: white; }
            """)

    def toggle_theme(self):
        self.is_dark = not self.is_dark
        self.apply_theme()

    def create_home_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        logo = QSvgWidget() 
        logo.setFixedSize(180, 180)
        layout.addWidget(logo)
        
        title = QLabel("NeoWEKA")
        title.setStyleSheet("font-size: 48px; font-weight: bold;")
        layout.addWidget(title)
        
        desc = QLabel("Открытый кросс-платформенный аналог WEKA\nс современным UI, blur-эффектами и поддержкой SVG")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)
        
        donate_frame = QFrame()
        donate_layout = QHBoxLayout(donate_frame)
        btn_crypto = QPushButton("💰 Поддержать (USDT / BTC)")
        btn_crypto.clicked.connect(lambda: print("Открыть QR/адреса"))
        donate_layout.addWidget(btn_crypto)
        layout.addWidget(donate_frame)
        
        return widget

    def create_analysis_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        btn_load = QPushButton("📂 Загрузить CSV / Excel")
        btn_load.clicked.connect(self.load_data)
        layout.addWidget(btn_load)
        
        self.data_info = QTextEdit()
        self.data_info.setReadOnly(True)
        layout.addWidget(self.data_info)
        
        hbox = QHBoxLayout()
        btn_stats = QPushButton("📈 Описательная статистика")
        btn_corr = QPushButton("🔗 Матрица корреляций")
        btn_ab = QPushButton("A/B-тестирование")
        btn_stats.clicked.connect(self.show_stats)
        btn_corr.clicked.connect(self.show_correlation)
        hbox.addWidget(btn_stats)
        hbox.addWidget(btn_corr)
        hbox.addWidget(btn_ab)
        layout.addLayout(hbox)
        
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.df = None
        return widget

    def load_data(self):
        file, _ = QFileDialog.getOpenFileName(self, "Открыть данные", "", "CSV (*.csv);;Excel (*.xlsx)")
        if file:
            self.df = pd.read_csv(file) if file.endswith('.csv') else pd.read_excel(file)
            self.data_info.setText(str(self.df.describe()))
            self.show_correlation()

    def show_stats(self):
        if self.df is None: return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.df.hist(ax=ax, figsize=(10,6))
        self.canvas.draw()

    def show_correlation(self):
        if self.df is None: return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        numeric_df = self.df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        self.canvas.draw()

    def create_ml_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        model_type = QComboBox()
        model_type.addItems(["Decision Tree", "Random Forest", "Linear Regression", "Logistic Regression"])
        layout.addWidget(QLabel("Выберите модель:"))
        layout.addWidget(model_type)
        
        btn_train = QPushButton("🚀 Обучить модель")
        btn_train.clicked.connect(lambda: self.train_model(model_type.currentText()))
        layout.addWidget(btn_train)
        
        self.ml_log = QTextEdit()
        layout.addWidget(self.ml_log)
        return widget

    def train_model(self, model_name):
        if self.df is None:
            self.ml_log.setText("Сначала загрузите данные!")
            return
        X = self.df.iloc[:, :-1] 
        y = self.df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        if model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_name == "Random Forest":
            model = RandomForestClassifier()
        elif model_name == "Linear Regression":
            model = LinearRegression()
        else:
            model = LogisticRegression()
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test) if hasattr(model, 'score') else "N/A"
        self.ml_log.setText(f"Модель {model_name} обучена!\nAccuracy/R²: {score:.4f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())