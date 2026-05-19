"""Headless screenshot helper for the Fenrir DA/ML desktop app."""
from __future__ import annotations

import os
import sys
import pathlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

import main as fenrir


def grab(window, path: pathlib.Path) -> None:
    pixmap = window.grab()
    path.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(str(path))
    print(f"saved {path}")


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = fenrir.FenrirMiningWindow()
    window.resize(1280, 800)
    window.show()
    app.processEvents()

    screenshots = ROOT / "docs" / "screenshots"

    tab_names = ("home", "loading", "preprocessing", "analysis", "visualization", "ml")
    for index, name in enumerate(tab_names):
        window.tabs.setCurrentIndex(index)
        app.processEvents()
        grab(window, screenshots / f"dark-{name}.png")

    window.toggle_theme()
    app.processEvents()
    for index, name in enumerate(tab_names):
        window.tabs.setCurrentIndex(index)
        app.processEvents()
        grab(window, screenshots / f"light-{name}.png")


if __name__ == "__main__":
    main()
