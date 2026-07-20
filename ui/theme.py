"""Dark theme stylesheet (Qt Style Sheets) for the application."""
from __future__ import annotations

DARK_STYLESHEET = """
* {
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
    font-size: 13px;
    color: #e6e6e6;
}
QMainWindow, QWidget {
    background-color: #1e1f22;
}
QLabel#VideoLabel {
    background-color: #101114;
    border: 1px solid #33343a;
    border-radius: 8px;
}
QGroupBox {
    background-color: #26272b;
    border: 1px solid #33343a;
    border-radius: 8px;
    margin-top: 14px;
    padding: 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: #8ab4f8;
}
QPushButton {
    background-color: #2f80ed;
    border: none;
    border-radius: 6px;
    padding: 8px 12px;
    color: #ffffff;
    font-weight: 600;
}
QPushButton:hover { background-color: #4a90f0; }
QPushButton:pressed { background-color: #1c66c9; }
QPushButton:disabled { background-color: #3a3b40; color: #7a7b80; }
QPushButton#StopButton { background-color: #e05c5c; }
QPushButton#StopButton:hover { background-color: #f07070; }
QPushButton#StopButton:disabled { background-color: #3a3b40; color: #7a7b80; }
QPlainTextEdit, QTextEdit {
    background-color: #16171a;
    border: 1px solid #33343a;
    border-radius: 6px;
    font-family: 'Consolas', 'Menlo', monospace;
    font-size: 12px;
}
QComboBox {
    background-color: #26272b;
    border: 1px solid #33343a;
    border-radius: 6px;
    padding: 4px 8px;
}
QComboBox QAbstractItemView {
    background-color: #26272b;
    selection-background-color: #2f80ed;
}
QStatusBar { color: #9a9b9f; }
QScrollBar:vertical {
    background: #1e1f22; width: 10px; margin: 0;
}
QScrollBar::handle:vertical {
    background: #3a3b40; border-radius: 5px; min-height: 24px;
}
"""
