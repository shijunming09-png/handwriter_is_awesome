"""Shared, restrained desktop theme."""

APP_STYLE = """
QMainWindow, QWidget { font-family: 'Microsoft YaHei UI', 'Segoe UI'; font-size: 10pt; color: #21324b; }
QMainWindow { background: #f3f6fa; }
QWidget#header { background: #142c4b; }
QLabel#brand { color: white; font-size: 19pt; font-weight: 700; }
QLabel#brandSub { color: #b7cae3; font-size: 9pt; }
QLabel#localBadge { color: #bde9d8; background: #214b5c; border-radius: 9px; padding: 5px 12px; font-weight: 600; }
QLabel#sectionTitle { font-weight: 700; font-size: 11pt; color: #203a5e; }
QLabel#projectTitle { font-size: 14pt; font-weight: 700; }
QLabel#muted, QLabel#statusText { color: #6e7f94; font-size: 9pt; }
QLabel#warning { color: #8c5a17; background: #fff4dc; border-radius: 5px; padding: 5px; font-size: 9pt; }
QLabel#emptyTitle { font-size: 20pt; color: #234568; font-weight: 700; }
QLabel#emptySub { color: #71849a; font-size: 11pt; }
QWidget#sidebar, QWidget#reviewPane { background: white; border: 1px solid #e1e8f1; border-radius: 9px; }
QWidget#lineCard { background: #fbfcfe; border: 1px solid #dfe7f0; border-radius: 7px; }
QWidget#lineCard[selected='true'] { background: #eef5ff; border: 2px solid #518be2; }
QPushButton { background: white; border: 1px solid #cdd9e7; border-radius: 6px; padding: 7px 11px; min-height: 19px; }
QPushButton:hover { background: #edf4ff; border-color: #78a0d4; }
QPushButton:pressed { background: #dceaff; }
QPushButton:disabled { color: #9aa8b8; background: #f0f3f7; border-color: #e0e6ed; }
QPushButton#primary { background: #2466bb; color: white; border-color: #2466bb; font-weight: 700; }
QPushButton#primary:hover { background: #1d58a4; }
QPushButton#primary:disabled { background: #b6c7dc; border-color: #b6c7dc; }
QPushButton#danger { color: #b04444; }
QListWidget { border: none; background: white; outline: none; }
QListWidget::item { padding: 11px 10px; border-radius: 6px; margin: 2px 0px; }
QListWidget::item:selected { background: #e7f0fd; color: #205aa3; font-weight: 600; }
QListWidget::item:hover { background: #f0f5fc; }
QPlainTextEdit { border: 1px solid #d6e1ee; border-radius: 5px; padding: 6px; background: white; selection-background-color: #b9d5fa; font-family: 'Segoe UI', 'Microsoft YaHei UI'; font-size: 11pt; }
QPlainTextEdit:focus { border: 1px solid #5592de; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #eef2f7; width: 9px; border: none; }
QScrollBar::handle:vertical { background: #becbdc; min-height: 25px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QProgressBar { border: none; background: #e0e8f2; border-radius: 4px; min-height: 7px; max-height: 7px; text-align: center; }
QProgressBar::chunk { background: #3278cd; border-radius: 4px; }
QSplitter::handle { background: #f3f6fa; width: 9px; }
QStatusBar { background: #eaf0f7; color: #49617e; border-top: 1px solid #dce5ef; }
QMenuBar { background: #142c4b; color: #dae5f2; }
QMenuBar::item:selected { background: #2d4767; }
QMenu { background: white; padding: 5px; border: 1px solid #d0ddeb; }
QMenu::item { padding: 7px 25px; }
QMenu::item:selected { background: #e7f0fd; }
QToolTip { background: #15324f; color: white; padding: 6px; border: none; }
QCheckBox { spacing: 6px; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #9fb0c5; border-radius: 3px; background: white; }
QCheckBox::indicator:checked { background: #2466bb; border-color: #2466bb; }
QCheckBox::indicator:disabled { border-color: #cbd4df; background: #edf1f6; }
QDialog { background: #f3f6fa; }
QTabWidget::pane { border: 1px solid #dae4f0; background: white; border-radius: 5px; }
QTabBar::tab { padding: 8px 17px; margin-right: 3px; background: #eaf0f7; color: #61748d; border-top-left-radius: 5px; border-top-right-radius: 5px; }
QTabBar::tab:selected { background: #2466bb; color: white; }
QLineEdit, QComboBox { background: white; border: 1px solid #cdd9e7; border-radius: 5px; padding: 6px; }
QHeaderView::section { background: #eaf0f7; border: none; padding: 6px; color: #234568; }
"""
