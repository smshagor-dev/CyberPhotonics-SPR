from __future__ import annotations

BG = "#07111d"
BG_DEEP = "#050c15"
SIDEBAR = "#06101b"
PANEL = "#0b1622"
PANEL_ALT = "#0d1926"
BORDER = "#263746"
BORDER_SOFT = "#1d2b38"
TEXT = "#f7fafc"
MUTED = "#a8b3bf"
MUTED_2 = "#7f8c99"
BLUE = "#2493ff"
BLUE_DARK = "#0d4f91"
GREEN = "#25e75b"
GREEN_DARK = "#0c5b2c"
ORANGE = "#ff8a00"
PURPLE = "#b76cff"
CYAN = "#2de2e6"
TEAL = "#18c7bf"
RED = "#ff4f64"

APP_STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "Inter", "Arial";
    color: {TEXT};
}}
QMainWindow, QWidget#root {{
    background: {BG_DEEP};
}}
QFrame#sidebar {{
    background: {SIDEBAR};
    border-right: 1px solid {BORDER_SOFT};
}}
QFrame#topbar, QFrame#statusbar {{
    background: {BG_DEEP};
    border-color: {BORDER_SOFT};
}}
QFrame#topbar {{ border-bottom: 1px solid {BORDER_SOFT}; }}
QFrame#statusbar {{ border-top: 1px solid {BORDER_SOFT}; }}
QFrame.card, QFrame#card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QFrame#panel {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QLabel#brandTitle {{ font-size: 18px; font-weight: 700; }}
QLabel#brandSub {{ color: {MUTED}; font-size: 12px; }}
QLabel#pageTitle {{ font-size: 20px; font-weight: 700; }}
QLabel#sectionTitle {{ font-size: 16px; font-weight: 700; }}
QLabel#smallTitle {{ font-size: 13px; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#mutedSmall {{ color: {MUTED}; font-size: 11px; }}
QLabel#valueGreen {{ color: {GREEN}; font-size: 20px; font-weight: 700; }}
QLabel#valueBlue {{ color: {BLUE}; font-size: 20px; font-weight: 700; }}
QLabel#valuePurple {{ color: {PURPLE}; font-size: 20px; font-weight: 700; }}
QLabel#valueOrange {{ color: {ORANGE}; font-size: 20px; font-weight: 700; }}
QLabel#valueWhite {{ color: {TEXT}; font-size: 20px; font-weight: 700; }}
QLabel#success {{ color: {GREEN}; font-weight: 600; }}
QLabel#warning {{ color: {ORANGE}; font-weight: 600; }}
QLabel#error {{ color: {RED}; font-weight: 600; }}
QPushButton {{
    background: transparent;
    border: 1px solid {BORDER};
    border-radius: 7px;
    min-height: 30px;
    padding: 5px 10px;
}}
QPushButton:hover {{ border-color: {BLUE}; background: #0b2238; }}
QPushButton:pressed {{ background: #0c2c49; }}
QPushButton#navButton {{
    text-align: left;
    border: none;
    border-radius: 6px;
    padding: 8px 12px;
    color: #d5dde6;
    min-height: 28px;
}}
QPushButton#navButton:hover {{ background: #0a1b2c; }}
QPushButton#navButton:checked {{
    background: #082d59;
    border: 1px solid #0c5cae;
    color: white;
}}
QPushButton#quickBlue {{ background: #0b3d73; border-color: #167de0; }}
QPushButton#quickGreen {{ background: #0d4b28; border-color: #21a94f; }}
QPushButton#quickPurple {{ background: #3f2367; border-color: #7e52b7; }}
QPushButton#quickOrange {{ background: #623300; border-color: #c56e00; }}
QPushButton#quickCyan {{ background: #0a3b55; border-color: #1d94c8; }}
QPushButton#quickTeal {{ background: #07504c; border-color: #109b94; }}
QPushButton#primary {{
    background: #0b4f91;
    border-color: #1f8fff;
    font-weight: 600;
}}
QPushButton#successButton {{
    background: #0d572f;
    border-color: #1bc958;
    font-weight: 600;
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {{
    background: #08131f;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #0d5aa6;
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QProgressBar {{
    background: #142331;
    border: none;
    border-radius: 4px;
    height: 7px;
    text-align: center;
}}
QProgressBar::chunk {{ background: {GREEN}; border-radius: 4px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: #07111b; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #263746; border-radius: 5px; min-height: 25px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QTableWidget {{
    background: {PANEL};
    alternate-background-color: {PANEL_ALT};
    border: 1px solid {BORDER};
    gridline-color: {BORDER_SOFT};
}}
QHeaderView::section {{
    background: #0b1622;
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px;
    font-weight: 600;
}}
QDialog {{ background: {BG_DEEP}; }}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER}; }}
"""
