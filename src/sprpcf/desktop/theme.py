from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette


@dataclass(frozen=True)
class ThemePalette:
    name: str
    window: str
    canvas: str
    sidebar: str
    panel: str
    panel_alt: str
    border: str
    border_soft: str
    text: str
    muted: str
    muted_2: str
    nav_text: str
    nav_hover: str
    nav_selected: str
    input_bg: str
    scroll_bg: str
    scroll_handle: str
    grid: str
    plot_bg: str
    tooltip_bg: str
    status_good_bg: str
    status_warn_bg: str
    primary: str
    primary_hover: str


# Shared semantic accents. They remain readable in both themes and are used for
# state/series meaning rather than page backgrounds.
BLUE = "#2493ff"
BLUE_DARK = "#0d4f91"
GREEN = "#1fbd5a"
GREEN_DARK = "#0c5b2c"
ORANGE = "#e97a00"
PURPLE = "#8d56d9"
CYAN = "#1294b8"
TEAL = "#138c85"
RED = "#dc4052"


DARK_THEME = ThemePalette(
    name="dark",
    window="#050c15",
    canvas="#07111d",
    sidebar="#06101b",
    panel="#0b1622",
    panel_alt="#0d1926",
    border="#263746",
    border_soft="#1d2b38",
    text="#f7fafc",
    muted="#a8b3bf",
    muted_2="#7f8c99",
    nav_text="#d5dde6",
    nav_hover="#0a1b2c",
    nav_selected="#0b315d",
    input_bg="#08131f",
    scroll_bg="#07111b",
    scroll_handle="#263746",
    grid="#20303e",
    plot_bg="#0b1622",
    tooltip_bg="#0b1622",
    status_good_bg="#0d3d20",
    status_warn_bg="#4b310a",
    primary="#0b4f91",
    primary_hover="#1262ae",
)

LIGHT_THEME = ThemePalette(
    name="light",
    # Light mode is deliberately neutral white, not a light-blue wash.
    window="#ffffff",
    canvas="#ffffff",
    sidebar="#ffffff",
    panel="#ffffff",
    panel_alt="#f7f8fa",
    border="#d7dee7",
    border_soft="#e7ebf0",
    text="#0b1f33",
    muted="#5d6b7a",
    muted_2="#7a8793",
    nav_text="#17324d",
    nav_hover="#f2f4f7",
    nav_selected="#0b2a4a",
    input_bg="#ffffff",
    scroll_bg="#f4f6f8",
    scroll_handle="#bec9d4",
    grid="#e3e8ee",
    plot_bg="#ffffff",
    tooltip_bg="#ffffff",
    status_good_bg="#e9f8ee",
    status_warn_bg="#fff4e5",
    # Navy is the primary interaction color in light mode.
    primary="#0b2a4a",
    primary_hover="#123b63",
)

THEMES = {"dark": DARK_THEME, "light": LIGHT_THEME}

# Backward-compatible dark constants used by older modules at import time.
BG = DARK_THEME.canvas
BG_DEEP = DARK_THEME.window
SIDEBAR = DARK_THEME.sidebar
PANEL = DARK_THEME.panel
PANEL_ALT = DARK_THEME.panel_alt
BORDER = DARK_THEME.border
BORDER_SOFT = DARK_THEME.border_soft
TEXT = DARK_THEME.text
MUTED = DARK_THEME.muted
MUTED_2 = DARK_THEME.muted_2


def normalize_theme(mode: str | None) -> str:
    value = str(mode or "dark").strip().lower()
    # Previous releases stored "high-contrast". Migrate it to the supported
    # dark palette instead of leaving users in a half-themed legacy state.
    if value == "high-contrast":
        return "dark"
    return value if value in THEMES else "dark"


def palette_for(mode: str | None) -> ThemePalette:
    return THEMES[normalize_theme(mode)]


def qt_palette_for(mode: str | None) -> QPalette:
    theme = palette_for(mode)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(theme.window))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Base, QColor(theme.panel))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(theme.panel_alt))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(theme.tooltip_bg))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Text, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Button, QColor(theme.panel))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(RED))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(theme.primary))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(theme.muted_2))
    palette.setColor(QPalette.ColorRole.Mid, QColor(theme.border))
    palette.setColor(QPalette.ColorRole.Midlight, QColor(theme.border_soft))
    palette.setColor(QPalette.ColorRole.Dark, QColor(theme.border))
    return palette


def stylesheet_for(mode: str | None) -> str:
    t = palette_for(mode)
    is_light = t.name == "light"

    quick_text = t.primary if is_light else "#f7fafc"
    quick_hover_text = "#ffffff"
    quick_background = "#ffffff" if is_light else t.panel
    quick_blue_border = t.primary if is_light else "#167de0"
    quick_green_border = "#18864a" if is_light else "#21a94f"
    quick_purple_border = "#7150a5" if is_light else "#7e52b7"
    quick_orange_border = "#b46500" if is_light else "#c56e00"
    quick_cyan_border = "#16799d" if is_light else "#1d94c8"
    quick_teal_border = "#14776f" if is_light else "#109b94"

    return f"""
* {{
    font-family: "Segoe UI", "Inter", "Arial";
    color: {t.text};
}}
QMainWindow, QWidget#root {{
    background: {t.window};
}}
QFrame#sidebar {{
    background: {t.sidebar};
    border-right: 1px solid {t.border_soft};
}}
QFrame#topbar, QFrame#statusbar {{
    background: {t.window};
    border-color: {t.border_soft};
}}
QFrame#topbar {{ border-bottom: 1px solid {t.border_soft}; }}
QFrame#statusbar {{ border-top: 1px solid {t.border_soft}; }}
QFrame.card, QFrame#card {{
    background: {t.panel};
    border: 1px solid {t.border};
    border-radius: 10px;
}}
QFrame#panel {{
    background: {t.panel};
    border: 1px solid {t.border};
    border-radius: 10px;
}}
QLabel#brandTitle {{ font-size: 18px; font-weight: 700; color:{t.text}; }}
QLabel#brandSub {{ color: {t.muted}; font-size: 12px; }}
QLabel#pageTitle {{ font-size: 20px; font-weight: 700; color:{t.text}; }}
QLabel#sectionTitle {{ font-size: 16px; font-weight: 700; color:{t.text}; }}
QLabel#smallTitle {{ font-size: 13px; font-weight: 600; color:{t.text}; }}
QLabel#muted {{ color: {t.muted}; }}
QLabel#mutedSmall {{ color: {t.muted}; font-size: 11px; }}
QLabel#valueGreen {{ color: {GREEN}; font-size: 20px; font-weight: 700; }}
QLabel#valueBlue {{ color: {BLUE}; font-size: 20px; font-weight: 700; }}
QLabel#valuePurple {{ color: {PURPLE}; font-size: 20px; font-weight: 700; }}
QLabel#valueOrange {{ color: {ORANGE}; font-size: 20px; font-weight: 700; }}
QLabel#valueWhite {{ color: {t.text}; font-size: 20px; font-weight: 700; }}
QLabel#success {{ color: {GREEN}; font-weight: 600; }}
QLabel#warning {{ color: {ORANGE}; font-weight: 600; }}
QLabel#error {{ color: {RED}; font-weight: 600; }}
QPushButton {{
    background: transparent;
    color: {t.text};
    border: 1px solid {t.border};
    border-radius: 7px;
    min-height: 30px;
    padding: 5px 10px;
}}
QPushButton:hover {{ border-color: {t.primary}; background: {t.nav_hover}; }}
QPushButton:pressed {{ background: {t.border_soft}; }}
QPushButton#topIconButton {{ border:none; padding:5px; background:transparent; }}
QPushButton#topIconButton:hover {{ background:{t.nav_hover}; border:none; border-radius:7px; }}
QPushButton#navButton {{
    text-align: left;
    border: none;
    border-radius: 6px;
    padding: 8px 12px;
    color: {t.nav_text};
    min-height: 28px;
}}
QPushButton#navButton:hover {{ background: {t.nav_hover}; }}
QPushButton#navButton:checked {{
    background: {t.nav_selected};
    border: 1px solid {t.nav_selected};
    color: #ffffff;
}}
QPushButton#quickBlue,
QPushButton#quickGreen,
QPushButton#quickPurple,
QPushButton#quickOrange,
QPushButton#quickCyan,
QPushButton#quickTeal {{
    background:{quick_background};
    color:{quick_text};
    font-weight:600;
}}
QPushButton#quickBlue {{ border-color:{quick_blue_border}; }}
QPushButton#quickGreen {{ border-color:{quick_green_border}; }}
QPushButton#quickPurple {{ border-color:{quick_purple_border}; }}
QPushButton#quickOrange {{ border-color:{quick_orange_border}; }}
QPushButton#quickCyan {{ border-color:{quick_cyan_border}; }}
QPushButton#quickTeal {{ border-color:{quick_teal_border}; }}
QPushButton#quickBlue:hover,
QPushButton#quickGreen:hover,
QPushButton#quickPurple:hover,
QPushButton#quickOrange:hover,
QPushButton#quickCyan:hover,
QPushButton#quickTeal:hover {{
    background:{t.primary};
    color:{quick_hover_text};
}}
QPushButton#primary {{
    background: {t.primary};
    border-color: {t.primary};
    color:#ffffff;
    font-weight: 600;
}}
QPushButton#primary:hover {{ background:{t.primary_hover}; border-color:{t.primary_hover}; }}
QPushButton#successButton {{
    background: {GREEN_DARK if not is_light else '#e9f8ee'};
    color:{'#ffffff' if not is_light else '#0f6c36'};
    border-color: {GREEN};
    font-weight: 600;
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {{
    background: {t.input_bg};
    color:{t.text};
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: {t.primary};
    selection-color:#ffffff;
}}
QComboBox QAbstractItemView {{
    background:{t.panel};
    color:{t.text};
    border:1px solid {t.border};
    selection-background-color:{t.primary};
    selection-color:#ffffff;
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QCheckBox {{ color:{t.text}; }}
QProgressBar {{
    background: {t.border_soft};
    border: none;
    border-radius: 4px;
    height: 7px;
    text-align: center;
}}
QProgressBar::chunk {{ background: {GREEN}; border-radius: 4px; }}
QScrollArea {{ border: none; background: {t.canvas}; }}
QScrollArea > QWidget > QWidget {{ background:{t.canvas}; }}
QScrollBar:vertical {{ background: {t.scroll_bg}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {t.scroll_handle}; border-radius: 5px; min-height: 25px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: {t.scroll_bg}; height: 10px; margin:0; }}
QScrollBar::handle:horizontal {{ background:{t.scroll_handle}; border-radius:5px; min-width:25px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width:0; }}
QTableWidget, QTableView {{
    background: {t.panel};
    color:{t.text};
    alternate-background-color: {t.panel_alt};
    border: 1px solid {t.border};
    gridline-color: {t.border_soft};
}}
QHeaderView::section {{
    background: {t.panel_alt};
    color:{t.text};
    border: none;
    border-bottom: 1px solid {t.border};
    padding: 6px;
    font-weight: 600;
}}
QDialog {{ background: {t.window}; color:{t.text}; }}
QDialogButtonBox QPushButton {{ min-width:80px; }}
QGroupBox {{
    border: 1px solid {t.border};
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QToolTip {{ background: {t.tooltip_bg}; color: {t.text}; border: 1px solid {t.border}; }}
"""


APP_STYLESHEET = stylesheet_for("dark")
