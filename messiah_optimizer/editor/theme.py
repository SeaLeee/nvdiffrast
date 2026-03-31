"""
Premium dark theme stylesheet for the Messiah Optimizer.
Rounded corners, subtle gradients, clear separation.
"""

# Color palette
PALETTE = {
    'bg_darkest':   '#0d1117',
    'bg_dark':      '#161b22',
    'bg_medium':    '#1c2128',
    'bg_panel':     '#21262d',
    'bg_input':     '#0d1117',
    'bg_hover':     '#30363d',
    'bg_selected':  '#1f6feb33',
    'border':       '#30363d',
    'border_light': '#3d444d',
    'text':         '#e6edf3',
    'text_dim':     '#8b949e',
    'text_muted':   '#484f58',
    'accent':       '#58a6ff',
    'accent_hover': '#79c0ff',
    'green':        '#3fb950',
    'green_hover':  '#56d364',
    'green_bg':     '#238636',
    'red':          '#f85149',
    'red_hover':    '#ff6e6e',
    'orange':       '#d29922',
    'purple':       '#bc8cff',
    'scrollbar':    '#484f58',
    'scrollbar_bg': '#161b22',
}

P = PALETTE

DARK_STYLESHEET = f"""

/* ========== Global ========== */
QWidget {{
    background-color: {P['bg_dark']};
    color: {P['text']};
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    font-size: 13px;
    selection-background-color: {P['accent']};
    selection-color: #ffffff;
}}

QMainWindow {{
    background-color: {P['bg_darkest']};
}}

QMainWindow::separator {{
    background-color: {P['bg_darkest']};
    width: 2px;
    height: 2px;
}}

/* ========== Menu Bar ========== */
QMenuBar {{
    background-color: {P['bg_darkest']};
    color: {P['text']};
    border-bottom: 1px solid {P['border']};
    padding: 2px 0px;
    spacing: 2px;
}}

QMenuBar::item {{
    background: transparent;
    padding: 6px 12px;
    border-radius: 6px;
    margin: 2px 1px;
}}

QMenuBar::item:selected {{
    background-color: {P['bg_hover']};
}}

QMenuBar::item:pressed {{
    background-color: {P['bg_selected']};
}}

QMenu {{
    background-color: {P['bg_panel']};
    border: 1px solid {P['border']};
    border-radius: 10px;
    padding: 6px 4px;
}}

QMenu::item {{
    padding: 8px 32px 8px 16px;
    border-radius: 6px;
    margin: 2px 4px;
}}

QMenu::item:selected {{
    background-color: {P['accent']};
    color: #ffffff;
}}

QMenu::separator {{
    height: 1px;
    background-color: {P['border']};
    margin: 4px 12px;
}}

QMenu::indicator {{
    width: 16px;
    height: 16px;
    margin-left: 6px;
}}

/* ========== Dock Widgets ========== */
QDockWidget {{
    color: {P['text']};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    font-weight: 600;
    font-size: 12px;
}}

QDockWidget::title {{
    background-color: {P['bg_medium']};
    text-align: left;
    padding: 8px 12px;
    border: 1px solid {P['border']};
    border-radius: 8px 8px 0px 0px;
}}

QDockWidget::close-button, QDockWidget::float-button {{
    background: transparent;
    border: none;
    padding: 4px;
    border-radius: 4px;
    icon-size: 14px;
}}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background-color: {P['bg_hover']};
}}

QDockWidget > QWidget {{
    border: 1px solid {P['border']};
    border-top: none;
    border-radius: 0px 0px 8px 8px;
}}

/* ========== Splitter ========== */
QSplitter::handle {{
    background-color: {P['bg_darkest']};
}}

QSplitter::handle:horizontal {{
    width: 3px;
}}

QSplitter::handle:vertical {{
    height: 3px;
}}

QSplitter::handle:hover {{
    background-color: {P['accent']};
}}

/* ========== Tab Widget ========== */
QTabWidget::pane {{
    background-color: {P['bg_panel']};
    border: 1px solid {P['border']};
    border-radius: 0px 0px 8px 8px;
    top: -1px;
}}

QTabBar {{
    background: transparent;
}}

QTabBar::tab {{
    background-color: {P['bg_medium']};
    color: {P['text_dim']};
    border: 1px solid {P['border']};
    border-bottom: none;
    padding: 8px 18px;
    margin-right: 2px;
    border-radius: 8px 8px 0px 0px;
    min-width: 80px;
    font-weight: 500;
}}

QTabBar::tab:selected {{
    background-color: {P['bg_panel']};
    color: {P['text']};
    border-bottom: 2px solid {P['accent']};
}}

QTabBar::tab:hover:!selected {{
    background-color: {P['bg_hover']};
    color: {P['text']};
}}

/* ========== Buttons ========== */
QPushButton {{
    background-color: {P['bg_panel']};
    color: {P['text']};
    border: 1px solid {P['border']};
    border-radius: 8px;
    padding: 7px 18px;
    font-weight: 500;
    min-height: 18px;
}}

QPushButton:hover {{
    background-color: {P['bg_hover']};
    border-color: {P['border_light']};
}}

QPushButton:pressed {{
    background-color: {P['bg_darkest']};
}}

QPushButton:disabled {{
    color: {P['text_muted']};
    background-color: {P['bg_medium']};
    border-color: {P['border']};
}}

/* Primary action buttons */
QPushButton[cssClass="primary"] {{
    background-color: {P['green_bg']};
    color: #ffffff;
    border: 1px solid {P['green']};
    font-weight: 600;
}}

QPushButton[cssClass="primary"]:hover {{
    background-color: {P['green']};
    border-color: {P['green_hover']};
}}

QPushButton[cssClass="primary"]:pressed {{
    background-color: #196c2e;
}}

/* Accent action buttons */
QPushButton[cssClass="accent"] {{
    background-color: #1a3a5c;
    color: {P['accent']};
    border: 1px solid {P['accent']};
    font-weight: 600;
}}

QPushButton[cssClass="accent"]:hover {{
    background-color: #1f4d7a;
    color: {P['accent_hover']};
}}

/* Danger buttons */
QPushButton[cssClass="danger"] {{
    background-color: #4a1c1c;
    color: {P['red']};
    border: 1px solid {P['red']};
}}

QPushButton[cssClass="danger"]:hover {{
    background-color: #601f1f;
}}

/* ========== Input Fields ========== */
QLineEdit {{
    background-color: {P['bg_input']};
    color: {P['text']};
    border: 1px solid {P['border']};
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 13px;
}}

QLineEdit:focus {{
    border-color: {P['accent']};
    background-color: {P['bg_darkest']};
}}

QLineEdit:disabled {{
    color: {P['text_muted']};
    background-color: {P['bg_medium']};
}}

/* ========== Spin Boxes ========== */
QSpinBox, QDoubleSpinBox {{
    background-color: {P['bg_input']};
    color: {P['text']};
    border: 1px solid {P['border']};
    border-radius: 8px;
    padding: 5px 10px;
    min-height: 20px;
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {P['accent']};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid {P['border']};
    border-top-right-radius: 8px;
    background-color: {P['bg_panel']};
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
    background-color: {P['bg_hover']};
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border-left: 1px solid {P['border']};
    border-bottom-right-radius: 8px;
    background-color: {P['bg_panel']};
}}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {P['bg_hover']};
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {P['text_dim']};
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {P['text_dim']};
}}

/* ========== Combo Box ========== */
QComboBox {{
    background-color: {P['bg_input']};
    color: {P['text']};
    border: 1px solid {P['border']};
    border-radius: 8px;
    padding: 6px 12px;
    min-height: 20px;
    min-width: 80px;
}}

QComboBox:hover {{
    border-color: {P['border_light']};
}}

QComboBox:focus {{
    border-color: {P['accent']};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 28px;
    border-left: 1px solid {P['border']};
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
    background-color: transparent;
}}

QComboBox::down-arrow {{
    width: 0; height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {P['text_dim']};
}}

QComboBox QAbstractItemView {{
    background-color: {P['bg_panel']};
    border: 1px solid {P['border']};
    border-radius: 8px;
    padding: 4px;
    selection-background-color: {P['accent']};
    selection-color: #ffffff;
    outline: none;
}}

QComboBox QAbstractItemView::item {{
    padding: 6px 12px;
    border-radius: 4px;
    min-height: 24px;
}}

/* ========== Check Box ========== */
QCheckBox {{
    spacing: 8px;
    color: {P['text']};
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {P['border_light']};
    border-radius: 5px;
    background-color: {P['bg_input']};
}}

QCheckBox::indicator:hover {{
    border-color: {P['accent']};
}}

QCheckBox::indicator:checked {{
    background-color: {P['accent']};
    border-color: {P['accent']};
}}

/* ========== Group Box ========== */
QGroupBox {{
    background-color: {P['bg_medium']};
    border: 1px solid {P['border']};
    border-radius: 10px;
    margin-top: 14px;
    padding: 16px 12px 10px 12px;
    font-weight: 600;
    font-size: 12px;
    color: {P['text']};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0px 8px;
    color: {P['accent']};
    background-color: {P['bg_medium']};
    border-radius: 4px;
}}

/* ========== Tree Widget ========== */
QTreeWidget, QTreeView {{
    background-color: {P['bg_input']};
    border: 1px solid {P['border']};
    border-radius: 8px;
    padding: 4px;
    alternate-background-color: {P['bg_medium']};
    outline: none;
}}

QTreeWidget::item, QTreeView::item {{
    padding: 5px 4px;
    border-radius: 4px;
    margin: 1px 0px;
}}

QTreeWidget::item:selected, QTreeView::item:selected {{
    background-color: {P['bg_selected']};
    color: {P['accent']};
}}

QTreeWidget::item:hover, QTreeView::item:hover {{
    background-color: {P['bg_hover']};
}}

QTreeWidget::branch {{
    background: transparent;
}}

QHeaderView::section {{
    background-color: {P['bg_medium']};
    color: {P['text_dim']};
    border: none;
    border-bottom: 1px solid {P['border']};
    padding: 6px 10px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
}}

/* ========== List Widget ========== */
QListWidget {{
    background-color: {P['bg_input']};
    border: 1px solid {P['border']};
    border-radius: 8px;
    padding: 4px;
    outline: none;
}}

QListWidget::item {{
    padding: 6px 10px;
    border-radius: 6px;
    margin: 1px 2px;
}}

QListWidget::item:selected {{
    background-color: {P['bg_selected']};
    color: {P['accent']};
}}

QListWidget::item:hover {{
    background-color: {P['bg_hover']};
}}

/* ========== Text Edit / Log ========== */
QTextEdit {{
    background-color: {P['bg_input']};
    color: {P['text']};
    border: 1px solid {P['border']};
    border-radius: 8px;
    padding: 8px;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
}}

/* ========== Scroll Bars ========== */
QScrollBar:vertical {{
    background-color: {P['scrollbar_bg']};
    width: 10px;
    margin: 0;
    border-radius: 5px;
}}

QScrollBar::handle:vertical {{
    background-color: {P['scrollbar']};
    min-height: 30px;
    border-radius: 5px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {P['text_dim']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {P['scrollbar_bg']};
    height: 10px;
    margin: 0;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal {{
    background-color: {P['scrollbar']};
    min-width: 30px;
    border-radius: 5px;
    margin: 2px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {P['text_dim']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

/* ========== Slider ========== */
QSlider::groove:horizontal {{
    height: 6px;
    background-color: {P['bg_input']};
    border: 1px solid {P['border']};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background-color: {P['accent']};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    border: 2px solid {P['bg_darkest']};
}}

QSlider::handle:horizontal:hover {{
    background-color: {P['accent_hover']};
}}

QSlider::sub-page:horizontal {{
    background-color: {P['accent']};
    border-radius: 3px;
}}

/* ========== Status Bar ========== */
QStatusBar {{
    background-color: {P['bg_darkest']};
    color: {P['text_dim']};
    border-top: 1px solid {P['border']};
    font-size: 12px;
    padding: 2px 8px;
}}

QStatusBar::item {{
    border: none;
}}

/* ========== Label ========== */
QLabel {{
    background: transparent;
    border: none;
    color: {P['text']};
}}

QLabel[cssClass="dim"] {{
    color: {P['text_dim']};
    font-size: 11px;
}}

QLabel[cssClass="accent"] {{
    color: {P['accent']};
    font-weight: 600;
}}

QLabel[cssClass="title"] {{
    font-size: 16px;
    font-weight: 700;
    color: {P['text']};
}}

/* ========== Form Layout Labels ========== */
QFormLayout {{
    spacing: 8px;
}}

/* ========== Tool Tips ========== */
QToolTip {{
    background-color: {P['bg_panel']};
    color: {P['text']};
    border: 1px solid {P['border']};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}

/* ========== Progress Bar ========== */
QProgressBar {{
    background-color: {P['bg_input']};
    border: 1px solid {P['border']};
    border-radius: 6px;
    text-align: center;
    color: {P['text']};
    font-size: 11px;
    min-height: 18px;
}}

QProgressBar::chunk {{
    background-color: {P['accent']};
    border-radius: 5px;
}}

/* ========== Message Box ========== */
QMessageBox {{
    background-color: {P['bg_panel']};
}}

QMessageBox QLabel {{
    color: {P['text']};
}}

/* ========== File Dialog ========== */
QFileDialog {{
    background-color: {P['bg_panel']};
}}

/* ========== Viewport special style ========== */
QLabel#viewport_display {{
    background-color: {P['bg_darkest']};
    border: 1px solid {P['border']};
    border-radius: 8px;
}}

QLabel#viewport_info {{
    color: {P['text_dim']};
    font-size: 11px;
    font-family: "Cascadia Code", "Consolas", monospace;
    padding: 2px 8px;
}}

/* ========== Compare widget ========== */
QLabel#compare_rendered, QLabel#compare_reference {{
    background-color: {P['bg_darkest']};
    border: 1px solid {P['border']};
    border-radius: 8px;
}}
"""
