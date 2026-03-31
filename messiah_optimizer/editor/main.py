"""
NvDiffRast Messiah Optimizer - Application Entry Point
"""

import sys
import os
import traceback
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from editor.main_window import OptimizerMainWindow
from editor.theme import DARK_STYLESHEET

# Crash log path
_CRASH_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'crash.log')


def _global_exception_handler(exc_type, exc_value, exc_tb):
    """Catch ALL unhandled exceptions (including PyQt6 slot exceptions) and log them."""
    msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    # Print to stderr
    sys.stderr.write(f"\n{'='*60}\n[UNHANDLED EXCEPTION]\n{msg}{'='*60}\n")
    # Append to crash log
    try:
        with open(_CRASH_LOG, 'a', encoding='utf-8') as f:
            import datetime
            f.write(f"\n{'='*60}\n")
            f.write(f"[{datetime.datetime.now().isoformat()}] UNHANDLED EXCEPTION\n")
            f.write(msg)
            f.write(f"{'='*60}\n")
    except Exception:
        pass
    # Do NOT call sys.exit() — keep the app alive


def main():
    # Install global exception handler BEFORE anything else
    sys.excepthook = _global_exception_handler

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    # High DPI must be set before QApplication is created
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("NvDiffRast Messiah Optimizer")
    app.setOrganizationName("MessiahEngine")

    # Apply dark theme
    app.setStyleSheet(DARK_STYLESHEET)

    # Set default font
    font = QFont("Segoe UI", 10)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)

    window = OptimizerMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
