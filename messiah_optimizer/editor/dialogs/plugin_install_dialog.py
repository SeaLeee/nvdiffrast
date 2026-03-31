"""
Dialog for installing/uninstalling the Optimizer plugin into Messiah Engine.
"""

import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSpinBox, QCheckBox, QTextEdit, QFileDialog,
    QGroupBox, QFormLayout, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor


class _InstallWorker(QThread):
    """Run install/uninstall in a background thread to keep UI responsive."""
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, summary

    def __init__(self, action, engine_root, port=9800, auto_start=True):
        super().__init__()
        self.action = action  # 'install' or 'uninstall'
        self.engine_root = engine_root
        self.port = port
        self.auto_start = auto_start

    def run(self):
        import sys
        import io

        # Capture print output from install.py
        capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = capture

        try:
            from messiah_plugin.install import install, uninstall

            if self.action == 'install':
                ok = install(self.engine_root, port=self.port,
                             auto_start=self.auto_start)
            else:
                ok = uninstall(self.engine_root)

            sys.stdout = old_stdout
            log_text = capture.getvalue()
            self.log.emit(log_text)

            if ok:
                summary = "安装成功！" if self.action == 'install' else "卸载成功！"
            else:
                summary = "操作失败，请检查引擎路径是否正确。"
            self.finished.emit(ok, summary)

        except Exception as e:
            sys.stdout = old_stdout
            log_text = capture.getvalue()
            self.log.emit(log_text)
            self.log.emit(f"\n错误: {e}")
            self.finished.emit(False, f"操作异常: {e}")


class PluginInstallDialog(QDialog):
    """One-click dialog for installing Optimizer plugin into Messiah Engine."""

    def __init__(self, parent=None, default_engine_root=""):
        super().__init__(parent)
        self.setWindowTitle("安装 Messiah 引擎插件")
        self.setMinimumSize(560, 420)
        self._worker = None
        self._setup_ui(default_engine_root)

    def _setup_ui(self, default_engine_root):
        layout = QVBoxLayout(self)

        # --- Description ---
        desc = QLabel(
            "将 Optimizer Bridge 插件安装到 Messiah 引擎的 Editor/QtScript/ 目录，\n"
            "使引擎启动时自动运行 RPC 服务端，从而支持与本工具的实时联动。"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # --- Engine Root ---
        path_group = QGroupBox("引擎路径")
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit(default_engine_root)
        self.path_edit.setPlaceholderText("例如: D:\\NewTrunk\\Engine\\src\\Engine")
        path_layout.addWidget(self.path_edit)
        browse_btn = QPushButton("浏览...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_engine_root)
        path_layout.addWidget(browse_btn)
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)

        # --- Options ---
        opt_group = QGroupBox("安装选项")
        opt_layout = QFormLayout()

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(9800)
        opt_layout.addRow("RPC 服务端口:", self.port_spin)

        self.autostart_cb = QCheckBox("引擎启动时自动运行插件服务")
        self.autostart_cb.setChecked(True)
        opt_layout.addRow(self.autostart_cb)

        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)

        # --- Status indicator ---
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.status_label)

        # --- Log output ---
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMaximumHeight(140)
        self.log_text.setPlaceholderText("安装日志将显示在此处...")
        layout.addWidget(self.log_text)

        # --- Buttons ---
        btn_layout = QHBoxLayout()

        self.install_btn = QPushButton("一键安装")
        self.install_btn.setMinimumHeight(36)
        self.install_btn.setStyleSheet(
            "QPushButton { background-color: #238636; color: white; "
            "font-weight: bold; border-radius: 4px; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #2ea043; }"
            "QPushButton:disabled { background-color: #484f58; }"
        )
        self.install_btn.clicked.connect(self._on_install)
        btn_layout.addWidget(self.install_btn)

        self.uninstall_btn = QPushButton("卸载插件")
        self.uninstall_btn.setMinimumHeight(36)
        self.uninstall_btn.clicked.connect(self._on_uninstall)
        btn_layout.addWidget(self.uninstall_btn)

        self.close_btn = QPushButton("关闭")
        self.close_btn.setMinimumHeight(36)
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

        # --- Check current status ---
        self._check_install_status(default_engine_root)

    def _browse_engine_root(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择 Messiah 引擎根目录",
            self.path_edit.text() or "",
        )
        if path:
            self.path_edit.setText(path)
            self._check_install_status(path)

    def _check_install_status(self, engine_root):
        """Check if plugin is already installed."""
        if not engine_root:
            self.status_label.setText("")
            return

        plugin_dir = os.path.join(engine_root, 'Editor', 'QtScript', 'optimizer_plugin')
        qtmain = os.path.join(engine_root, 'Editor', 'QtScript', 'qtmain.py')

        if not os.path.isdir(os.path.join(engine_root, 'Editor', 'QtScript')):
            self.status_label.setText("⚠ 未找到 Editor/QtScript/ 目录，请检查路径")
            self.status_label.setStyleSheet("font-weight: bold; color: #d29922;")
            return

        if os.path.isdir(plugin_dir):
            # Check if auto-start is injected
            has_autostart = False
            if os.path.exists(qtmain):
                with open(qtmain, 'r', encoding='utf-8') as f:
                    has_autostart = '# === nvdiffrast Optimizer Bridge ===' in f.read()

            status = "✅ 插件已安装"
            if has_autostart:
                status += "（自动启动已配置）"
            else:
                status += "（未配置自动启动）"
            self.status_label.setText(status)
            self.status_label.setStyleSheet("font-weight: bold; color: #3fb950;")
        else:
            self.status_label.setText("⬚ 插件未安装")
            self.status_label.setStyleSheet("font-weight: bold; color: #8b949e;")

    def _validate_path(self) -> str:
        engine_root = self.path_edit.text().strip()
        if not engine_root:
            QMessageBox.warning(self, "路径为空", "请先选择 Messiah 引擎根目录。")
            return ""
        if not os.path.isdir(engine_root):
            QMessageBox.warning(self, "路径无效", f"目录不存在:\n{engine_root}")
            return ""
        return engine_root

    def _set_busy(self, busy):
        self.install_btn.setEnabled(not busy)
        self.uninstall_btn.setEnabled(not busy)
        self.path_edit.setEnabled(not busy)

    def _on_install(self):
        engine_root = self._validate_path()
        if not engine_root:
            return

        self._set_busy(True)
        self.log_text.clear()
        self.status_label.setText("⏳ 正在安装...")
        self.status_label.setStyleSheet("font-weight: bold; color: #58a6ff;")

        self._worker = _InstallWorker(
            action='install',
            engine_root=engine_root,
            port=self.port_spin.value(),
            auto_start=self.autostart_cb.isChecked(),
        )
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_uninstall(self):
        engine_root = self._validate_path()
        if not engine_root:
            return

        reply = QMessageBox.question(
            self, "确认卸载",
            f"确定要从以下引擎中卸载 Optimizer 插件吗？\n\n{engine_root}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._set_busy(True)
        self.log_text.clear()
        self.status_label.setText("⏳ 正在卸载...")
        self.status_label.setStyleSheet("font-weight: bold; color: #58a6ff;")

        self._worker = _InstallWorker(
            action='uninstall',
            engine_root=engine_root,
        )
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, text):
        self.log_text.append(text)
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)

    def _on_finished(self, success, summary):
        self._set_busy(False)
        if success:
            self.status_label.setStyleSheet("font-weight: bold; color: #3fb950;")
        else:
            self.status_label.setStyleSheet("font-weight: bold; color: #f85149;")
        self.status_label.setText(summary)
        self._check_install_status(self.path_edit.text().strip())
        self._worker = None
