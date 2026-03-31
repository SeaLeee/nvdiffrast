"""
One-click engine launcher: configure engine path, launch Messiah Editor,
auto-install optimizer plugin, and connect the bridge.
"""

import os
import subprocess
import time
import socket
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSpinBox, QCheckBox, QTextEdit, QFileDialog,
    QGroupBox, QFormLayout, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor


class _LaunchWorker(QThread):
    """Background worker: install plugin → launch engine → wait for RPC ready."""
    log = pyqtSignal(str)
    stage_changed = pyqtSignal(str)   # stage description
    finished = pyqtSignal(bool, str)  # success, summary

    def __init__(self, engine_root: str, engine_bat: str,
                 port: int, auto_install: bool,
                 connect_timeout: float = 60.0):
        super().__init__()
        self.engine_root = engine_root
        self.engine_bat = engine_bat
        self.port = port
        self.auto_install = auto_install
        self.connect_timeout = connect_timeout
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            # Step 1: Install plugin
            if self.auto_install:
                self.stage_changed.emit("正在安装插件...")
                self.log.emit("[1/3] 安装 Optimizer 插件到引擎...")
                ok = self._install_plugin()
                if not ok:
                    self.finished.emit(False, "插件安装失败")
                    return
                self.log.emit("[1/3] 插件安装完成\n")
            else:
                self.log.emit("[1/3] 跳过插件安装（未勾选）\n")

            if self._cancelled:
                self.finished.emit(False, "已取消")
                return

            # Step 2: Launch engine
            self.stage_changed.emit("正在启动引擎...")
            self.log.emit("[2/3] 启动 Messiah Editor...")
            ok = self._launch_engine()
            if not ok:
                self.finished.emit(False, "引擎启动失败")
                return
            self.log.emit("[2/3] 引擎进程已启动\n")

            if self._cancelled:
                self.finished.emit(False, "已取消")
                return

            # Step 3: Wait for RPC server to be ready
            self.stage_changed.emit(f"等待 RPC 服务就绪 (端口 {self.port})...")
            self.log.emit(f"[3/3] 等待引擎 RPC 端口 {self.port} 就绪...")
            ok = self._wait_for_rpc()
            if not ok:
                self.finished.emit(
                    False,
                    f"引擎已启动，但 RPC 端口 {self.port} 未就绪 "
                    f"(超时 {self.connect_timeout:.0f}s)。\n"
                    "请确认引擎已完全加载，且插件已正确安装。"
                )
                return
            self.log.emit(f"[3/3] RPC 端口 {self.port} 已就绪！\n")

            self.finished.emit(True, "引擎已启动，RPC 连接就绪")

        except Exception as e:
            self.log.emit(f"\n错误: {e}")
            self.finished.emit(False, f"异常: {e}")

    def _install_plugin(self) -> bool:
        import sys
        import io
        capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = capture
        try:
            from messiah_plugin.install import install
            ok = install(self.engine_root, port=self.port, auto_start=True)
            sys.stdout = old_stdout
            self.log.emit(capture.getvalue())
            return ok
        except Exception as e:
            sys.stdout = old_stdout
            self.log.emit(capture.getvalue())
            self.log.emit(f"安装异常: {e}")
            return False

    def _launch_engine(self) -> bool:
        # Find the bat file
        # engine_bat is relative to the parent of engine_root
        # e.g. engine_root = D:\NewTrunk\Engine\src\Engine
        #      engine_bat = Messiah_Editor.bat
        #      full bat path = D:\NewTrunk\Engine\src\Messiah_Editor.bat
        parent_dir = os.path.dirname(self.engine_root)
        bat_path = os.path.join(parent_dir, self.engine_bat)

        if not os.path.exists(bat_path):
            # Also try engine_root itself
            bat_path2 = os.path.join(self.engine_root, self.engine_bat)
            if os.path.exists(bat_path2):
                bat_path = bat_path2
            else:
                self.log.emit(f"未找到启动脚本:\n  {bat_path}\n  {bat_path2}")
                return False

        self.log.emit(f"执行: {bat_path}")
        try:
            # Use START to launch detached
            subprocess.Popen(
                f'cmd /c "{bat_path}"',
                cwd=os.path.dirname(bat_path),
                shell=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                             | subprocess.DETACHED_PROCESS,
            )
            return True
        except Exception as e:
            self.log.emit(f"启动失败: {e}")
            return False

    def _wait_for_rpc(self) -> bool:
        start = time.monotonic()
        attempt = 0
        while time.monotonic() - start < self.connect_timeout:
            if self._cancelled:
                return False
            attempt += 1
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2.0)
                s.connect(('127.0.0.1', self.port))
                s.close()
                return True
            except (ConnectionRefusedError, socket.timeout, OSError):
                pass
            finally:
                try:
                    s.close()
                except Exception:
                    pass
            if attempt % 5 == 0:
                elapsed = time.monotonic() - start
                self.log.emit(f"  ... 已等待 {elapsed:.0f}s")
            time.sleep(2.0)
        return False


class EngineLauncherDialog(QDialog):
    """One-click dialog: set engine path → launch → install plugin → connect."""

    # Emitted when launch+connect succeeds, carries (host, port) for bridge
    engine_ready = pyqtSignal(str, int)

    def __init__(self, parent=None, config: dict = None):
        super().__init__(parent)
        self.setWindowTitle("一键启动引擎")
        self.setMinimumSize(600, 480)
        self._worker = None
        self._config = config or {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        bridge_cfg = self._config.get('bridge', {})

        # --- Description ---
        desc = QLabel(
            "指定 Messiah 引擎目录，一键完成：\n"
            "  1. 安装 Optimizer Bridge 插件\n"
            "  2. 启动 Messiah Editor\n"
            "  3. 等待引擎 RPC 端口就绪并自动连接"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # --- Engine Root ---
        path_group = QGroupBox("引擎路径")
        path_form = QFormLayout()

        row = QHBoxLayout()
        self.path_edit = QLineEdit(bridge_cfg.get('engine_root', ''))
        self.path_edit.setPlaceholderText("例如: D:\\NewTrunk\\Engine\\src\\Engine")
        row.addWidget(self.path_edit)
        browse_btn = QPushButton("浏览...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_engine_root)
        row.addWidget(browse_btn)
        path_form.addRow("引擎根目录:", row)

        self.bat_edit = QLineEdit(bridge_cfg.get('engine_bat', 'Messiah_Editor.bat'))
        self.bat_edit.setPlaceholderText("Messiah_Editor.bat")
        path_form.addRow("启动脚本 (bat):", self.bat_edit)

        path_group.setLayout(path_form)
        layout.addWidget(path_group)

        # --- Options ---
        opt_group = QGroupBox("选项")
        opt_layout = QFormLayout()

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(bridge_cfg.get('port', 9800))
        opt_layout.addRow("RPC 端口:", self.port_spin)

        self.install_cb = QCheckBox("自动安装 / 更新插件")
        self.install_cb.setChecked(bridge_cfg.get('auto_install_plugin', True))
        opt_layout.addRow(self.install_cb)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(30, 600)
        self.timeout_spin.setValue(180)
        self.timeout_spin.setSuffix(" 秒")
        opt_layout.addRow("连接超时:", self.timeout_spin)

        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)

        # --- Status ---
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.status_label)

        # --- Log ---
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMaximumHeight(180)
        self.log_text.setPlaceholderText("日志输出...")
        layout.addWidget(self.log_text)

        # --- Buttons ---
        btn_layout = QHBoxLayout()

        self.launch_btn = QPushButton("一键启动引擎")
        self.launch_btn.setMinimumHeight(40)
        self.launch_btn.setStyleSheet(
            "QPushButton { background-color: #1f6feb; color: white; "
            "font-weight: bold; font-size: 14px; border-radius: 4px; padding: 8px 20px; }"
            "QPushButton:hover { background-color: #388bfd; }"
            "QPushButton:disabled { background-color: #484f58; }"
        )
        self.launch_btn.clicked.connect(self._on_launch)
        btn_layout.addWidget(self.launch_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setMinimumHeight(40)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.cancel_btn)

        self.close_btn = QPushButton("关闭")
        self.close_btn.setMinimumHeight(40)
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

    # ---- Actions ----

    def _browse_engine_root(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择 Messiah 引擎根目录 (Engine/)",
            self.path_edit.text() or "",
        )
        if path:
            self.path_edit.setText(path)

    def _on_launch(self):
        engine_root = self.path_edit.text().strip()
        if not engine_root:
            QMessageBox.warning(self, "路径为空", "请先选择引擎根目录。")
            return
        if not os.path.isdir(engine_root):
            QMessageBox.warning(self, "路径无效", f"目录不存在:\n{engine_root}")
            return

        # Check Editor/QtScript exists
        qtscript = os.path.join(engine_root, 'Editor', 'QtScript')
        if not os.path.isdir(qtscript):
            QMessageBox.warning(
                self, "路径错误",
                f"未找到 Editor/QtScript/ 目录:\n{qtscript}\n\n"
                "请确认选择的是 Engine 根目录 (如 D:\\NewTrunk\\Engine\\src\\Engine)。"
            )
            return

        self._set_busy(True)
        self.log_text.clear()
        self.status_label.setText("⏳ 启动中...")
        self.status_label.setStyleSheet("font-weight: bold; color: #58a6ff;")

        self._worker = _LaunchWorker(
            engine_root=engine_root,
            engine_bat=self.bat_edit.text().strip() or 'Messiah_Editor.bat',
            port=self.port_spin.value(),
            auto_install=self.install_cb.isChecked(),
            connect_timeout=self.timeout_spin.value(),
        )
        self._worker.log.connect(self._append_log)
        self._worker.stage_changed.connect(
            lambda s: self.status_label.setText(f"⏳ {s}")
        )
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_cancel(self):
        if self._worker:
            self._worker.cancel()
            self._append_log("\n正在取消...")

    def _on_finished(self, success: bool, summary: str):
        self._set_busy(False)
        if success:
            self.status_label.setText(f"✅ {summary}")
            self.status_label.setStyleSheet("font-weight: bold; color: #3fb950;")
            self.engine_ready.emit('127.0.0.1', self.port_spin.value())
        else:
            self.status_label.setText(f"❌ {summary}")
            self.status_label.setStyleSheet("font-weight: bold; color: #f85149;")
        self._worker = None

    def _set_busy(self, busy: bool):
        self.launch_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.path_edit.setEnabled(not busy)
        self.bat_edit.setEnabled(not busy)
        self.port_spin.setEnabled(not busy)

    def _append_log(self, text: str):
        self.log_text.append(text)
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)

    def get_engine_root(self) -> str:
        return self.path_edit.text().strip()

    def get_port(self) -> int:
        return self.port_spin.value()
