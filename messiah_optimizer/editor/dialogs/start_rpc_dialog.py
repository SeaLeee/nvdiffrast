"""
Dialog that guides the user through starting the RPC server inside the engine.

Since the engine's Python environment only becomes fully active when the user
opens the Python console panel (toolbar 🐍 button), this dialog:
1. Checks if the RPC port is already listening
2. If not, shows step-by-step instructions with a visual guide
3. Polls the port at 2-second intervals
4. Auto-connects when the port becomes available
"""

import socket
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QGroupBox, QFormLayout,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont


class StartRpcDialog(QDialog):
    """Guide user to start the engine RPC server and auto-detect it."""

    # Emitted when RPC port is detected as listening
    rpc_ready = pyqtSignal(str, int)  # host, port

    def __init__(self, parent=None, port: int = 9800):
        super().__init__(parent)
        self.setWindowTitle("启动引擎 RPC 服务")
        self.setMinimumSize(520, 400)
        self._port = port
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(2000)
        self._poll_timer.timeout.connect(self._check_port)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # --- Port ---
        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("RPC 端口:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(self._port)
        port_row.addWidget(self.port_spin)
        port_row.addStretch()
        layout.addLayout(port_row)

        # --- Steps ---
        steps_group = QGroupBox("操作步骤")
        steps_layout = QVBoxLayout()

        step_font = QFont()
        step_font.setPointSize(10)

        steps = [
            "① 确保 Messiah 引擎编辑器已启动并完全加载",
            "② 确保已安装 Optimizer 插件\n"
            "    （通过菜单 Messiah Bridge → 安装引擎端插件）",
            "③ 在引擎编辑器工具栏中，点击 Python 按钮 🐍\n"
            "    （工具提示: \"打开编辑器Python\"）",
            "④ 点击后，引擎会初始化 Python 环境并自动启动 RPC 服务",
        ]
        for text in steps:
            label = QLabel(text)
            label.setFont(step_font)
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.PlainText)
            steps_layout.addWidget(label)

        steps_group.setLayout(steps_layout)
        layout.addWidget(steps_group)

        # --- Tip ---
        tip = QLabel(
            "💡 提示：Python 按钮在编辑器场景视图工具栏右侧，\n"
            "图标为蛇形 🐍，鼠标悬停显示 \"Editor Python|打开编辑器Python\"。\n"
            "点击后会弹出一个 Python 脚本窗口，此时 RPC 服务会自动启动。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #8b949e; padding: 4px;")
        layout.addWidget(tip)

        # --- Status ---
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # --- Buttons ---
        btn_layout = QHBoxLayout()

        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.setMinimumHeight(38)
        self.detect_btn.setStyleSheet(
            "QPushButton { background-color: #1f6feb; color: white; "
            "font-weight: bold; font-size: 13px; border-radius: 4px; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #388bfd; }"
            "QPushButton:disabled { background-color: #484f58; }"
        )
        self.detect_btn.clicked.connect(self._on_detect)
        btn_layout.addWidget(self.detect_btn)

        self.close_btn = QPushButton("关闭")
        self.close_btn.setMinimumHeight(38)
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

    # ---- Actions ----

    def _on_detect(self):
        # Check once immediately
        if self._try_connect():
            return
        # Start polling
        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("检测中...")
        self.status_label.setText("⏳ 正在检测端口，请在引擎中点击 🐍 按钮...")
        self.status_label.setStyleSheet(
            "font-weight: bold; font-size: 12px; color: #58a6ff;"
        )
        self._poll_timer.start()

    def _check_port(self):
        self._try_connect()

    def _try_connect(self) -> bool:
        port = self.port_spin.value()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect(('127.0.0.1', port))
            s.close()
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False

        # Success
        self._poll_timer.stop()
        self.detect_btn.setEnabled(True)
        self.detect_btn.setText("已连接 ✓")
        self.detect_btn.setEnabled(False)
        self.status_label.setText(f"✅ 引擎 RPC 服务已就绪 (端口 {port})")
        self.status_label.setStyleSheet(
            "font-weight: bold; font-size: 12px; color: #3fb950;"
        )
        self.rpc_ready.emit('127.0.0.1', port)
        return True

    def closeEvent(self, event):
        self._poll_timer.stop()
        super().closeEvent(event)
