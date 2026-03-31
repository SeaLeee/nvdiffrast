"""
Iteration History Panel — visualize optimization progress step by step.

Shows:
  - Texture evolution gallery (how textures change over iterations)
  - Side-by-side: engine reference vs current nvdiffrast render
  - Shader parameter diff (before → after)
  - Loss / PSNR convergence sparkline
"""

import numpy as np
from collections import deque, OrderedDict
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QGridLayout, QScrollArea, QSplitter, QComboBox, QSlider,
    QPushButton, QSizePolicy, QTabWidget, QTextEdit, QFrame,
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPainter


class IterationHistoryPanel(QWidget):
    """Panel showing the optimization iteration history and evolution."""

    # Emitted when user clicks a snapshot thumbnail to view full-size
    snapshot_selected = pyqtSignal(int)  # iteration number

    def __init__(self, parent=None, max_snapshots: int = 100):
        super().__init__(parent)
        self._max_snapshots = max_snapshots

        # --- Data ---
        # Texture snapshots: list of (iteration, numpy_image_HWC_uint8)
        self._texture_snapshots: list[tuple[int, np.ndarray]] = []
        # Render snapshots: list of (iteration, rendered_HWC, reference_HWC)
        self._render_snapshots: list[tuple[int, np.ndarray, np.ndarray | None]] = []
        # Shader parameter history: list of (iteration, params_dict)
        self._shader_history: list[tuple[int, dict]] = []
        # Initial shader params (baseline for diff)
        self._shader_baseline: dict | None = None
        # Engine reference image (from MPlatform.ScreenShot)
        self._engine_reference: np.ndarray | None = None
        # Loss trace for sparkline
        self._loss_trace = deque(maxlen=2000)
        self._psnr_trace = deque(maxlen=2000)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        # --- Tab 1: Texture Evolution ---
        self._tex_tab = QWidget()
        self._setup_texture_tab(self._tex_tab)
        self._tabs.addTab(self._tex_tab, "贴图演变")

        # --- Tab 2: Render Comparison ---
        self._cmp_tab = QWidget()
        self._setup_compare_tab(self._cmp_tab)
        self._tabs.addTab(self._cmp_tab, "渲染对比")

        # --- Tab 3: Shader Diff ---
        self._diff_tab = QWidget()
        self._setup_shader_diff_tab(self._diff_tab)
        self._tabs.addTab(self._diff_tab, "Shader Diff")

        layout.addWidget(self._tabs)

    # ================================================================
    # Tab 1: Texture Evolution Gallery
    # ================================================================

    def _setup_texture_tab(self, parent: QWidget):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(2, 2, 2, 2)

        # Controls bar
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("迭代步:"))
        self._tex_slider = QSlider(Qt.Orientation.Horizontal)
        self._tex_slider.setRange(0, 0)
        self._tex_slider.valueChanged.connect(self._on_tex_slider_changed)
        ctrl.addWidget(self._tex_slider, 1)
        self._tex_iter_label = QLabel("0 / 0")
        self._tex_iter_label.setMinimumWidth(80)
        ctrl.addWidget(self._tex_iter_label)
        layout.addLayout(ctrl)

        # Thumbnail gallery (scrollable horizontal strip)
        self._gallery_scroll = QScrollArea()
        self._gallery_scroll.setWidgetResizable(True)
        self._gallery_scroll.setMaximumHeight(120)
        self._gallery_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._gallery_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._gallery_container = QWidget()
        self._gallery_layout = QHBoxLayout(self._gallery_container)
        self._gallery_layout.setContentsMargins(2, 2, 2, 2)
        self._gallery_layout.setSpacing(4)
        self._gallery_layout.addStretch()
        self._gallery_scroll.setWidget(self._gallery_container)
        layout.addWidget(self._gallery_scroll)

        # Large preview of selected iteration
        self._tex_preview = QLabel("加载场景并开始优化后，贴图演变过程将显示在此处")
        self._tex_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tex_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._tex_preview.setMinimumSize(200, 200)
        self._tex_preview.setStyleSheet(
            "QLabel { background: #0d1117; border: 1px solid #30363d; }")
        layout.addWidget(self._tex_preview, 1)

        # Sparkline for loss below
        self._sparkline_label = QLabel("")
        self._sparkline_label.setMaximumHeight(40)
        layout.addWidget(self._sparkline_label)

    def _on_tex_slider_changed(self, idx: int):
        if 0 <= idx < len(self._texture_snapshots):
            iteration, img = self._texture_snapshots[idx]
            self._tex_iter_label.setText(
                f"{iteration} / {self._texture_snapshots[-1][0] if self._texture_snapshots else 0}")
            self._show_image(self._tex_preview, img)
            self.snapshot_selected.emit(iteration)

    # ================================================================
    # Tab 2: Render Comparison (Engine ref vs Optimizer render)
    # ================================================================

    def _setup_compare_tab(self, parent: QWidget):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(2, 2, 2, 2)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("对比模式:"))
        self._cmp_mode = QComboBox()
        self._cmp_mode.addItems([
            "并排对比", "交叉叠加", "差异放大 (5x)", "仅引擎参考",
        ])
        self._cmp_mode.currentIndexChanged.connect(self._update_compare_display)
        ctrl.addWidget(self._cmp_mode)

        self._cmp_blend_slider = QSlider(Qt.Orientation.Horizontal)
        self._cmp_blend_slider.setRange(0, 100)
        self._cmp_blend_slider.setValue(50)
        self._cmp_blend_slider.valueChanged.connect(self._update_compare_display)
        ctrl.addWidget(QLabel("混合:"))
        ctrl.addWidget(self._cmp_blend_slider)

        ctrl.addWidget(QLabel("迭代:"))
        self._cmp_slider = QSlider(Qt.Orientation.Horizontal)
        self._cmp_slider.setRange(0, 0)
        self._cmp_slider.valueChanged.connect(self._on_cmp_slider_changed)
        ctrl.addWidget(self._cmp_slider, 1)
        self._cmp_iter_label = QLabel("0")
        ctrl.addWidget(self._cmp_iter_label)
        layout.addLayout(ctrl)

        # Two-image display
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._cmp_left = QLabel("引擎渲染 (参考)")
        self._cmp_left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cmp_left.setMinimumSize(160, 160)
        self._cmp_left.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._cmp_left.setStyleSheet(
            "QLabel { background: #0d1117; border: 1px solid #30363d; }")

        self._cmp_right = QLabel("可微渲染 (当前)")
        self._cmp_right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cmp_right.setMinimumSize(160, 160)
        self._cmp_right.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._cmp_right.setStyleSheet(
            "QLabel { background: #0d1117; border: 1px solid #30363d; }")

        splitter.addWidget(self._cmp_left)
        splitter.addWidget(self._cmp_right)
        layout.addWidget(splitter, 1)

        # Metrics
        self._cmp_metrics = QLabel("")
        self._cmp_metrics.setStyleSheet("color: #8b949e;")
        layout.addWidget(self._cmp_metrics)

    def _on_cmp_slider_changed(self, idx: int):
        if 0 <= idx < len(self._render_snapshots):
            iteration, _, _ = self._render_snapshots[idx]
            self._cmp_iter_label.setText(str(iteration))
            self._update_compare_display()

    def _update_compare_display(self):
        idx = self._cmp_slider.value()
        if idx < 0 or idx >= len(self._render_snapshots):
            return

        iteration, rendered, ref_or_none = self._render_snapshots[idx]
        ref = ref_or_none if ref_or_none is not None else self._engine_reference
        if ref is None:
            self._cmp_left.setText("无引擎参考图\n请通过 Bridge 菜单获取引擎截图")
            self._show_image(self._cmp_right, rendered)
            return

        mode = self._cmp_mode.currentIndex()
        blend = self._cmp_blend_slider.value() / 100.0

        # Resize to match
        h = min(ref.shape[0], rendered.shape[0])
        w = min(ref.shape[1], rendered.shape[1])
        ref_r = ref[:h, :w]
        ren_r = rendered[:h, :w]

        if mode == 0:  # Side by side
            self._show_image(self._cmp_left, ref_r)
            self._show_image(self._cmp_right, ren_r)
        elif mode == 1:  # Overlay blend
            blended = (ref_r.astype(np.float32) * (1 - blend) +
                       ren_r.astype(np.float32) * blend).clip(0, 255).astype(np.uint8)
            self._show_image(self._cmp_left, ref_r)
            self._show_image(self._cmp_right, blended)
        elif mode == 2:  # Difference 5x
            diff = np.abs(ref_r.astype(np.float32) - ren_r.astype(np.float32))
            diff = (diff * 5.0).clip(0, 255).astype(np.uint8)
            self._show_image(self._cmp_left, ref_r)
            self._show_image(self._cmp_right, diff)
        elif mode == 3:  # Reference only
            self._show_image(self._cmp_left, ref_r)
            self._cmp_right.setText("—")

        # Compute metrics
        mse = np.mean((ref_r.astype(np.float32) - ren_r.astype(np.float32)) ** 2)
        psnr = -10 * np.log10(mse / (255.0 ** 2) + 1e-10)
        self._cmp_metrics.setText(
            f"Iter {iteration}  |  MSE: {mse:.1f}  |  PSNR: {psnr:.1f} dB  |  "
            f"Δ均值: {np.mean(np.abs(ref_r.astype(float) - ren_r.astype(float))):.1f}")

    # ================================================================
    # Tab 3: Shader Parameter Diff
    # ================================================================

    def _setup_shader_diff_tab(self, parent: QWidget):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(2, 2, 2, 2)

        # Iteration selector
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("迭代:"))
        self._diff_slider = QSlider(Qt.Orientation.Horizontal)
        self._diff_slider.setRange(0, 0)
        self._diff_slider.valueChanged.connect(self._update_shader_diff)
        ctrl.addWidget(self._diff_slider, 1)
        self._diff_iter_label = QLabel("0")
        ctrl.addWidget(self._diff_iter_label)
        layout.addLayout(ctrl)

        # Diff display — styled like a code diff
        self._diff_text = QTextEdit()
        self._diff_text.setReadOnly(True)
        self._diff_text.setFont(QFont("Consolas", 10))
        self._diff_text.setStyleSheet(
            "QTextEdit { background: #0d1117; color: #e6edf3; "
            "border: 1px solid #30363d; }")
        layout.addWidget(self._diff_text, 1)

        # Summary label
        self._diff_summary = QLabel("开始 Shader Simplification 优化后，参数变化将显示在此处")
        self._diff_summary.setStyleSheet("color: #8b949e;")
        layout.addWidget(self._diff_summary)

    def _update_shader_diff(self):
        idx = self._diff_slider.value()
        if idx < 0 or idx >= len(self._shader_history):
            return

        iteration, current_params = self._shader_history[idx]
        self._diff_iter_label.setText(str(iteration))

        baseline = self._shader_baseline or {}
        lines = self._format_param_diff(baseline, current_params, iteration)
        self._diff_text.setHtml(lines)

        # Summary
        n_changed = sum(
            1 for k in current_params
            if k in baseline and current_params[k] != baseline[k]
        )
        self._diff_summary.setText(
            f"Iter {iteration}: {n_changed} 个参数已变更 / {len(current_params)} 总参数")

    def _format_param_diff(self, before: dict, after: dict,
                           iteration: int) -> str:
        """Format parameter changes as HTML diff (GitHub-style coloring)."""
        html_lines = [
            '<pre style="font-family: Consolas, monospace; font-size: 10pt;">',
            f'<span style="color: #8b949e;">--- Baseline (初始参数)</span>',
            f'<span style="color: #8b949e;">+++ Iteration {iteration} (当前参数)</span>',
            '<span style="color: #8b949e;">────────────────────────────────</span>',
        ]

        all_keys = sorted(set(list(before.keys()) + list(after.keys())))
        for key in all_keys:
            old_val = before.get(key)
            new_val = after.get(key)

            if old_val is None:
                # New parameter
                html_lines.append(
                    f'<span style="color: #3fb950;">+ {key}: {self._fmt_val(new_val)}</span>')
            elif new_val is None:
                # Removed parameter
                html_lines.append(
                    f'<span style="color: #f85149;">- {key}: {self._fmt_val(old_val)}</span>')
            elif self._vals_differ(old_val, new_val):
                # Changed
                delta = self._compute_delta(old_val, new_val)
                html_lines.append(
                    f'<span style="color: #f85149;">- {key}: {self._fmt_val(old_val)}</span>')
                html_lines.append(
                    f'<span style="color: #3fb950;">+ {key}: {self._fmt_val(new_val)}'
                    f'  <span style="color: #d29922;">({delta})</span></span>')
            else:
                # Unchanged
                html_lines.append(
                    f'<span style="color: #484f58;">  {key}: {self._fmt_val(new_val)}</span>')

        html_lines.append('</pre>')
        return '\n'.join(html_lines)

    @staticmethod
    def _fmt_val(v) -> str:
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    @staticmethod
    def _vals_differ(a, b) -> bool:
        if isinstance(a, float) and isinstance(b, float):
            return abs(a - b) > 1e-6
        return a != b

    @staticmethod
    def _compute_delta(old, new) -> str:
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            d = new - old
            pct = (d / abs(old) * 100) if abs(old) > 1e-8 else 0
            return f"{'+'if d>=0 else ''}{d:.4f}, {pct:+.1f}%"
        return "changed"

    # ================================================================
    # Public API — called by main_window during optimization
    # ================================================================

    def set_engine_reference(self, image: np.ndarray):
        """Set the engine reference image (from MPlatform.ScreenShot).
        
        Args:
            image: [H, W, 3] uint8 numpy array
        """
        self._engine_reference = image
        self._show_image(self._cmp_left, image)

    def add_texture_snapshot(self, iteration: int, texture_image: np.ndarray):
        """Record a texture state at the given iteration.
        
        Args:
            iteration: current optimization iteration
            texture_image: [H, W, 3] uint8 numpy array of the texture
        """
        if len(self._texture_snapshots) >= self._max_snapshots:
            # Keep first, last, and evenly spaced middle snapshots
            self._thin_snapshots()

        self._texture_snapshots.append((iteration, texture_image))

        # Update slider
        self._tex_slider.setRange(0, len(self._texture_snapshots) - 1)
        self._tex_slider.setValue(len(self._texture_snapshots) - 1)

        # Add thumbnail to gallery
        self._add_thumbnail(iteration, texture_image)

    def add_render_snapshot(self, iteration: int, rendered: np.ndarray,
                            reference: np.ndarray = None):
        """Record a rendered image at the given iteration.
        
        Args:
            iteration: current optimization iteration
            rendered: [H, W, 3] uint8 numpy array of the nvdiffrast render
            reference: optional [H, W, 3] uint8 engine reference for this view
        """
        self._render_snapshots.append((iteration, rendered, reference))
        self._cmp_slider.setRange(0, len(self._render_snapshots) - 1)
        self._cmp_slider.setValue(len(self._render_snapshots) - 1)

    def add_shader_params(self, iteration: int, params: dict):
        """Record shader parameters at the given iteration.
        
        Args:
            iteration: current optimization iteration
            params: dict of parameter_name → value
        """
        if self._shader_baseline is None:
            self._shader_baseline = dict(params)

        self._shader_history.append((iteration, dict(params)))
        self._diff_slider.setRange(0, len(self._shader_history) - 1)
        self._diff_slider.setValue(len(self._shader_history) - 1)

    def add_loss_point(self, iteration: int, loss: float, psnr: float):
        """Add a loss/PSNR data point for the sparkline."""
        self._loss_trace.append((iteration, loss))
        self._psnr_trace.append((iteration, psnr))

        # Update sparkline every 20 iterations
        if iteration % 20 == 0:
            self._draw_sparkline()

    def reset(self):
        """Clear all history data."""
        self._texture_snapshots.clear()
        self._render_snapshots.clear()
        self._shader_history.clear()
        self._shader_baseline = None
        self._loss_trace.clear()
        self._psnr_trace.clear()

        self._tex_slider.setRange(0, 0)
        self._cmp_slider.setRange(0, 0)
        self._diff_slider.setRange(0, 0)

        # Clear gallery
        while self._gallery_layout.count() > 1:  # keep stretch
            item = self._gallery_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._tex_preview.setText("加载场景并开始优化后，贴图演变过程将显示在此处")
        self._diff_text.clear()
        self._cmp_left.setText("引擎渲染 (参考)")
        self._cmp_right.setText("可微渲染 (当前)")
        self._sparkline_label.clear()

    # ================================================================
    # Private helpers
    # ================================================================

    def _show_image(self, label: QLabel, img: np.ndarray):
        """Display numpy [H, W, 3] uint8 image on a QLabel."""
        if img is None or img.size == 0:
            return
        h, w = img.shape[:2]
        c = img.shape[2] if img.ndim == 3 else 1
        if c == 3:
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        elif c == 4:
            qimg = QImage(img.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
        else:
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(
            label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _add_thumbnail(self, iteration: int, img: np.ndarray):
        """Add a small clickable thumbnail to the gallery strip."""
        thumb_size = 80
        h, w = img.shape[:2]
        # Simple center-crop + resize for thumbnail
        scale = thumb_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        thumb_label = QLabel()
        thumb_label.setFixedSize(thumb_size, thumb_size)
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_label.setToolTip(f"Iteration {iteration}")
        thumb_label.setStyleSheet(
            "QLabel { border: 1px solid #30363d; background: #161b22; }")

        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            thumb_size, thumb_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        thumb_label.setPixmap(pixmap)

        # Insert before the stretch
        idx = self._gallery_layout.count() - 1
        self._gallery_layout.insertWidget(idx, thumb_label)

        # Auto-scroll to end
        sb = self._gallery_scroll.horizontalScrollBar()
        sb.setValue(sb.maximum())

    def _thin_snapshots(self):
        """Keep only evenly-spaced snapshots when at capacity."""
        n = len(self._texture_snapshots)
        keep = self._max_snapshots // 2
        step = max(1, n // keep)
        self._texture_snapshots = [
            self._texture_snapshots[i]
            for i in range(0, n, step)
        ]

    def _draw_sparkline(self):
        """Draw a tiny loss/PSNR sparkline on the label."""
        if len(self._loss_trace) < 2:
            return

        w, h = 400, 30
        img = np.full((h, w, 3), 13, dtype=np.uint8)  # #0d1117

        # Normalize loss values to pixel Y
        losses = [p[1] for p in self._loss_trace]
        lo, hi = min(losses), max(losses)
        if hi - lo < 1e-10:
            return
        
        n = len(losses)
        for i in range(1, n):
            x0 = int((i - 1) / (n - 1) * (w - 1))
            x1 = int(i / (n - 1) * (w - 1))
            y0 = h - 1 - int((losses[i - 1] - lo) / (hi - lo) * (h - 3))
            y1 = h - 1 - int((losses[i] - lo) / (hi - lo) * (h - 3))
            y0 = max(0, min(h - 1, y0))
            y1 = max(0, min(h - 1, y1))
            # Simple line (no anti-aliasing, just set pixels)
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for s in range(steps + 1):
                t = s / steps
                px = int(x0 + t * (x1 - x0))
                py = int(y0 + t * (y1 - y0))
                if 0 <= px < w and 0 <= py < h:
                    img[py, px] = [88, 166, 255]  # #58a6ff

        qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._sparkline_label.setPixmap(QPixmap.fromImage(qimg))
