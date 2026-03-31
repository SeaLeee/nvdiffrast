"""
RenderDoc Comparison Panel — browse .rdc captures, inspect passes,
extract intermediate framebuffers, and replace textures for A/B comparison.

Workflow:
  1. Load .rdc  →  display action tree
  2. Auto-detect passes  →  highlight pre/post-process boundary
  3. Click any event  →  preview framebuffer
  4. Select texture  →  assign optimized replacement  →  replay & compare
"""

import os
import logging
from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QFileDialog, QTreeWidget, QTreeWidgetItem, QSplitter,
    QComboBox, QTabWidget, QScrollArea, QSizePolicy, QMessageBox,
    QHeaderView, QFormLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QColor, QBrush

logger = logging.getLogger(__name__)


def _numpy_to_qpixmap(arr: np.ndarray, max_size: int = 512) -> QPixmap:
    """Convert a float32/uint8 HWC numpy array to QPixmap."""
    if arr is None:
        return QPixmap()

    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    h, w = arr.shape[:2]
    if arr.ndim == 2:
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
    elif arr.shape[2] == 3:
        bytes_per_line = 3 * w
        qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    elif arr.shape[2] == 4:
        bytes_per_line = 4 * w
        qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
    else:
        return QPixmap()

    pix = QPixmap.fromImage(qimg)
    if max(w, h) > max_size:
        pix = pix.scaled(max_size, max_size,
                         Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    return pix


class RenderDocPanel(QWidget):
    """
    Panel for RenderDoc-based pipeline inspection and texture replacement.
    """

    # Emitted when a framebuffer is extracted (event_id, numpy_image)
    framebuffer_extracted = pyqtSignal(int, object)
    # Emitted when comparison completes (original, replaced, diff)
    comparison_done = pyqtSignal(object, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._replay = None   # RenderDocReplay instance (lazy)
        self._current_rdc = ''
        self._selected_event_id = None
        self._replacement_textures = {}  # resource_id → numpy
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ---- File selector ----
        file_group = QGroupBox("RenderDoc 捕获文件")
        file_layout = QHBoxLayout(file_group)

        self._lbl_rdc = QLabel("未加载")
        self._lbl_rdc.setWordWrap(True)
        file_layout.addWidget(self._lbl_rdc, stretch=1)

        self._btn_open = QPushButton("打开 .rdc...")
        self._btn_open.clicked.connect(self._on_open_rdc)
        file_layout.addWidget(self._btn_open)

        self._btn_close = QPushButton("关闭")
        self._btn_close.setEnabled(False)
        self._btn_close.clicked.connect(self._on_close_rdc)
        file_layout.addWidget(self._btn_close)

        layout.addWidget(file_group)

        # ---- Main splitter: action tree + preview ----
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(True)

        # LEFT: Action tree + pass list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        # Tab: Draw Call Tree
        self._action_tree = QTreeWidget()
        self._action_tree.setHeaderLabels(["Event", "Name", "Type"])
        self._action_tree.setAlternatingRowColors(True)
        self._action_tree.itemClicked.connect(self._on_action_clicked)
        header = self._action_tree.header()
        header.setStretchLastSection(True)
        header.resizeSection(0, 70)
        self._tabs.addTab(self._action_tree, "Draw Calls")

        # Tab: Passes
        self._pass_tree = QTreeWidget()
        self._pass_tree.setHeaderLabels(["Pass", "Events", "Actions", "PostProc?"])
        self._pass_tree.setAlternatingRowColors(True)
        self._pass_tree.itemClicked.connect(self._on_pass_clicked)
        self._tabs.addTab(self._pass_tree, "渲染 Pass")

        # Tab: Textures
        self._tex_tree = QTreeWidget()
        self._tex_tree.setHeaderLabels(["ID", "Name", "Size", "Format", "RT?"])
        self._tex_tree.setAlternatingRowColors(True)
        self._tex_tree.itemClicked.connect(self._on_texture_clicked)
        self._tabs.addTab(self._tex_tree, "贴图资源")

        left_layout.addWidget(self._tabs)

        # Quick actions under tree
        quick_layout = QHBoxLayout()
        self._btn_goto_pre_pp = QPushButton("跳到后处理前")
        self._btn_goto_pre_pp.setToolTip("跳转到后处理开始之前的最后一个事件")
        self._btn_goto_pre_pp.setEnabled(False)
        self._btn_goto_pre_pp.clicked.connect(self._on_goto_pre_postprocess)
        quick_layout.addWidget(self._btn_goto_pre_pp)

        self._btn_extract = QPushButton("提取帧缓冲")
        self._btn_extract.setEnabled(False)
        self._btn_extract.clicked.connect(self._on_extract_framebuffer)
        quick_layout.addWidget(self._btn_extract)

        left_layout.addLayout(quick_layout)
        splitter.addWidget(left)

        # RIGHT: Preview area
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Preview image
        preview_group = QGroupBox("帧缓冲预览")
        preview_layout = QVBoxLayout(preview_group)

        self._preview_label = QLabel("选择一个事件以预览帧缓冲")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(256, 256)
        self._preview_label.setStyleSheet(
            "background: #1e1e1e; border: 1px solid #444; color: #888;"
        )
        preview_layout.addWidget(self._preview_label)

        # Event info
        self._lbl_event_info = QLabel("")
        self._lbl_event_info.setWordWrap(True)
        preview_layout.addWidget(self._lbl_event_info)

        right_layout.addWidget(preview_group)

        # Comparison controls
        cmp_group = QGroupBox("贴图替换 & 对比")
        cmp_layout = QVBoxLayout(cmp_group)

        form = QFormLayout()
        self._cmb_replace_tex = QComboBox()
        self._cmb_replace_tex.setPlaceholderText("选择要替换的贴图...")
        form.addRow("目标贴图:", self._cmb_replace_tex)

        self._lbl_replacement = QLabel("未设置替换图")
        form.addRow("替换源:", self._lbl_replacement)
        cmp_layout.addLayout(form)

        repl_btns = QHBoxLayout()
        self._btn_load_replacement = QPushButton("加载替换贴图...")
        self._btn_load_replacement.clicked.connect(self._on_load_replacement)
        repl_btns.addWidget(self._btn_load_replacement)

        self._btn_use_optimized = QPushButton("使用优化结果")
        self._btn_use_optimized.setToolTip("从当前优化器获取优化后的贴图")
        self._btn_use_optimized.clicked.connect(self._on_use_optimized)
        repl_btns.addWidget(self._btn_use_optimized)
        cmp_layout.addLayout(repl_btns)

        self._btn_compare = QPushButton("  ▶  执行替换对比")
        self._btn_compare.setEnabled(False)
        self._btn_compare.setStyleSheet(
            "QPushButton { background: #0e639c; color: white; "
            "padding: 8px; font-weight: bold; }"
        )
        self._btn_compare.clicked.connect(self._on_run_comparison)
        cmp_layout.addWidget(self._btn_compare)

        # Comparison result preview
        self._comparison_tabs = QTabWidget()
        self._comparison_tabs.setDocumentMode(True)

        self._lbl_original = QLabel("原始")
        self._lbl_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_original.setStyleSheet("background: #1e1e1e;")
        self._comparison_tabs.addTab(self._lbl_original, "原始")

        self._lbl_replaced = QLabel("替换后")
        self._lbl_replaced.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_replaced.setStyleSheet("background: #1e1e1e;")
        self._comparison_tabs.addTab(self._lbl_replaced, "替换后")

        self._lbl_diff = QLabel("差异")
        self._lbl_diff.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_diff.setStyleSheet("background: #1e1e1e;")
        self._comparison_tabs.addTab(self._lbl_diff, "差异 ×10")

        cmp_layout.addWidget(self._comparison_tabs)

        # Metrics
        self._lbl_metrics = QLabel("")
        self._lbl_metrics.setWordWrap(True)
        cmp_layout.addWidget(self._lbl_metrics)

        right_layout.addWidget(cmp_group)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    # ============ RenderDoc Replay Access ============

    def _get_replay(self):
        """Lazy init the RenderDocReplay instance."""
        if self._replay is None:
            from bridge.renderdoc_replay import RenderDocReplay, is_available
            if not is_available():
                QMessageBox.warning(
                    self, "RenderDoc 不可用",
                    "未找到 renderdoc Python 模块 (renderdoc.pyd)。\n\n"
                    "请安装 RenderDoc 或设置 RENDERDOC_PATH 环境变量。"
                )
                return None
            self._replay = RenderDocReplay()
        return self._replay

    # ============ File Operations ============

    def _on_open_rdc(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开 RenderDoc 捕获", "",
            "RenderDoc Captures (*.rdc);;All Files (*)"
        )
        if not path:
            return

        replay = self._get_replay()
        if replay is None:
            return

        if not replay.open(path):
            QMessageBox.critical(
                self, "打开失败",
                f"无法打开捕获文件:\n{path}"
            )
            return

        self._current_rdc = path
        self._lbl_rdc.setText(os.path.basename(path))
        self._btn_close.setEnabled(True)
        self._btn_goto_pre_pp.setEnabled(True)
        self._btn_extract.setEnabled(True)

        self._refresh_action_tree()
        self._refresh_pass_list()
        self._refresh_texture_list()

    def _on_close_rdc(self):
        if self._replay:
            self._replay.close()
        self._current_rdc = ''
        self._lbl_rdc.setText("未加载")
        self._btn_close.setEnabled(False)
        self._btn_goto_pre_pp.setEnabled(False)
        self._btn_extract.setEnabled(False)
        self._btn_compare.setEnabled(False)
        self._action_tree.clear()
        self._pass_tree.clear()
        self._tex_tree.clear()
        self._cmb_replace_tex.clear()
        self._preview_label.setText("选择一个事件以预览帧缓冲")
        self._preview_label.setPixmap(QPixmap())

    # ============ Tree Population ============

    def _refresh_action_tree(self):
        self._action_tree.clear()
        if not self._replay or not self._replay.is_open:
            return

        actions = self._replay.get_actions()
        post_event = self._replay.find_pre_postprocess_event()

        def _add_items(parent, nodes):
            for n in nodes:
                item = QTreeWidgetItem()
                item.setText(0, str(n.event_id))
                item.setText(1, n.name)

                if n.is_draw:
                    tp = "Draw"
                elif n.is_clear:
                    tp = "Clear"
                elif n.is_present:
                    tp = "Present"
                elif n.is_marker:
                    tp = "Marker"
                else:
                    tp = ""
                item.setText(2, tp)
                item.setData(0, Qt.ItemDataRole.UserRole, n.event_id)

                # Highlight post-processing boundary
                if post_event and n.event_id == post_event:
                    for col in range(3):
                        item.setBackground(col, QBrush(QColor(80, 180, 80, 60)))

                if parent is None:
                    self._action_tree.addTopLevelItem(item)
                else:
                    parent.addChild(item)

                if n.children:
                    _add_items(item, n.children)

        _add_items(None, actions)
        self._action_tree.expandToDepth(1)

    def _refresh_pass_list(self):
        self._pass_tree.clear()
        if not self._replay or not self._replay.is_open:
            return

        passes = self._replay.identify_passes()
        for p in passes:
            item = QTreeWidgetItem()
            item.setText(0, p.name or "(unnamed)")
            item.setText(1, f"{p.event_start} - {p.event_end}")
            item.setText(2, str(p.action_count))
            item.setText(3, "✓" if p.is_postprocess else "")
            item.setData(0, Qt.ItemDataRole.UserRole, p.event_end)

            if p.is_postprocess:
                for col in range(4):
                    item.setBackground(col, QBrush(QColor(200, 120, 50, 40)))

            self._pass_tree.addTopLevelItem(item)

    def _refresh_texture_list(self):
        self._tex_tree.clear()
        self._cmb_replace_tex.clear()
        if not self._replay or not self._replay.is_open:
            return

        textures = self._replay.get_textures()
        for t in textures:
            item = QTreeWidgetItem()
            item.setText(0, str(t.resource_id))
            item.setText(1, t.name)
            item.setText(2, f"{t.width}×{t.height}")
            item.setText(3, t.format_str)
            item.setText(4, "RT" if t.is_render_target else
                         ("D" if t.is_depth else ""))
            item.setData(0, Qt.ItemDataRole.UserRole, t.resource_id)
            self._tex_tree.addTopLevelItem(item)

            # Only non-RT, non-depth textures are replaceable
            if not t.is_render_target and not t.is_depth:
                label = f"{t.name} ({t.width}×{t.height})" if t.name else \
                    f"ID:{t.resource_id} ({t.width}×{t.height})"
                self._cmb_replace_tex.addItem(label, t.resource_id)

    # ============ Event Selection & Preview ============

    def _on_action_clicked(self, item: QTreeWidgetItem, col: int):
        event_id = item.data(0, Qt.ItemDataRole.UserRole)
        if event_id is not None:
            self._selected_event_id = event_id
            self._update_event_info(event_id)
            self._update_preview(event_id)
            self._btn_compare.setEnabled(True)

    def _on_pass_clicked(self, item: QTreeWidgetItem, col: int):
        event_id = item.data(0, Qt.ItemDataRole.UserRole)
        if event_id is not None:
            self._selected_event_id = event_id
            self._update_event_info(event_id)
            self._update_preview(event_id)
            self._btn_compare.setEnabled(True)

    def _on_texture_clicked(self, item: QTreeWidgetItem, col: int):
        resource_id = item.data(0, Qt.ItemDataRole.UserRole)
        if resource_id is not None:
            self._preview_texture(resource_id)

    def _on_goto_pre_postprocess(self):
        if not self._replay or not self._replay.is_open:
            return

        event_id = self._replay.find_pre_postprocess_event()
        if event_id is None:
            QMessageBox.information(self, "提示", "未能自动识别后处理边界")
            return

        self._selected_event_id = event_id
        self._update_event_info(event_id)
        self._update_preview(event_id)
        self._btn_compare.setEnabled(True)

        # Select the matching item in the action tree
        self._select_event_in_tree(event_id)

    def _on_extract_framebuffer(self):
        if self._selected_event_id is None:
            QMessageBox.information(self, "提示", "请先选择一个事件")
            return

        fb = self._replay.extract_framebuffer(self._selected_event_id)
        if fb is not None:
            self.framebuffer_extracted.emit(self._selected_event_id, fb)
            path, _ = QFileDialog.getSaveFileName(
                self, "保存帧缓冲", f"framebuffer_event{self._selected_event_id}.png",
                "PNG (*.png);;HDR (*.hdr);;EXR (*.exr)"
            )
            if path:
                self._replay.save_framebuffer(
                    self._selected_event_id, path
                )
                logger.info(f"Framebuffer saved to {path}")
        else:
            QMessageBox.warning(self, "提取失败", "无法提取选中事件的帧缓冲")

    def _update_event_info(self, event_id: int):
        if not self._replay or not self._replay.is_open:
            return

        info = self._replay.get_pipeline_info_at_event(event_id)
        lines = [f"Event ID: {event_id}"]
        lines.append(f"输出数: {info.get('output_count', '?')}")
        lines.append(f"深度: {'有' if info.get('has_depth') else '无'}")

        for stage in ['vertex', 'pixel', 'geometry', 'compute']:
            key = f'{stage}_shader'
            if key in info:
                lines.append(f"{stage} shader: {info[key].get('entry_point', 'N/A')}")

        self._lbl_event_info.setText('\n'.join(lines))

    def _update_preview(self, event_id: int):
        if not self._replay or not self._replay.is_open:
            return

        fb = self._replay.extract_framebuffer(event_id)
        if fb is not None:
            pix = _numpy_to_qpixmap(fb, max_size=480)
            self._preview_label.setPixmap(pix)
            self._preview_label.setText("")
        else:
            self._preview_label.setText(f"Event {event_id}: 无可用帧缓冲")
            self._preview_label.setPixmap(QPixmap())

    def _preview_texture(self, resource_id: int):
        if not self._replay or not self._replay.is_open:
            return

        img = self._replay._read_texture_as_numpy(resource_id)
        if img is not None:
            pix = _numpy_to_qpixmap(img, max_size=480)
            self._preview_label.setPixmap(pix)
            self._preview_label.setText("")
        else:
            self._preview_label.setText(f"无法预览贴图 {resource_id}")
            self._preview_label.setPixmap(QPixmap())

    def _select_event_in_tree(self, event_id: int):
        """Select and scroll to an event in the action tree."""
        it = self._action_tree.invisibleRootItem()

        def _search(parent):
            for i in range(parent.childCount()):
                child = parent.child(i)
                if child.data(0, Qt.ItemDataRole.UserRole) == event_id:
                    self._action_tree.setCurrentItem(child)
                    self._action_tree.scrollToItem(child)
                    return True
                if child.childCount() > 0 and _search(child):
                    return True
            return False

        _search(it)

    # ============ Texture Replacement & Comparison ============

    def _on_load_replacement(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择替换贴图", "",
            "Images (*.png *.jpg *.tga *.bmp *.hdr *.exr);;All Files (*)"
        )
        if not path:
            return

        try:
            from PIL import Image
            img = Image.open(path).convert('RGBA')
            arr = np.array(img, dtype=np.float32) / 255.0
            tex_id = self._cmb_replace_tex.currentData()
            if tex_id is not None:
                self._replacement_textures[tex_id] = arr
                self._lbl_replacement.setText(
                    f"{os.path.basename(path)} ({img.width}×{img.height})"
                )
                self._btn_compare.setEnabled(True)
            else:
                QMessageBox.information(self, "提示", "请先选择目标贴图")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def _on_use_optimized(self):
        """Get the optimized texture from the main optimizer window."""
        # This will be connected externally by main_window.py
        self.use_optimized_requested = True

    def set_optimized_texture(self, resource_id: int, image: np.ndarray):
        """Set optimized texture data from the optimizer."""
        self._replacement_textures[resource_id] = image
        self._lbl_replacement.setText(
            f"优化结果 ({image.shape[1]}×{image.shape[0]})"
        )
        # Select this resource in the combo
        for i in range(self._cmb_replace_tex.count()):
            if self._cmb_replace_tex.itemData(i) == resource_id:
                self._cmb_replace_tex.setCurrentIndex(i)
                break
        self._btn_compare.setEnabled(True)

    def _on_run_comparison(self):
        if not self._replay or not self._replay.is_open:
            return

        tex_id = self._cmb_replace_tex.currentData()
        if tex_id is None or tex_id not in self._replacement_textures:
            # No replacement texture → just extract the framebuffer comparison
            # between current event and pre-post-process
            if self._selected_event_id is not None:
                fb = self._replay.extract_framebuffer(self._selected_event_id)
                if fb is not None:
                    pix = _numpy_to_qpixmap(fb, max_size=400)
                    self._lbl_original.setPixmap(pix)
                    self._lbl_original.setText("")
            return

        replacement = self._replacement_textures[tex_id]
        compare_event = self._selected_event_id

        result = self._replay.replace_texture_and_compare(
            tex_id, replacement, compare_event
        )

        if result.success:
            # Show original
            if result.original_image is not None:
                pix = _numpy_to_qpixmap(result.original_image, max_size=400)
                self._lbl_original.setPixmap(pix)
                self._lbl_original.setText("")

            # Show replaced
            if result.replaced_image is not None:
                pix = _numpy_to_qpixmap(result.replaced_image, max_size=400)
                self._lbl_replaced.setPixmap(pix)
                self._lbl_replaced.setText("")

            # Show diff (amplified ×10)
            if result.diff_image is not None:
                diff_vis = np.clip(result.diff_image * 10, 0, 1)
                pix = _numpy_to_qpixmap(diff_vis, max_size=400)
                self._lbl_diff.setPixmap(pix)
                self._lbl_diff.setText("")

            # Metrics
            self._lbl_metrics.setText(
                f"MSE: {result.mse:.6f}  |  PSNR: {result.psnr:.2f} dB  |  "
                f"Event: {result.event_id}"
            )

            self.comparison_done.emit(
                result.original_image,
                result.replaced_image,
                result.diff_image,
            )
        else:
            QMessageBox.warning(
                self, "对比失败",
                f"替换对比未能完成:\n{result.error}"
            )
            self._lbl_metrics.setText(f"错误: {result.error}")

    # ============ Public API ============

    def load_rdc(self, path: str) -> bool:
        """Programmatically open a .rdc capture file."""
        replay = self._get_replay()
        if replay is None:
            return False

        if not replay.open(path):
            return False

        self._current_rdc = path
        self._lbl_rdc.setText(os.path.basename(path))
        self._btn_close.setEnabled(True)
        self._btn_goto_pre_pp.setEnabled(True)
        self._btn_extract.setEnabled(True)

        self._refresh_action_tree()
        self._refresh_pass_list()
        self._refresh_texture_list()
        return True

    def get_pre_postprocess_image(self) -> Optional[np.ndarray]:
        """Extract the framebuffer just before post-processing."""
        if not self._replay or not self._replay.is_open:
            return None

        event_id = self._replay.find_pre_postprocess_event()
        if event_id is None:
            return None

        return self._replay.extract_framebuffer(event_id)

    def get_passes(self) -> list:
        """Get the identified rendering passes."""
        if not self._replay or not self._replay.is_open:
            return []
        return self._replay.identify_passes()

    def cleanup(self):
        """Release resources on shutdown."""
        if self._replay:
            self._replay.shutdown()
            self._replay = None
