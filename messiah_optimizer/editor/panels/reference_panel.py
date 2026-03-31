"""
Reference Panel - Manage reference images for optimization.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QSpinBox, QFileDialog, QGroupBox,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSignal, Qt
import numpy as np


class ReferencePanel(QWidget):
    """Manage reference images used as optimization targets."""

    images_loaded = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_paths = []
        self.image_arrays = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Controls
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("  \u2795  Add")
        self.btn_add.clicked.connect(self._on_add)
        btn_layout.addWidget(self.btn_add)

        self.btn_remove = QPushButton("  \u2796  Remove")
        self.btn_remove.clicked.connect(self._on_remove)
        btn_layout.addWidget(self.btn_remove)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setProperty("cssClass", "danger")
        self.btn_clear.clicked.connect(self._on_clear)
        btn_layout.addWidget(self.btn_clear)
        layout.addLayout(btn_layout)

        # Auto-generate group
        gen_group = QGroupBox("Auto-Generate from Messiah")
        gen_layout = QHBoxLayout(gen_group)
        gen_layout.addWidget(QLabel("Views:"))
        self.spin_views = QSpinBox()
        self.spin_views.setRange(1, 64)
        self.spin_views.setValue(16)
        gen_layout.addWidget(self.spin_views)

        self.btn_generate = QPushButton("\U0001F4F7  Capture from Messiah")
        self.btn_generate.setProperty("cssClass", "accent")
        self.btn_generate.clicked.connect(self._on_generate)
        gen_layout.addWidget(self.btn_generate)
        layout.addWidget(gen_group)

        # Image list
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget)

        # Preview
        self.preview_label = QLabel()
        self.preview_label.setObjectName("compare_reference")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(120)
        layout.addWidget(self.preview_label)

        self.info_label = QLabel("No reference images")
        self.info_label.setProperty("cssClass", "dim")
        layout.addWidget(self.info_label)

    def _on_add(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Reference Images",
            filter="Images (*.png *.jpg *.jpeg *.exr *.hdr *.bmp);;All (*)"
        )
        if paths:
            self.load_images(paths)

    def _on_remove(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.list_widget.takeItem(row)
            self.image_paths.pop(row)
            if row < len(self.image_arrays):
                self.image_arrays.pop(row)
            self._update_info()

    def _on_clear(self):
        self.list_widget.clear()
        self.image_paths.clear()
        self.image_arrays.clear()
        self.preview_label.clear()
        self._update_info()

    def _on_generate(self):
        """Request multi-view captures from Messiah bridge."""
        # This will be connected to the bridge
        pass

    def _on_selection_changed(self, row):
        if 0 <= row < len(self.image_paths):
            pixmap = QPixmap(self.image_paths[row])
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.preview_label.setPixmap(scaled)

    def load_images(self, paths: list):
        """Load reference images from file paths."""
        from PIL import Image

        for path in paths:
            if os.path.exists(path):
                self.image_paths.append(path)
                self.list_widget.addItem(os.path.basename(path))

                # Load as numpy array
                img = Image.open(path).convert('RGB')
                arr = np.array(img, dtype=np.float32) / 255.0
                self.image_arrays.append(arr)

        self._update_info()
        self.images_loaded.emit(self.image_paths)

    def get_reference_tensor(self, device='cuda'):
        """Get all reference images as a batched tensor [N, H, W, 3]."""
        import torch
        if not self.image_arrays:
            return None
        tensors = [torch.from_numpy(a).to(device) for a in self.image_arrays]
        return torch.stack(tensors, dim=0)

    def add_image_from_array(self, image: np.ndarray, name: str = "Reference"):
        """Add a reference image from a numpy uint8 [H, W, 3] array."""
        self.image_paths.append(name)
        self.list_widget.addItem(name)
        arr = image.astype(np.float32) / 255.0
        self.image_arrays.append(arr)
        self._update_info()

    def _update_info(self):
        n = len(self.image_paths)
        if n == 0:
            self.info_label.setText("No reference images")
        else:
            self.info_label.setText(f"{n} reference image(s)")
