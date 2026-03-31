"""
Image Compare Widget - Side-by-side, overlay, and diff views for
comparing rendered output vs reference.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QSplitter, QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


class ImageCompareWidget(QWidget):
    """Compare rendered vs reference images with multiple view modes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rendered = None   # numpy [H, W, 3] uint8
        self._reference = None  # numpy [H, W, 3] uint8
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # Mode selector
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Compare Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Side by Side", "Overlay Blend", "Difference", "Reference Only"
        ])
        self.mode_combo.currentIndexChanged.connect(self._update_display)
        top_bar.addWidget(self.mode_combo)

        self.blend_slider = QSlider(Qt.Orientation.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.blend_slider.setValue(50)
        self.blend_slider.valueChanged.connect(self._update_display)
        top_bar.addWidget(QLabel("Blend:"))
        top_bar.addWidget(self.blend_slider)
        layout.addLayout(top_bar)

        # Image display area
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.lbl_rendered = QLabel("Rendered")
        self.lbl_rendered.setObjectName("compare_rendered")
        self.lbl_rendered.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_rendered.setMinimumSize(160, 160)
        self.lbl_rendered.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.lbl_reference = QLabel("Reference")
        self.lbl_reference.setObjectName("compare_reference")
        self.lbl_reference.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_reference.setMinimumSize(160, 160)
        self.lbl_reference.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.splitter.addWidget(self.lbl_rendered)
        self.splitter.addWidget(self.lbl_reference)
        layout.addWidget(self.splitter)

        # Info
        self.info_label = QLabel("")
        self.info_label.setProperty("cssClass", "dim")
        layout.addWidget(self.info_label)

    def set_rendered(self, image):
        """Set rendered image. Accepts tensor [B,H,W,3] or numpy [H,W,3]."""
        import torch
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            image = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        self._rendered = image
        self._update_display()

    def set_reference(self, image):
        """Set reference image. Accepts tensor or numpy."""
        import torch
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            image = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        self._reference = image
        self._update_display()

    def _update_display(self):
        mode = self.mode_combo.currentIndex()

        if mode == 0:  # Side by side
            self._show_side_by_side()
        elif mode == 1:  # Overlay blend
            self._show_blend()
        elif mode == 2:  # Difference
            self._show_diff()
        elif mode == 3:  # Reference only
            self._show_single(self._reference, self.lbl_rendered)

    def _show_side_by_side(self):
        self._show_single(self._rendered, self.lbl_rendered)
        self._show_single(self._reference, self.lbl_reference)

    def _show_blend(self):
        if self._rendered is None or self._reference is None:
            return
        alpha = self.blend_slider.value() / 100.0

        # Resize to match if needed
        h = min(self._rendered.shape[0], self._reference.shape[0])
        w = min(self._rendered.shape[1], self._reference.shape[1])
        r = self._rendered[:h, :w]
        ref = self._reference[:h, :w]

        blended = (r.astype(float) * alpha + ref.astype(float) * (1 - alpha))
        blended = blended.clip(0, 255).astype(np.uint8)
        self._show_single(blended, self.lbl_rendered)
        self.info_label.setText(f"Blend: {alpha:.0%} rendered / {1-alpha:.0%} reference")

    def _show_diff(self):
        if self._rendered is None or self._reference is None:
            return
        h = min(self._rendered.shape[0], self._reference.shape[0])
        w = min(self._rendered.shape[1], self._reference.shape[1])
        r = self._rendered[:h, :w].astype(float)
        ref = self._reference[:h, :w].astype(float)

        diff = np.abs(r - ref)
        # Amplify for visibility
        diff = (diff * 5.0).clip(0, 255).astype(np.uint8)
        self._show_single(diff, self.lbl_rendered)

        mse = np.mean((r / 255.0 - ref / 255.0) ** 2)
        self.info_label.setText(f"MSE: {mse:.6f} | Diff amplified 5x")

    def _show_single(self, arr, label):
        if arr is None:
            return
        h, w, c = arr.shape
        img = QImage(arr.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        scaled = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)
