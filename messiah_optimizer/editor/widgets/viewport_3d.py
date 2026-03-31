"""
3D Viewport Widget - Displays nvdiffrast rendered output and allows camera orbiting.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QWheelEvent
import math


class Viewport3D(QWidget):
    """
    3D viewport for displaying nvdiffrast render output.

    Supports mouse-based camera orbit:
      - Left drag:   Orbit
      - Right drag:  Pan
      - Scroll:      Zoom
    """

    camera_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)

        # Camera orbit state
        self._azimuth = 0.0     # degrees
        self._elevation = 20.0  # degrees
        self._distance = 3.0
        self._target = [0.0, 0.0, 0.0]
        self._fov = 60.0

        self._last_mouse_pos = QPoint()
        self._dragging = False
        self._drag_button = Qt.MouseButton.NoButton

        # Display
        self._current_image = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.image_label = QLabel()
        self.image_label.setObjectName("viewport_display")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.image_label, 1)

        self.info_label = QLabel("No render output")
        self.info_label.setObjectName("viewport_info")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.info_label)

    def set_image_from_tensor(self, tensor):
        """
        Display a rendered image from a PyTorch tensor.

        Args:
            tensor: [B, H, W, 3] or [H, W, 3] float32 in [0, 1]
        """
        import torch
        if tensor is None:
            return

        if tensor.dim() == 4:
            tensor = tensor[0]
        arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        self.set_image(arr)

    def set_image(self, arr: np.ndarray):
        """
        Display an image from a numpy array.

        Args:
            arr: [H, W, 3] uint8 RGB array
        """
        if arr is None:
            return

        self._current_image = arr
        h, w, c = arr.shape
        bytes_per_line = w * c
        img = QImage(arr.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)

        # Scale to fit widget
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.info_label.setText(f"{w}x{h}")

    # ============ Mouse camera control ============

    def mousePressEvent(self, event: QMouseEvent):
        self._last_mouse_pos = event.pos()
        self._dragging = True
        self._drag_button = event.button()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._dragging:
            return

        dx = event.pos().x() - self._last_mouse_pos.x()
        dy = event.pos().y() - self._last_mouse_pos.y()
        self._last_mouse_pos = event.pos()

        if self._drag_button == Qt.MouseButton.LeftButton:
            # Orbit
            self._azimuth += dx * 0.5
            self._elevation = max(-89, min(89, self._elevation - dy * 0.5))
        elif self._drag_button == Qt.MouseButton.RightButton:
            # Pan
            scale = self._distance * 0.002
            self._target[0] -= dx * scale
            self._target[1] += dy * scale

        self.camera_changed.emit()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragging = False
        event.accept()

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        self._distance *= 0.95 if delta > 0 else 1.05
        self._distance = max(0.1, min(100.0, self._distance))
        self.camera_changed.emit()
        event.accept()

    def get_camera_params(self) -> dict:
        """Get current camera parameters for rendering."""
        elev_rad = math.radians(self._elevation)
        azim_rad = math.radians(self._azimuth)

        y = self._distance * math.sin(elev_rad)
        r = self._distance * math.cos(elev_rad)
        x = r * math.cos(azim_rad)
        z = r * math.sin(azim_rad)

        return {
            'position': [
                x + self._target[0],
                y + self._target[1],
                z + self._target[2],
            ],
            'target': self._target.copy(),
            'fov': self._fov,
            'distance': self._distance,
            'azimuth': self._azimuth,
            'elevation': self._elevation,
        }
