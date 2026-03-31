"""
Export Panel - Export optimized textures and material parameters.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QLabel, QPushButton, QLineEdit, QFileDialog, QCheckBox,
    QFormLayout, QTextEdit, QMessageBox,
)
from PyQt6.QtCore import pyqtSignal


class ExportPanel(QWidget):
    """Export optimized assets and generate reports."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Output directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Output:"))
        self.edit_dir = QLineEdit("./output")
        dir_layout.addWidget(self.edit_dir)
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._on_browse)
        dir_layout.addWidget(self.btn_browse)
        layout.addLayout(dir_layout)

        # Texture export
        tex_group = QGroupBox("Texture Export")
        tex_form = QFormLayout(tex_group)

        self.combo_tex_fmt = QComboBox()
        self.combo_tex_fmt.addItems(["PNG", "DDS (BC7)", "DDS (BC1)", "TGA", "EXR"])
        tex_form.addRow("Format:", self.combo_tex_fmt)

        self.check_mipmap = QCheckBox("Generate Mipmaps")
        self.check_mipmap.setChecked(True)
        tex_form.addRow(self.check_mipmap)

        self.check_srgb_export = QCheckBox("sRGB Color Space")
        self.check_srgb_export.setChecked(True)
        tex_form.addRow(self.check_srgb_export)

        self.btn_export_tex = QPushButton("  \u2B07  Export Textures")
        self.btn_export_tex.clicked.connect(self._on_export_textures)
        tex_form.addRow(self.btn_export_tex)
        layout.addWidget(tex_group)

        # Material export
        mat_group = QGroupBox("Material Export")
        mat_form = QFormLayout(mat_group)

        self.combo_mat_fmt = QComboBox()
        self.combo_mat_fmt.addItems(["Messiah XML", "JSON", "USD"])
        mat_form.addRow("Format:", self.combo_mat_fmt)

        self.btn_export_mat = QPushButton("  \u2B07  Export Material")
        self.btn_export_mat.clicked.connect(self._on_export_material)
        mat_form.addRow(self.btn_export_mat)
        layout.addWidget(mat_group)

        # Push to Messiah
        push_group = QGroupBox("Messiah Integration")
        push_layout = QVBoxLayout(push_group)

        self.btn_push = QPushButton("  \u26A1  Push to Messiah & Hot Reload")
        self.btn_push.setProperty("cssClass", "accent")
        self.btn_push.clicked.connect(self._on_push_to_messiah)
        push_layout.addWidget(self.btn_push)
        layout.addWidget(push_group)

        # Report log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        self.log.setPlaceholderText("Export log output will appear here...")
        layout.addWidget(self.log)

    def _on_browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.edit_dir.setText(path)

    def _on_export_textures(self):
        output_dir = self.edit_dir.text()
        os.makedirs(output_dir, exist_ok=True)
        fmt = self.combo_tex_fmt.currentText().lower()
        self.log.append(f"Exporting textures to {output_dir} ({fmt})...")
        # Actual export happens in do_export()

    def _on_export_material(self):
        output_dir = self.edit_dir.text()
        os.makedirs(output_dir, exist_ok=True)
        fmt = self.combo_mat_fmt.currentText()
        self.log.append(f"Exporting material ({fmt}) to {output_dir}...")

    def _on_push_to_messiah(self):
        self.log.append("Pushing results to Messiah...")

    def do_export(self, optimizer_instance):
        """Perform actual export using optimizer data."""
        if optimizer_instance is None:
            self.log.append("ERROR: No optimizer data to export")
            return

        output_dir = self.edit_dir.text()
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Export based on optimizer type
            if hasattr(optimizer_instance, 'export_texture'):
                tex_np = optimizer_instance.export_texture()
                from io_utils.texture_io import save_texture
                path = os.path.join(output_dir, "optimized_base_color.png")
                save_texture(tex_np, path)
                self.log.append(f"Saved texture: {path}")

            if hasattr(optimizer_instance, 'export_material'):
                mat_data = optimizer_instance.export_material()
                from io_utils.material_io import save_material
                path = os.path.join(output_dir, "optimized_material.json")
                save_material(mat_data, path)
                self.log.append(f"Saved material: {path}")

            if hasattr(optimizer_instance, 'export_textures'):
                textures = optimizer_instance.export_textures()
                from io_utils.texture_io import save_texture
                for name, tex_np in textures.items():
                    path = os.path.join(output_dir, f"optimized_{name}.png")
                    save_texture(tex_np, path)
                    self.log.append(f"Saved: {path}")

            if hasattr(optimizer_instance, 'export_normal_map'):
                nm_np = optimizer_instance.export_normal_map()
                from io_utils.texture_io import save_texture
                path = os.path.join(output_dir, "optimized_normal.png")
                save_texture(nm_np, path)
                self.log.append(f"Saved normal map: {path}")

            self.log.append("Export complete!")
        except Exception as e:
            self.log.append(f"ERROR: {e}")
