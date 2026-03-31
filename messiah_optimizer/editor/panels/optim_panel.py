"""
Optimization Panel - Controls for optimization mode, parameters, and constraints.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QLabel, QDoubleSpinBox, QSpinBox, QPushButton, QCheckBox,
    QFormLayout, QTabWidget,
)
from PyQt6.QtCore import pyqtSignal


class OptimizationPanel(QWidget):
    """Optimization parameter controls."""

    start_requested = pyqtSignal(dict)   # config dict
    stop_requested = pyqtSignal()

    def __init__(self, config: dict = None, parent=None):
        super().__init__(parent)
        self.config = config or {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Mode selection
        mode_group = QGroupBox("Optimization Mode")
        mode_layout = QVBoxLayout(mode_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Texture Optimization",
            "Material Fitting",
            "Shader Simplification",
            "Normal Map Baking",
        ])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        layout.addWidget(mode_group)

        # Tabs for mode-specific settings
        self.tabs = QTabWidget()

        # General tab
        self.tabs.addTab(self._create_general_tab(), "General")
        # Loss tab
        self.tabs.addTab(self._create_loss_tab(), "Loss Weights")
        # Comparison / post-processing tab
        self.tabs.addTab(self._create_comparison_tab(), "对比模式")
        # Constraints tab
        self.tabs.addTab(self._create_constraints_tab(), "Constraints")

        layout.addWidget(self.tabs)

        # Shading model (for shader simplification)
        shading_group = QGroupBox("Target Shading Model")
        shading_layout = QFormLayout(shading_group)

        self.shading_combo = QComboBox()
        self.shading_combo.addItems([
            "DefaultLit (1)", "Unlit (0)", "SSS (3)",
            "Hair (4)", "Cloth (10)", "ClearCoat (17)",
        ])
        shading_layout.addRow("Source Model:", self.shading_combo)

        self.target_combo = QComboBox()
        self.target_combo.addItems(["DefaultLit (1)", "Unlit (0)"])
        shading_layout.addRow("Target Model:", self.target_combo)
        layout.addWidget(shading_group)

        # Action buttons
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("  \u25B6  Start Optimization")
        self.btn_start.setProperty("cssClass", "primary")
        self.btn_start.clicked.connect(self._on_start)
        btn_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("  \u25A0  Stop")
        self.btn_stop.setProperty("cssClass", "danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

    def _create_general_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 1.0)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setValue(0.01)
        form.addRow("Learning Rate:", self.spin_lr)

        self.spin_iters = QSpinBox()
        self.spin_iters.setRange(100, 100000)
        self.spin_iters.setSingleStep(500)
        self.spin_iters.setValue(5000)
        form.addRow("Max Iterations:", self.spin_iters)

        self.combo_scheduler = QComboBox()
        self.combo_scheduler.addItems(["Cosine Annealing", "Step", "Exponential"])
        form.addRow("LR Schedule:", self.combo_scheduler)

        self.spin_views = QSpinBox()
        self.spin_views.setRange(1, 64)
        self.spin_views.setValue(16)
        form.addRow("Num Views:", self.spin_views)

        return w

    def _create_loss_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.spin_w_l2 = QDoubleSpinBox()
        self.spin_w_l2.setRange(0.0, 10.0)
        self.spin_w_l2.setValue(1.0)
        self.spin_w_l2.setDecimals(3)
        form.addRow("L2 Weight:", self.spin_w_l2)

        self.spin_w_percep = QDoubleSpinBox()
        self.spin_w_percep.setRange(0.0, 10.0)
        self.spin_w_percep.setValue(0.1)
        self.spin_w_percep.setDecimals(3)
        form.addRow("Perceptual Weight:", self.spin_w_percep)

        self.spin_w_ssim = QDoubleSpinBox()
        self.spin_w_ssim.setRange(0.0, 10.0)
        self.spin_w_ssim.setValue(0.05)
        self.spin_w_ssim.setDecimals(3)
        form.addRow("SSIM Weight:", self.spin_w_ssim)

        self.spin_w_smooth = QDoubleSpinBox()
        self.spin_w_smooth.setRange(0.0, 1.0)
        self.spin_w_smooth.setValue(0.001)
        self.spin_w_smooth.setDecimals(4)
        form.addRow("Smoothness Weight:", self.spin_w_smooth)

        return w

    def _create_constraints_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.spin_tex_res = QComboBox()
        self.spin_tex_res.addItems(["64x64", "128x128", "256x256", "512x512", "1024x1024"])
        self.spin_tex_res.setCurrentIndex(2)
        form.addRow("Target Resolution:", self.spin_tex_res)

        self.check_clamp = QCheckBox("Clamp to [0, 1]")
        self.check_clamp.setChecked(True)
        form.addRow(self.check_clamp)

        self.check_srgb = QCheckBox("sRGB output")
        self.check_srgb.setChecked(True)
        form.addRow(self.check_srgb)

        return w

    def _create_comparison_tab(self) -> QWidget:
        """Comparison mode and post-processing controls."""
        w = QWidget()
        layout = QVBoxLayout(w)

        # Reference source selector
        ref_group = QGroupBox("参考图来源")
        ref_layout = QVBoxLayout(ref_group)

        self.ref_mode_combo = QComboBox()
        self.ref_mode_combo.addItems([
            "自监督 (同管线渲染，无需引擎)",
            "手动导入参考图",
            "引擎截图 (MPlatform.ScreenShot)",
        ])
        self.ref_mode_combo.setToolTip(
            "自监督: nvdiffrast渲染高精度版本作为参考，再优化低精度版本\n"
            "手动导入: File→Import Reference Images 导入引擎截图或照片\n"
            "引擎截图: 通过TCP连接Messiah引擎实时截图"
        )
        ref_layout.addWidget(self.ref_mode_combo)

        ref_hint = QLabel(
            "<small>💡 <b>自监督模式</b>不需要连接引擎。<br>"
            "参考图和优化图使用相同管线渲染，<br>"
            "后处理效果自动抵消，对比公平。</small>")
        ref_hint.setWordWrap(True)
        ref_layout.addWidget(ref_hint)
        layout.addWidget(ref_group)

        # Post-processing matching
        pp_group = QGroupBox("后处理匹配 (仅引擎截图模式)")
        pp_layout = QFormLayout(pp_group)

        self.pp_mode_combo = QComboBox()
        self.pp_mode_combo.addItems(["关闭", "匹配引擎", "自定义"])
        self.pp_mode_combo.setToolTip(
            "关闭: 不加后处理 (自监督模式推荐)\n"
            "匹配引擎: 添加bloom/色彩分级逼近引擎后处理\n"
            "自定义: 手动调整后处理参数"
        )
        pp_layout.addRow("后处理模式:", self.pp_mode_combo)

        self.spin_bloom_intensity = QDoubleSpinBox()
        self.spin_bloom_intensity.setRange(0.0, 1.0)
        self.spin_bloom_intensity.setValue(0.3)
        self.spin_bloom_intensity.setDecimals(2)
        self.spin_bloom_intensity.setSingleStep(0.05)
        pp_layout.addRow("Bloom 强度:", self.spin_bloom_intensity)

        self.spin_bloom_threshold = QDoubleSpinBox()
        self.spin_bloom_threshold.setRange(0.1, 2.0)
        self.spin_bloom_threshold.setValue(0.8)
        self.spin_bloom_threshold.setDecimals(2)
        pp_layout.addRow("Bloom 阈值:", self.spin_bloom_threshold)

        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setRange(-3.0, 3.0)
        self.spin_exposure.setValue(0.0)
        self.spin_exposure.setDecimals(2)
        pp_layout.addRow("曝光补偿:", self.spin_exposure)

        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.setRange(0.5, 2.0)
        self.spin_contrast.setValue(1.05)
        self.spin_contrast.setDecimals(2)
        pp_layout.addRow("对比度:", self.spin_contrast)

        self.spin_saturation = QDoubleSpinBox()
        self.spin_saturation.setRange(0.0, 3.0)
        self.spin_saturation.setValue(1.1)
        self.spin_saturation.setDecimals(2)
        pp_layout.addRow("饱和度:", self.spin_saturation)

        layout.addWidget(pp_group)

        # Info about domain gap
        info = QLabel(
            "<small>⚠️ 引擎有 Bloom/DOF/SSAO/色彩分级 等后处理，<br>"
            "nvdiffrast 没有这些效果。<b>自监督模式</b>下后处理<br>"
            "对 ref & rendered 相同，自动抵消，无域差。<br>"
            "使用引擎截图时建议开启「匹配引擎」后处理。</small>")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()

        # Connect mode changes
        self.ref_mode_combo.currentIndexChanged.connect(self._on_ref_mode_changed)
        self.pp_mode_combo.currentIndexChanged.connect(self._on_pp_mode_changed)

        return w

    def _on_ref_mode_changed(self, index):
        """Auto-adjust post-processing when reference source changes."""
        if index == 0:  # Self-supervised
            self.pp_mode_combo.setCurrentIndex(0)  # Disable post-proc
        elif index == 2:  # Engine screenshot
            self.pp_mode_combo.setCurrentIndex(1)  # Enable match_engine

    def _on_pp_mode_changed(self, index):
        """Enable/disable post-processing controls based on mode."""
        enabled = index >= 1  # match_engine or custom
        self.spin_bloom_intensity.setEnabled(enabled)
        self.spin_bloom_threshold.setEnabled(enabled)
        self.spin_exposure.setEnabled(enabled)
        self.spin_contrast.setEnabled(enabled)
        self.spin_saturation.setEnabled(enabled)

    def _on_mode_changed(self, index):
        pass  # Could show/hide mode-specific UI

    def _on_start(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.start_requested.emit(self.get_config())

    def _on_stop(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.stop_requested.emit()

    def get_config(self) -> dict:
        """Gather current optimization config from UI."""
        res_text = self.spin_tex_res.currentText()
        res = int(res_text.split('x')[0])

        # Post-processing mode mapping
        pp_modes = {0: 'disabled', 1: 'match_engine', 2: 'custom'}
        pp_mode = pp_modes.get(self.pp_mode_combo.currentIndex(), 'disabled')

        # Reference source mapping
        ref_modes = {0: 'self_supervised', 1: 'manual_import', 2: 'engine_capture'}
        ref_mode = ref_modes.get(self.ref_mode_combo.currentIndex(), 'self_supervised')

        return {
            'mode': self.mode_combo.currentText(),
            'learning_rate': self.spin_lr.value(),
            'max_iterations': self.spin_iters.value(),
            'num_views': self.spin_views.value(),
            'target_resolution': (res, res),
            'loss_weights': {
                'l2': self.spin_w_l2.value(),
                'perceptual': self.spin_w_percep.value(),
                'ssim': self.spin_w_ssim.value(),
                'smoothness': self.spin_w_smooth.value(),
            },
            'reference_mode': ref_mode,
            'postprocess_mode': pp_mode,
            'postprocess_config': {
                'bloom': {
                    'enabled': pp_mode != 'disabled',
                    'threshold': self.spin_bloom_threshold.value(),
                    'intensity': self.spin_bloom_intensity.value(),
                },
                'color_grading': {
                    'enabled': pp_mode != 'disabled',
                    'exposure': self.spin_exposure.value(),
                    'contrast': self.spin_contrast.value(),
                    'saturation': self.spin_saturation.value(),
                },
            },
        }
