"""
Main application window for NvDiffRast Messiah Optimizer.
"""

import os
import logging
import yaml
import torch

logger = logging.getLogger(__name__)
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QSplitter, QWidget, QVBoxLayout,
    QMenuBar, QMenu, QStatusBar, QFileDialog, QMessageBox,
    QScrollArea, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QAction

from editor.panels.scene_panel import ScenePanel
from editor.panels.reference_panel import ReferencePanel
from editor.panels.optim_panel import OptimizationPanel
from editor.panels.monitor_panel import MonitorPanel
from editor.panels.export_panel import ExportPanel
from editor.panels.iteration_panel import IterationHistoryPanel
from editor.panels.renderdoc_panel import RenderDocPanel
from editor.widgets.viewport_3d import Viewport3D
from editor.widgets.image_compare import ImageCompareWidget

from pipeline.procedural import create_uv_sphere, create_default_textures, create_solid_color_textures
from pipeline.camera import Camera, create_orbit_cameras, transform_pos
from io_utils.texture_io import create_uniform_texture


class OptimizerMainWindow(QMainWindow):
    """Main window: viewport, panels, and optimization controls."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NvDiffRast Shader/Texture Optimizer for Messiah")
        self.setMinimumSize(960, 600)

        # State
        self.pipeline = None
        self.optimizer_instance = None
        self.is_optimizing = False
        self.bridge = None
        self.config = self._load_config()

        # Render preview state
        self._render_mesh = None     # dict with vertices, triangles, normals, uvs
        self._render_textures = None # dict with base_color, roughness, metallic
        self._render_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._use_software_renderer = False

        self._setup_menus()
        self._setup_ui()
        self._setup_statusbar()
        self._connect_signals()

        # Optimization timer
        self._opt_timer = QTimer()
        self._opt_timer.timeout.connect(self._optimization_step)

        # Render debounce timer (avoid rendering on every mouse pixel move)
        self._render_timer = QTimer()
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(30)  # ~33 fps max
        self._render_timer.timeout.connect(self._do_preview_render)

        # Default size for comfortable layout
        self.resize(1600, 900)

        # Auto-connect to engine project if previously configured
        saved_root = self.config.get('bridge', {}).get('engine_root', '')
        if saved_root and os.path.isdir(os.path.join(saved_root, 'Editor')):
            from bridge.local_bridge import LocalBridgeServer
            self.bridge = LocalBridgeServer(saved_root)
            self.bridge.start()
            self.statusBar().showMessage(f"已自动连接引擎项目: {saved_root}")

    def _load_config(self) -> dict:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base, 'config', 'default_config.yaml')
        cfg = {}
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
        # Merge user overrides (preserves default_config.yaml comments)
        user_path = os.path.join(base, 'config', 'user_config.yaml')
        if os.path.exists(user_path):
            with open(user_path, 'r', encoding='utf-8') as f:
                user = yaml.safe_load(f) or {}
            for key, val in user.items():
                if isinstance(val, dict) and isinstance(cfg.get(key), dict):
                    cfg[key].update(val)
                else:
                    cfg[key] = val
        return cfg

    def _save_user_config(self, section: str, values: dict):
        """Persist user-changed settings to config/user_config.yaml."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        user_path = os.path.join(base, 'config', 'user_config.yaml')
        existing = {}
        if os.path.exists(user_path):
            with open(user_path, 'r', encoding='utf-8') as f:
                existing = yaml.safe_load(f) or {}
        existing.setdefault(section, {}).update(values)
        with open(user_path, 'w', encoding='utf-8') as f:
            yaml.dump(existing, f, allow_unicode=True, default_flow_style=False)

    # ============ Menu Bar ============

    def _setup_menus(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_scene = QAction("Open Scene...", self)
        open_scene.setShortcut("Ctrl+O")
        open_scene.triggered.connect(self._on_open_scene)
        file_menu.addAction(open_scene)

        import_ref = QAction("Import Reference Images...", self)
        import_ref.triggered.connect(self._on_import_reference)
        file_menu.addAction(import_ref)

        file_menu.addSeparator()

        export_action = QAction("Export Results...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Optimize menu
        opt_menu = menubar.addMenu("Optimize")

        self._start_action = QAction("Start Optimization", self)
        self._start_action.setShortcut("F5")
        self._start_action.triggered.connect(self._on_start_optimize)
        opt_menu.addAction(self._start_action)

        self._stop_action = QAction("Stop Optimization", self)
        self._stop_action.setShortcut("Shift+F5")
        self._stop_action.setEnabled(False)
        self._stop_action.triggered.connect(self._on_stop_optimize)
        opt_menu.addAction(self._stop_action)

        opt_menu.addSeparator()

        reset_action = QAction("Reset Parameters", self)
        reset_action.triggered.connect(self._on_reset)
        opt_menu.addAction(reset_action)

        # Bridge menu
        bridge_menu = menubar.addMenu("Messiah Bridge")

        connect_action = QAction("连接引擎项目...", self)
        connect_action.triggered.connect(self._on_bridge_connect)
        bridge_menu.addAction(connect_action)

        disconnect_action = QAction("断开连接", self)
        disconnect_action.triggered.connect(self._on_bridge_disconnect)
        bridge_menu.addAction(disconnect_action)

        bridge_menu.addSeparator()

        pull_action = QAction("从引擎读取场景", self)
        pull_action.triggered.connect(self._on_bridge_pull)
        bridge_menu.addAction(pull_action)

        push_action = QAction("推送优化结果到引擎", self)
        push_action.triggered.connect(self._on_bridge_push)
        bridge_menu.addAction(push_action)

        bridge_menu.addSeparator()

        info_action = QAction("引擎项目信息", self)
        info_action.triggered.connect(self._on_bridge_info)
        bridge_menu.addAction(info_action)

        bridge_menu.addSeparator()

        # Unified pipeline actions
        inventory_action = QAction("资源清单 (仓库+RenderDoc)", self)
        inventory_action.triggered.connect(self._on_resource_inventory)
        bridge_menu.addAction(inventory_action)

        snap_before_action = QAction("拍摄快照: 优化前", self)
        snap_before_action.triggered.connect(self._on_snapshot_before)
        bridge_menu.addAction(snap_before_action)

        snap_after_action = QAction("拍摄快照: 优化后", self)
        snap_after_action.triggered.connect(self._on_snapshot_after)
        bridge_menu.addAction(snap_after_action)

        capture_ref_action = QAction("获取引擎参考截图 (MPlatform)", self)
        capture_ref_action.triggered.connect(self._on_capture_engine_reference)
        bridge_menu.addAction(capture_ref_action)

        compare_action = QAction("对比分析 (A/B比较)", self)
        compare_action.triggered.connect(self._on_run_comparison)
        bridge_menu.addAction(compare_action)

        bridge_menu.addSeparator()

        extract_rdc_action = QAction("从RenderDoc捕获提取资源...", self)
        extract_rdc_action.triggered.connect(self._on_extract_rdc)
        bridge_menu.addAction(extract_rdc_action)

        rdoc_replay_action = QAction("RenderDoc 回放对比 (Pipeline Inspector)...", self)
        rdoc_replay_action.triggered.connect(self._on_open_renderdoc_panel)
        bridge_menu.addAction(rdoc_replay_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    # ============ UI Layout ============

    def _setup_ui(self):
        # Central widget: split viewport and comparison
        central = QSplitter(Qt.Orientation.Horizontal)
        central.setChildrenCollapsible(True)
        central.setHandleWidth(3)

        self.viewport = Viewport3D()
        self.viewport.setMinimumSize(320, 240)
        self.compare_widget = ImageCompareWidget()
        self.compare_widget.setMinimumSize(320, 240)

        central.addWidget(self.viewport)
        central.addWidget(self.compare_widget)
        central.setStretchFactor(0, 1)
        central.setStretchFactor(1, 1)
        self.setCentralWidget(central)

        # Left dock: Scene tree
        self.scene_panel = ScenePanel()
        self._add_dock("Scene", self.scene_panel,
                       Qt.DockWidgetArea.LeftDockWidgetArea, min_width=350)

        # Right dock: Optimization controls + Reference (tabified)
        self.optim_panel = OptimizationPanel(self.config)
        self._add_dock("Optimization", self.optim_panel,
                       Qt.DockWidgetArea.RightDockWidgetArea,
                       min_width=280, scrollable=True)

        self.reference_panel = ReferencePanel()
        self._add_dock("Reference", self.reference_panel,
                       Qt.DockWidgetArea.RightDockWidgetArea,
                       min_width=280, scrollable=True)

        # Tab the right side panels
        self.tabifyDockWidget(
            self.findChild(QDockWidget, "Optimization_dock"),
            self.findChild(QDockWidget, "Reference_dock"),
        )
        # Select Optimization tab by default
        opt_dock = self.findChild(QDockWidget, "Optimization_dock")
        if opt_dock:
            opt_dock.raise_()

        # Bottom dock: Monitor + Export (tabified)
        self.monitor_panel = MonitorPanel()
        self._add_dock("Monitor", self.monitor_panel,
                       Qt.DockWidgetArea.BottomDockWidgetArea, min_height=200)

        self.export_panel = ExportPanel()
        self._add_dock("Export", self.export_panel,
                       Qt.DockWidgetArea.BottomDockWidgetArea,
                       min_height=200, scrollable=True)

        # Iteration History panel — between Monitor and Export
        self.iteration_panel = IterationHistoryPanel()
        self._add_dock("迭代历史", self.iteration_panel,
                       Qt.DockWidgetArea.BottomDockWidgetArea,
                       min_height=200, scrollable=False)

        # RenderDoc Comparison panel
        self.renderdoc_panel = RenderDocPanel()
        self._add_dock("RenderDoc 对比", self.renderdoc_panel,
                       Qt.DockWidgetArea.BottomDockWidgetArea,
                       min_height=200, scrollable=False)

        # Tab the bottom panels
        self.tabifyDockWidget(
            self.findChild(QDockWidget, "Monitor_dock"),
            self.findChild(QDockWidget, "迭代历史_dock"),
        )
        self.tabifyDockWidget(
            self.findChild(QDockWidget, "迭代历史_dock"),
            self.findChild(QDockWidget, "Export_dock"),
        )
        self.tabifyDockWidget(
            self.findChild(QDockWidget, "Export_dock"),
            self.findChild(QDockWidget, "RenderDoc 对比_dock"),
        )
        mon_dock = self.findChild(QDockWidget, "Monitor_dock")
        if mon_dock:
            mon_dock.raise_()

    def _add_dock(self, title: str, widget: QWidget,
                  area: Qt.DockWidgetArea, min_width: int = 0,
                  min_height: int = 0, scrollable: bool = False):
        dock = QDockWidget(title, self)
        dock.setObjectName(f"{title}_dock")
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        if scrollable:
            scroll = QScrollArea()
            scroll.setWidget(widget)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            dock.setWidget(scroll)
        else:
            dock.setWidget(widget)

        if min_width > 0:
            dock.setMinimumWidth(min_width)
        if min_height > 0:
            dock.setMinimumHeight(min_height)

        self.addDockWidget(area, dock)

    def _setup_statusbar(self):
        self.statusBar().showMessage("Ready")

    def _connect_signals(self):
        """Connect inter-panel signals."""
        self.scene_panel.scene_loaded.connect(self._on_scene_loaded)
        self.scene_panel.mesh_selection_changed.connect(self._on_load_selected_meshes)
        self.scene_panel.texture_optimize_requested.connect(self._on_texture_optimize_requested)
        self.viewport.camera_changed.connect(self._on_camera_changed)
        self.optim_panel.start_requested.connect(self._on_optim_start_requested)
        self.optim_panel.stop_requested.connect(self._on_stop_optimize)

    # ============ Slot Handlers ============

    def _on_open_scene(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Scene",
            filter="Scene Files (*.gltf *.glb *.fbx *.obj *.json);;All Files (*)"
        )
        if path:
            self.scene_panel.load_scene(path)
            self.statusBar().showMessage(f"Loaded: {path}")

    def _on_import_reference(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Reference Images",
            filter="Images (*.png *.jpg *.exr *.hdr);;All Files (*)"
        )
        if paths:
            self.reference_panel.load_images(paths)
            self.statusBar().showMessage(f"Loaded {len(paths)} reference images")

    def _on_export(self):
        self.export_panel.do_export(self.optimizer_instance)

    def _on_start_optimize(self):
        """Menu action: Start optimization (uses current panel config)."""
        config = self.optim_panel.get_config()
        self._on_optim_start_requested(config)

    def _on_optim_start_requested(self, config: dict):
        """Create optimizer from config and start optimization loop."""
        if self.pipeline is None or self._use_software_renderer:
            QMessageBox.warning(self, "Warning",
                                "GPU pipeline not available.\n"
                                "Optimization requires nvdiffrast + CUDA.")
            self.optim_panel.btn_start.setEnabled(True)
            self.optim_panel.btn_stop.setEnabled(False)
            return

        if self._render_mesh is None:
            QMessageBox.warning(self, "Warning",
                                "No scene loaded. Open a scene first.")
            self.optim_panel.btn_start.setEnabled(True)
            self.optim_panel.btn_stop.setEnabled(False)
            return

        self.statusBar().showMessage("Preparing optimization...")
        try:
            self._create_optimizer(config)
        except Exception as e:
            QMessageBox.critical(self, "Optimization Error",
                                 f"Failed to create optimizer:\n{e}")
            self.optim_panel.btn_start.setEnabled(True)
            self.optim_panel.btn_stop.setEnabled(False)
            import traceback
            traceback.print_exc()
            return

        self.is_optimizing = True
        self._start_action.setEnabled(False)
        self._stop_action.setEnabled(True)
        self._opt_timer.start(0)
        self.statusBar().showMessage("Optimizing...")

    @torch.no_grad()
    def _create_optimizer(self, config: dict):
        """Instantiate the correct optimizer based on config mode."""
        device = self._render_device
        mode = config['mode']
        num_views = config.get('num_views', 16)
        target_res = config.get('target_resolution', (256, 256))

        # Update pipeline post-processing based on comparison mode
        pp_mode = config.get('postprocess_mode', 'disabled')
        pp_config = config.get('postprocess_config', None)
        if self.pipeline and not self._use_software_renderer:
            from pipeline.postprocess import PostProcessStack
            self.pipeline.postprocess = PostProcessStack(
                mode=pp_mode, config=pp_config).to(device)

            pp_desc = self.pipeline.postprocess.describe()
            self.statusBar().showMessage(f"创建优化器... {pp_desc}")

        # Build orbit cameras
        cameras = create_orbit_cameras(
            num_views=num_views, distance=3.0, elevation=20.0,
            fov=45.0, device=device,
        )

        # Build mesh_data dict for optimizers
        mesh = self._render_mesh
        mesh_data = {
            'vertices': mesh['vertices'],
            'triangles': mesh['triangles'],
            'vtx_attr': {
                'normal': mesh['normals'],
                'uv': mesh['uvs'],
                'pos_world': mesh['vertices'],
            },
        }

        # Provide texture maps in mesh_data
        tex = self._render_textures
        mesh_data['base_color_hires'] = tex['base_color']
        mesh_data['roughness_tex'] = tex['roughness']
        mesh_data['metallic_tex'] = tex['metallic']
        if 'normal' in tex:
            mesh_data['normal_tex'] = tex['normal']

        # Generate reference images: render from all cameras at full quality
        ref_images = self._generate_reference_images(cameras, mesh_data)

        # Show first reference in compare widget
        self.compare_widget.set_reference(ref_images[0:1])

        # Create optimizer
        if mode == 'Texture Optimization':
            from optimizer.texture_optimizer import TextureOptimizer
            self.optimizer_instance = TextureOptimizer(
                pipeline=self.pipeline,
                mesh_data=mesh_data,
                cameras=cameras,
                reference_images=ref_images,
                target_resolution=target_res,
                config=config,
            )
        elif mode == 'Material Fitting':
            from optimizer.material_fitter import MaterialFitter
            self.optimizer_instance = MaterialFitter(
                pipeline=self.pipeline,
                mesh_data=mesh_data,
                cameras=cameras,
                reference_images=ref_images,
                tex_resolution=target_res,
                config=config,
            )
        elif mode == 'Shader Simplification':
            from optimizer.shader_simplifier import ShaderSimplifier
            self.optimizer_instance = ShaderSimplifier(
                pipeline=self.pipeline,
                mesh_data=mesh_data,
                cameras=cameras,
                reference_images=ref_images,
                config=config,
            )
        elif mode == 'Normal Map Baking':
            from optimizer.normal_baker import NormalMapBaker
            self.optimizer_instance = NormalMapBaker(
                pipeline=self.pipeline,
                mesh_data=mesh_data,
                cameras=cameras,
                reference_images=ref_images,
                config=config,
            )
        else:
            raise ValueError(f"Unknown optimization mode: {mode}")

        # Reset monitor and iteration history
        self.monitor_panel.reset()
        self.iteration_panel.reset()

        # Capture initial texture snapshot (iteration 0 = baseline)
        tex = self._get_current_optimized_texture()
        if tex is not None:
            self.iteration_panel.add_texture_snapshot(0, tex)

        # If shader simplifier, record initial params
        if hasattr(self.optimizer_instance, 'get_params'):
            self.iteration_panel.add_shader_params(0, self.optimizer_instance.get_params())

    @torch.no_grad()
    def _generate_reference_images(self, cameras, mesh_data):
        """Render reference images from all cameras for self-supervised optimization.

        If the user loaded reference images in the Reference panel, use those instead.
        """
        device = self._render_device

        # Check if user provided reference images
        user_refs = self.reference_panel.get_reference_tensor(device=device)
        if user_refs is not None and user_refs.shape[0] > 0:
            return user_refs

        # Self-supervised: render current scene as reference
        ref_list = []
        tex = self._render_textures
        for cam in cameras:
            color, mask, _ = self.pipeline.render_from_camera(
                cam, mesh_data['vertices'], mesh_data['triangles'],
                mesh_data['vtx_attr'], tex,
                apply_tonemap=True,
            )
            # Composite on dark bg
            bg = torch.tensor([0.05, 0.05, 0.07], device=device)
            bg = bg.view(1, 1, 1, 3).expand_as(color)
            composited = color * mask + bg * (1.0 - mask)
            ref_list.append(composited)

        return torch.cat(ref_list, dim=0)  # [N, H, W, 3]

    def _on_stop_optimize(self):
        self.is_optimizing = False
        self._opt_timer.stop()
        self._start_action.setEnabled(True)
        self._stop_action.setEnabled(False)
        self.optim_panel.btn_start.setEnabled(True)
        self.optim_panel.btn_stop.setEnabled(False)
        self.statusBar().showMessage("Optimization stopped")

    def _on_reset(self):
        self.optimizer_instance = None
        self.monitor_panel.reset()
        self.iteration_panel.reset()
        self.statusBar().showMessage("Optimizer reset")

    def _on_bridge_connect(self):
        """Connect to engine project via local file system bridge."""
        cfg = self.config.get('bridge', {})
        default_root = cfg.get('engine_root', '')

        engine_root = QFileDialog.getExistingDirectory(
            self, "选择 Messiah 引擎根目录",
            default_root,
        )
        if not engine_root:
            return

        # Validate
        if not os.path.isdir(os.path.join(engine_root, 'Editor')):
            QMessageBox.warning(
                self, "路径错误",
                f"未找到 Editor/ 目录，请确认选择了正确的引擎根目录。\n\n{engine_root}"
            )
            return

        from bridge.local_bridge import LocalBridgeServer
        self.bridge = LocalBridgeServer(engine_root)
        self.bridge.start()

        # Persist
        self.config.setdefault('bridge', {})['engine_root'] = engine_root
        self._save_user_config('bridge', {'engine_root': engine_root})

        info = self.bridge.get_engine_info()
        self.statusBar().showMessage(
            f"已连接引擎项目: {engine_root}  "
            f"(RenderDoc: {'可用' if info.get('renderdoc_available') else '未找到'}, "
            f"{info.get('world_count', 0)} 个世界)")

    def _on_bridge_disconnect(self):
        if hasattr(self, 'bridge') and self.bridge is not None:
            self.bridge.stop()
            self.bridge = None
            self.statusBar().showMessage("已断开引擎项目连接")

    def _on_bridge_pull(self):
        if not hasattr(self, 'bridge') or self.bridge is None:
            QMessageBox.warning(self, "未连接", "请先通过 Messiah Bridge → 连接引擎项目")
            return

        # Let user select a world file instead of scanning everything
        worlds = self.bridge.list_worlds()
        if not worlds:
            QMessageBox.warning(self, "无世界文件",
                                f"未在 {self.bridge.worlds_dir} 中找到 .iworld 文件")
            return

        world_names = [w['name'] for w in worlds]
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getItem(
            self, "选择世界", "请选择要解析的世界 (.iworld):",
            world_names, 0, False)
        if not ok:
            return

        selected = next(w for w in worlds if w['name'] == name)
        self.statusBar().showMessage(f"正在解析世界: {name} ...")

        try:
            stats = self.bridge.select_world(selected['path'])
            n_tex = len(self.bridge.get_textures())
            n_mat = len(self.bridge.get_materials())
            n_mesh = len(self.bridge.get_meshes())
            self.statusBar().showMessage(
                f"世界 '{name}' 已解析: "
                f"直接引用 {stats.get('resolved', 0)} 个, "
                f"含依赖共 {stats.get('all_resources', 0)} 个 "
                f"({n_tex} 贴图, {n_mat} 材质, {n_mesh} 网格)  "
                f"正在加载资源到视口...")
        except Exception as e:
            QMessageBox.critical(self, "解析失败", str(e))
            return

        # Populate mesh selection list — user picks which meshes to load
        try:
            meshes = self.bridge.get_meshes()
            self.scene_panel.populate_engine_mesh_list(meshes, name)
            self._current_world_name = name
            self.statusBar().showMessage(
                f"世界 '{name}' 已解析: {n_mesh} 网格可选择加载  "
                f"(勾选左侧列表后点击 '加载选中网格')")
        except Exception as e:
            self.statusBar().showMessage(
                f"世界 '{name}' 已解析, 但显示网格列表失败: {e}")
            import traceback
            traceback.print_exc()

    def _load_engine_scene_to_viewport(self, world_name: str):
        """
        Load resolved engine resources (meshes, textures, materials) into
        the scene panel and 3D viewport for rendering.
        """
        from io_utils.engine_scene_loader import load_engine_scene

        world = self.bridge.get_current_world()
        resolver = self.bridge._resolver
        device = self._render_device

        self.statusBar().showMessage(f"正在加载引擎资源到视口 ({world_name}) ...")

        scene_data = load_engine_scene(resolver, world, device=device)

        n_meshes = len(scene_data.get('meshes', []))
        n_mats = len(scene_data.get('materials', []))
        n_texs = len(scene_data.get('textures_loaded', []))

        # Populate scene panel tree
        self.scene_panel.load_engine_scene(scene_data, world_name)

        # The scene_loaded signal from load_engine_scene will trigger
        # _on_scene_loaded → _init_render_pipeline → _do_preview_render
        # But we want a better status message after everything completes
        self.statusBar().showMessage(
            f"世界 '{world_name}' 已加载: "
            f"{scene_data['vertex_count']:,} 顶点, "
            f"{scene_data['triangle_count']:,} 三角形, "
            f"{n_meshes} 网格, {n_mats} 材质, {n_texs} 贴图")

    def _on_load_selected_meshes(self, selected_guids: list):
        """Load only user-selected meshes into the viewport."""
        if not hasattr(self, 'bridge') or self.bridge is None:
            return

        from io_utils.engine_scene_loader import load_engine_scene_selective

        world = self.bridge.get_current_world()
        resolver = self.bridge._resolver
        device = self._render_device
        world_name = getattr(self, '_current_world_name', 'unknown')

        self.statusBar().showMessage(
            f"正在加载 {len(selected_guids)} 个选中网格...")

        try:
            scene_data = load_engine_scene_selective(
                resolver, world, selected_guids, device=device)
            logger.info(f"Selective load returned: {scene_data.get('vertex_count', 0)} verts, "
                        f"{scene_data.get('triangle_count', 0)} tris, "
                        f"keys={list(scene_data.keys())}")
        except Exception as e:
            self.statusBar().showMessage(f"加载选中网格失败: {e}")
            logger.error(f"load_engine_scene_selective failed: {e}", exc_info=True)
            return

        try:
            # Populate scene panel tree with loaded data
            self.scene_panel.load_engine_scene(scene_data, world_name)

            n_meshes = len(scene_data.get('meshes', []))
            self.statusBar().showMessage(
                f"已加载 {n_meshes} 个选中网格: "
                f"{scene_data['vertex_count']:,} 顶点, "
                f"{scene_data['triangle_count']:,} 三角形")
        except Exception as e:
            self.statusBar().showMessage(f"加载选中网格失败: {e}")
            logger.error(f"scene panel load or render failed: {e}", exc_info=True)

    def _on_texture_optimize_requested(self, request: dict):
        """Handle right-click 'optimize texture' from scene panel."""
        tex_name = request.get('name', 'Unknown')
        optimize_as = request.get('optimize_as', 'base_color')
        mesh_name = request.get('mesh_name', '')
        tex_path = request.get('path', '')
        tex_guid = request.get('guid', '')

        # Store the optimization target for use when optimization starts
        self._pending_tex_optimize = request

        # Load the source texture as initial state / reference
        if tex_path:
            self._load_texture_preview(request)

        # Switch optimization panel to appropriate mode
        mode_map = {
            'base_color': 0,   # "Texture Optimization"
            'normal': 3,       # "Normal Map Baking"
        }
        mode_idx = mode_map.get(optimize_as, 0)
        self.optim_panel.mode_combo.setCurrentIndex(mode_idx)

        # Raise optimization dock
        opt_dock = self.findChild(QDockWidget, "Optimization_dock")
        if opt_dock:
            opt_dock.raise_()

        self.statusBar().showMessage(
            f"优化目标已设置: {tex_name} (作为 {optimize_as}) — "
            f"Mesh: {mesh_name}  |  点击 Start 开始优化")

    def _load_texture_preview(self, request: dict):
        """Load and display a texture from engine resource path."""
        import numpy as np
        tex_path = request.get('path', '')
        if not tex_path:
            return

        # Try source.tga/png/jpg
        for fname in ('source.tga', 'source.png', 'source.jpg'):
            img_path = os.path.join(tex_path, fname)
            if os.path.exists(img_path):
                try:
                    from PIL import Image
                    img = Image.open(img_path).convert('RGB')
                    img_np = np.array(img, dtype=np.uint8)
                    label = (f"{request.get('optimize_as', 'texture')}: "
                             f"{request.get('name', 'Unknown')}")
                    # Show in reference panel as the input texture
                    self.reference_panel.add_image_from_array(img_np, name=label)
                    # Raise reference panel
                    ref_dock = self.findChild(QDockWidget, "Reference_dock")
                    if ref_dock:
                        ref_dock.raise_()
                    return
                except Exception as e:
                    logger.debug(f"Failed to load texture preview {img_path}: {e}")

    def _on_bridge_push(self):
        if not hasattr(self, 'bridge') or self.bridge is None:
            QMessageBox.warning(self, "未连接", "请先通过 Messiah Bridge → 连接引擎项目")
            return
        # TODO: Push optimization results from the export panel
        self.statusBar().showMessage("推送功能将在优化完成后可用")

    def _on_bridge_info(self):
        if not hasattr(self, 'bridge') or self.bridge is None:
            QMessageBox.warning(self, "未连接", "请先通过 Messiah Bridge → 连接引擎项目")
            return
        info = self.bridge.get_engine_info()
        lines = [
            f"引擎根目录: {info['engine_root']}",
            f"Worlds 目录: {info.get('worlds_dir', 'N/A')}",
            f"Repository 目录: {info.get('repository_dir', 'N/A')}",
            f"可用世界数: {info.get('world_count', 0)}",
            f"RenderDoc: {'可用' if info.get('renderdoc_available') else '未找到'}",
        ]
        if 'current_world' in info:
            cw = info['current_world']
            lines.append(f"\n当前世界: {cw.get('world_name', 'N/A')}")
            lines.append(f"  关卡数: {cw.get('total_levels', 0)}")
            lines.append(f"  资源GUID: {cw.get('total_guids', 0)}")
            lines.append(f"  直接引用: {cw.get('resolved', 0)}")
            lines.append(f"  含依赖共: {cw.get('all_resources', 0)}")
            lines.append(f"\n  直接引用类型:")
            for t, c in cw.get('by_type', {}).items():
                lines.append(f"    {t}: {c}")
            all_by_type = cw.get('all_by_type', {})
            if all_by_type:
                lines.append(f"\n  全部资源类型 (含依赖):")
                for t, c in sorted(all_by_type.items(), key=lambda x: -x[1]):
                    lines.append(f"    {t}: {c}")
        QMessageBox.information(self, "引擎项目信息", '\n'.join(lines))

    # ============ Unified Pipeline Handlers ============

    def _on_resource_inventory(self):
        """Show unified resource inventory for current world."""
        if not self.bridge:
            QMessageBox.warning(self, "未连接", "请先连接引擎项目")
            return
        if not self.bridge.get_current_world():
            QMessageBox.warning(self, "未选择世界", "请先通过「从引擎读取场景」选择一个世界")
            return

        self.statusBar().showMessage("正在构建资源清单...")
        try:
            inventory = self.bridge.get_resource_inventory()
            lines = [
                f"世界: {inventory.get('world_name', 'N/A')}",
                f"总资源数: {inventory.get('total_resources', 0)}",
                f"找到数据文件: {inventory.get('files_found', 0)}",
                f"数据总大小: {inventory.get('total_data_size_mb', 0)} MB",
                f"数据来源: {inventory.get('source', 'N/A')}",
                f"RenderDoc: {'可用' if inventory.get('renderdoc_available') else '不可用'}",
                "",
                "按类型分布:",
            ]
            for t, info in inventory.get('by_type', {}).items():
                size_mb = round(info['total_size'] / (1024 * 1024), 2) if info['total_size'] else 0
                lines.append(f"  {t}: {info['count']} 个 ({info['with_data']} 有数据, {size_mb} MB)")
            QMessageBox.information(self, "资源清单", '\n'.join(lines))
            self.statusBar().showMessage("资源清单已生成")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def _on_snapshot_before(self):
        """Take a 'before optimization' snapshot."""
        if not self.bridge:
            QMessageBox.warning(self, "未连接", "请先连接引擎项目")
            return

        # Optionally select a .rdc file for visual data
        rdc_path = None
        reply = QMessageBox.question(
            self, "RenderDoc捕获",
            "是否选择一个RenderDoc捕获文件(.rdc)用于视觉对比?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            rdc_path, _ = QFileDialog.getOpenFileName(
                self, "选择优化前的RenderDoc捕获",
                filter="RenderDoc Captures (*.rdc);;All Files (*)"
            )
            if not rdc_path:
                rdc_path = None

        self.statusBar().showMessage("正在拍摄优化前快照...")
        result = self.bridge.take_snapshot_before(rdc_path)
        if 'error' in result:
            QMessageBox.warning(self, "快照失败", result['error'])
        else:
            src = result.get('source', '')
            fb = "有" if result.get('framebuffer') else "无"
            self.statusBar().showMessage(
                f"优化前快照已保存: {result.get('resources', 0)} 个资源, "
                f"来源={src}, 帧缓冲={fb}")

    def _on_snapshot_after(self):
        """Take an 'after optimization' snapshot."""
        if not self.bridge:
            QMessageBox.warning(self, "未连接", "请先连接引擎项目")
            return

        rdc_path = None
        reply = QMessageBox.question(
            self, "RenderDoc捕获",
            "是否选择一个RenderDoc捕获文件(.rdc)用于视觉对比?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            rdc_path, _ = QFileDialog.getOpenFileName(
                self, "选择优化后的RenderDoc捕获",
                filter="RenderDoc Captures (*.rdc);;All Files (*)"
            )
            if not rdc_path:
                rdc_path = None

        self.statusBar().showMessage("正在拍摄优化后快照...")
        result = self.bridge.take_snapshot_after(rdc_path)
        if 'error' in result:
            QMessageBox.warning(self, "快照失败", result['error'])
        else:
            src = result.get('source', '')
            fb = "有" if result.get('framebuffer') else "无"
            self.statusBar().showMessage(
                f"优化后快照已保存: {result.get('resources', 0)} 个资源, "
                f"来源={src}, 帧缓冲={fb}")

    def _on_run_comparison(self):
        """Run A/B comparison between before/after snapshots."""
        if not self.bridge:
            QMessageBox.warning(self, "未连接", "请先连接引擎项目")
            return

        self.statusBar().showMessage("正在运行对比分析...")
        result = self.bridge.run_comparison()
        if not result or 'error' in result:
            QMessageBox.warning(self, "对比失败",
                                result.get('error', '请先拍摄优化前和优化后快照'))
            return

        method = result.get('method', 'none')
        lines = [f"对比方式: {method}", ""]

        # Visual metrics
        visual = result.get('visual')
        if visual:
            lines.append("=== 视觉对比 (RenderDoc) ===")
            lines.append(f"  MSE: {visual.get('mse', 0):.4f}")
            lines.append(f"  PSNR: {visual.get('psnr', 0):.2f} dB")
            lines.append(f"  最大像素差: {visual.get('max_pixel_diff', 0):.1f}")
            lines.append(f"  平均像素差: {visual.get('mean_pixel_diff', 0):.4f}")
            lines.append(f"  完全一致: {'是' if visual.get('identical') else '否'}")
            lines.append("")

        # Resource metrics
        if result.get('resources_before', 0) > 0:
            lines.append("=== 资源对比 (仓库) ===")
            lines.append(f"  优化前资源数: {result.get('resources_before', 0)}")
            lines.append(f"  优化后资源数: {result.get('resources_after', 0)}")
            size_before_mb = result.get('total_size_before', 0) / (1024 * 1024)
            size_after_mb = result.get('total_size_after', 0) / (1024 * 1024)
            lines.append(f"  数据大小: {size_before_mb:.2f} MB → {size_after_mb:.2f} MB")
            lines.append(f"  大小变化: {result.get('size_delta', 0):+d} bytes")
            lines.append(f"  压缩率: {result.get('size_reduction_pct', 0):.1f}%")
            lines.append(f"  变更资源: {result.get('changed_count', 0)}")
            lines.append(f"  新增资源: {result.get('added_count', 0)}")
            lines.append(f"  移除资源: {result.get('removed_count', 0)}")
            lines.append("")

        lines.append(f"综合质量评分: {result.get('quality_score', 0):.3f} (1.0=无损)")

        QMessageBox.information(self, "对比分析结果", '\n'.join(lines))

        # Show diff image in the compare widget if available
        diff_img = result.get('diff_image', '')
        before_img = result.get('before_image', '')
        after_img = result.get('after_image', '')
        if before_img and after_img and os.path.exists(before_img) and os.path.exists(after_img):
            try:
                self.compare_widget.load_images(before_img, after_img)
                self.statusBar().showMessage(
                    f"对比完成: 质量={result.get('quality_score', 0):.3f}, "
                    f"PSNR={visual.get('psnr', 0):.1f}dB" if visual else "对比完成")
            except Exception:
                pass

    def _on_extract_rdc(self):
        """Extract resources from a RenderDoc .rdc capture file."""
        if not self.bridge:
            QMessageBox.warning(self, "未连接", "请先连接引擎项目")
            return

        rdc_path, _ = QFileDialog.getOpenFileName(
            self, "选择RenderDoc捕获文件",
            filter="RenderDoc Captures (*.rdc);;All Files (*)"
        )
        if not rdc_path:
            return

        self.statusBar().showMessage(f"正在从 {os.path.basename(rdc_path)} 提取资源...")
        result = self.bridge.extract_from_rdc(rdc_path)

        if result.get('success'):
            QMessageBox.information(self, "提取完成",
                f"帧缓冲: {'已提取' if result.get('framebuffer') else '未提取'}\n"
                f"贴图数: {result.get('textures', 0)}\n"
                f"输出目录: {result.get('output_dir', '')}\n"
                f"耗时: {result.get('time', 0):.1f}s")
        else:
            QMessageBox.warning(self, "提取失败", "无法从该.rdc文件提取数据")

    def _on_open_renderdoc_panel(self):
        """Open a .rdc file in the RenderDoc replay panel."""
        path, _ = QFileDialog.getOpenFileName(
            self, "打开 RenderDoc 捕获文件",
            filter="RenderDoc Captures (*.rdc);;All Files (*)"
        )
        if not path:
            return

        if self.renderdoc_panel.load_rdc(path):
            # Switch to the RenderDoc dock tab
            dock = self.findChild(QDockWidget, "RenderDoc 对比_dock")
            if dock:
                dock.raise_()
            self.statusBar().showMessage(f"已加载 RenderDoc 捕获: {os.path.basename(path)}")
        else:
            QMessageBox.warning(self, "打开失败", f"无法回放捕获文件:\n{path}")

    def _on_about(self):
        QMessageBox.about(
            self, "About",
            "NvDiffRast Shader/Texture Optimizer\n"
            "For Messiah Engine\n\n"
            "Uses nvdiffrast for differentiable rendering\n"
            "to optimize textures and shader parameters."
        )

    # ============ Engine Reference Capture ============

    def _on_capture_engine_reference(self):
        """Capture a reference image from the engine via MPlatform.ScreenShot.

        This image serves as the ground truth target for optimization —
        the differentiable renderer tries to match this output.
        
        The captured image goes to:
          1. Iteration History panel (render comparison tab)
          2. Reference panel (as user reference)
          3. ImageCompare widget (left side)
        """
        # Try TCP bridge first (engine running with plugin)
        from bridge.messiah_bridge import MessiahBridge
        tcp_bridge = getattr(self, '_tcp_bridge', None)

        if tcp_bridge is None:
            # Try to connect to the engine's RPC server
            cfg = self.config.get('bridge', {})
            host = cfg.get('host', '127.0.0.1')
            port = cfg.get('port', 9800)
            try:
                tcp_bridge = MessiahBridge(host, port, timeout=5.0)
                tcp_bridge.connect()
                self._tcp_bridge = tcp_bridge
            except Exception as e:
                QMessageBox.warning(
                    self, "无法连接引擎",
                    f"无法通过 TCP 连接到 Messiah 引擎 ({host}:{port})。\n\n"
                    f"请确保:\n"
                    f"1. Messiah Editor 正在运行\n"
                    f"2. optimizer_server.py 插件已加载\n\n"
                    f"错误: {e}\n\n"
                    f"也可以手动导入引擎截图:\n"
                    f"File → Import Reference Images")
                return

        self.statusBar().showMessage("正在从引擎获取截图 (MPlatform.ScreenShot)...")
        try:
            result = tcp_bridge.capture_frame(resolution=(1024, 1024))
        except Exception as e:
            QMessageBox.warning(self, "截图失败", f"引擎截图失败: {e}")
            return

        if 'error' in result:
            QMessageBox.warning(self, "截图失败", result['error'])
            return

        image_path = result.get('image_path', '')
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "截图失败", "引擎返回的截图文件不存在")
            return

        # Load the image
        import numpy as np
        from PyQt6.QtGui import QImage
        qimg = QImage(image_path)
        if qimg.isNull():
            QMessageBox.warning(self, "截图失败", f"无法加载截图: {image_path}")
            return

        # Convert to RGBA then RGB numpy
        qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(h * w * 3)
        img_np = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 3).copy()

        # Send to iteration panel
        self.iteration_panel.set_engine_reference(img_np)

        # Also add to reference panel for optimization use
        self.reference_panel.add_image_from_array(img_np, "Engine Reference")

        # Show in compare widget
        self.compare_widget.set_reference(img_np)

        self.statusBar().showMessage(
            f"已获取引擎参考截图: {w}x{h}  来源: {result.get('source', 'MPlatform')}")

    # ============ Scene Loading & Preview Rendering ============

    def _on_scene_loaded(self, scene_data: dict):
        """Handle scene loaded signal - initialize pipeline and render preview."""
        self.statusBar().showMessage("Initializing render pipeline...")
        logger.info(f"_on_scene_loaded: source={scene_data.get('source', '?')}, "
                     f"verts={scene_data.get('vertex_count', '?')}, "
                     f"has_vertex_colors={'vertex_colors' in scene_data}")

        try:
            self._init_render_pipeline(scene_data)
        except Exception as e:
            self.statusBar().showMessage(f"Render init failed: {e}")
            logger.error(f"_init_render_pipeline failed: {e}", exc_info=True)
            return

        try:
            self._do_preview_render()
        except Exception as e:
            self.statusBar().showMessage(f"Preview render failed: {e}")
            logger.error(f"_do_preview_render failed: {e}", exc_info=True)

    def _init_render_pipeline(self, scene_data: dict):
        """Initialize the nvdiffrast pipeline with scene data."""
        device = self._render_device

        # Try to create the nvdiffrast GPU pipeline
        try:
            from pipeline.messiah_pipeline import MessiahDiffPipeline
            render_cfg = self.config.get('rendering', {})
            resolution = tuple(render_cfg.get('resolution', [512, 512]))
            self.pipeline = MessiahDiffPipeline(resolution=resolution, device=device)
            self._use_software_renderer = False
        except Exception:
            # nvdiffrast or CUDA not available - use software renderer
            self.pipeline = 'software'
            self._use_software_renderer = True
            self.statusBar().showMessage(
                "nvdiffrast/CUDA not available - using software preview renderer"
            )

        # If we have real geometry from glTF loader, use it
        if scene_data and 'vertices' in scene_data:
            self._render_mesh = scene_data
        else:
            # Generate procedural sphere preview from scene metadata
            cpu_device = 'cpu' if self._use_software_renderer else device
            rings = 32 if self._use_software_renderer else 48
            sectors = 64 if self._use_software_renderer else 96
            mesh = create_uv_sphere(radius=1.0, rings=rings, sectors=sectors, device=cpu_device)
            self._render_mesh = mesh

        # Determine textures
        cpu_device = 'cpu' if self._use_software_renderer else device
        if scene_data and 'base_color_texture' in scene_data:
            # Engine-loaded texture tensor [1, H, W, 3]
            base_tex = scene_data['base_color_texture']
            if self._use_software_renderer:
                base_tex = base_tex.cpu()
            self._render_textures = {
                'base_color': base_tex,
                'roughness': create_uniform_texture(0.5, device=cpu_device),
                'metallic': create_uniform_texture(0.0, device=cpu_device),
            }
        elif scene_data and scene_data.get('materials'):
            mat = scene_data['materials'][0]
            color = mat.get('base_color', [0.8, 0.8, 0.82])
            rough = mat.get('roughness', 0.5)
            metal = mat.get('metallic', 0.0)
            self._render_textures = create_solid_color_textures(
                color, rough, metal, resolution=64, device=cpu_device
            )
        else:
            self._render_textures = create_default_textures(resolution=256, device=cpu_device)

        self.statusBar().showMessage("Pipeline ready")

    def _on_camera_changed(self):
        """Viewport camera was moved - schedule a re-render."""
        if self.pipeline is not None and self._render_mesh is not None:
            # Longer delay for CPU rendering to avoid lag
            if hasattr(self, '_use_software_renderer') and self._use_software_renderer:
                self._render_timer.setInterval(150)
            else:
                self._render_timer.setInterval(30)
            self._render_timer.start()

    @torch.no_grad()
    def _do_preview_render(self):
        """Render current scene from viewport camera and display."""
        if self.pipeline is None or self._render_mesh is None:
            return

        try:
            cam_params = self.viewport.get_camera_params()
            mesh = self._render_mesh

            if self._use_software_renderer:
                self._render_software(mesh, cam_params)
            else:
                self._render_nvdiffrast(mesh, cam_params)
        except Exception as e:
            self.statusBar().showMessage(f"Render error: {e}")
            logger.error(f"_do_preview_render failed: {e}", exc_info=True)

    def _render_software(self, mesh, cam_params):
        """Render using CPU software rasterizer."""
        from pipeline.software_renderer import render_preview

        # Use lower resolution for CPU rendering (faster)
        resolution = (384, 384)

        # Get light from scene data if available
        light_dir = None
        if self.scene_panel.scene_data:
            lights = self.scene_panel.scene_data.get('lights', [])
            if lights:
                light_dir = lights[0].get('direction', None)

        image = render_preview(
            vertices=mesh['vertices'],
            triangles=mesh['triangles'],
            normals=mesh['normals'],
            uvs=mesh.get('uvs'),
            textures=self._render_textures,
            camera_params=cam_params,
            resolution=resolution,
            light_dir=light_dir,
        )

        self.viewport.set_image(image)

        verts_n = mesh.get('vertex_count', 0)
        tris_n = mesh.get('triangle_count', 0)
        self.statusBar().showMessage(
            f"Preview (CPU): {verts_n} verts, {tris_n} tris | "
            f"Azim: {cam_params['azimuth']:.1f}\u00b0 Elev: {cam_params['elevation']:.1f}\u00b0"
        )

    def _render_nvdiffrast(self, mesh, cam_params):
        """Render using nvdiffrast GPU pipeline."""
        device = self._render_device

        camera = Camera(
            position=cam_params['position'],
            target=cam_params['target'],
            fov=cam_params['fov'],
            device=device,
        )

        vertices = mesh['vertices']
        tri = mesh['triangles']
        vtx_attr = {
            'normal': mesh['normals'],
            'uv': mesh['uvs'],
            'pos_world': vertices,
        }

        color, mask, rast_out = self.pipeline.render_from_camera(
            camera, vertices, tri, vtx_attr,
            self._render_textures,
            apply_tonemap=True,
        )

        # Apply per-mesh highlight color tint if available
        if 'vertex_colors' in mesh and mesh['vertex_colors'] is not None:
            try:
                import nvdiffrast.torch as dr
                vtx_colors = mesh['vertex_colors']
                # Ensure correct shape [V, 3] and device
                if vtx_colors.dim() == 2 and vtx_colors.shape[1] == 3:
                    vtx_colors = vtx_colors.to(device=device, dtype=torch.float32)
                    color_interp, _ = dr.interpolate(
                        vtx_colors[None, ...], rast_out, tri)
                    # Blend: 70% shaded + 30% mesh highlight
                    color = color * 0.7 + color_interp * mask * 0.3
                else:
                    logger.warning(f"vertex_colors unexpected shape: {vtx_colors.shape}")
            except Exception as e:
                logger.warning(f"Vertex color tinting failed: {e}")
                pass  # Fall back to uncolored render

        # Composite over dark background
        bg = torch.tensor([0.05, 0.05, 0.07], device=device)
        bg = bg.view(1, 1, 1, 3).expand_as(color)
        composited = color * mask + bg * (1.0 - mask)

        self.viewport.set_image_from_tensor(composited)

        verts_n = mesh.get('vertex_count', vertices.shape[0])
        tris_n = mesh.get('triangle_count', tri.shape[0])
        self.statusBar().showMessage(
            f"Preview (GPU): {verts_n} verts, {tris_n} tris | "
            f"Azim: {cam_params['azimuth']:.1f}\u00b0 Elev: {cam_params['elevation']:.1f}\u00b0"
        )

    def _optimization_step(self):
        """Called by timer during optimization."""
        if not self.is_optimizing or self.optimizer_instance is None:
            return

        result = self.optimizer_instance.step()
        iteration = result['iteration']
        loss = result['loss']
        psnr = result.get('psnr', 0)

        # Update monitor
        self.monitor_panel.update_metrics(
            iteration=iteration, loss=loss, psnr=psnr)

        # Update iteration history panel — loss sparkline every step
        self.iteration_panel.add_loss_point(iteration, loss, psnr)

        # Periodic updates (every 10 iterations) for rendered image
        if iteration % 10 == 0:
            self.compare_widget.set_rendered(result['rendered'])

            # Convert rendered tensor to numpy uint8 for iteration panel
            import numpy as np
            rendered_np = (result['rendered'][0].cpu().numpy() * 255
                           ).clip(0, 255).astype(np.uint8)
            self.iteration_panel.add_render_snapshot(iteration, rendered_np)

        # Texture snapshot (every 50 iterations for gallery)
        if iteration % 50 == 0:
            tex = self._get_current_optimized_texture()
            if tex is not None:
                self.iteration_panel.add_texture_snapshot(iteration, tex)

        # Shader params (every 20 iterations if ShaderSimplifier)
        if iteration % 20 == 0 and 'params' in result:
            self.iteration_panel.add_shader_params(iteration, result['params'])

        # Auto-stop at max iterations
        max_iter = self.optim_panel.get_config().get('max_iterations', 5000)
        if iteration >= max_iter:
            # Final snapshots
            tex = self._get_current_optimized_texture()
            if tex is not None:
                self.iteration_panel.add_texture_snapshot(iteration, tex)
            if 'params' in result:
                self.iteration_panel.add_shader_params(iteration, result['params'])

            self._on_stop_optimize()
            self.statusBar().showMessage(
                f"Optimization complete: {iteration} iterations, "
                f"Loss: {loss:.6f}, PSNR: {psnr:.2f} dB"
            )
            return

        # Update status
        self.statusBar().showMessage(
            f"Iter {iteration} | "
            f"Loss: {loss:.6f} | "
            f"PSNR: {psnr:.2f} dB"
        )

    @torch.no_grad()
    def _get_current_optimized_texture(self) -> 'np.ndarray | None':
        """Extract the current texture being optimized as a numpy uint8 image."""
        import numpy as np
        opt = self.optimizer_instance
        if opt is None:
            return None

        # TextureOptimizer has get_texture()
        if hasattr(opt, 'get_texture'):
            tex = opt.get_texture()  # [1, H, W, 3] in [0,1]
            return (tex[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # MaterialFitter has multiple textures
        if hasattr(opt, 'get_base_color'):
            tex = opt.get_base_color()
            return (tex[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        # ShaderSimplifier — show the color-biased texture
        if hasattr(opt, 'color_bias') and hasattr(opt, 'mesh'):
            base = opt.mesh.get('base_color_hires')
            if base is not None:
                adj = torch.clamp(base + opt.color_bias, 0, 1)
                return (adj[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        return None
