"""
Scene Panel - Import and browse scene hierarchy.
Supports selective mesh loading with per-mesh highlight colors for engine scenes.
"""

import os
import json
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QHBoxLayout, QLabel, QFileDialog,
    QMenu, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QAction


# Distinct colors for highlighting selected meshes in the viewport
MESH_HIGHLIGHT_COLORS = [
    (0.20, 0.60, 1.00),   # blue
    (1.00, 0.40, 0.20),   # orange
    (0.20, 0.85, 0.40),   # green
    (0.90, 0.25, 0.70),   # pink
    (1.00, 0.85, 0.10),   # yellow
    (0.40, 0.90, 0.90),   # cyan
    (0.70, 0.45, 1.00),   # purple
    (0.95, 0.55, 0.40),   # salmon
    (0.50, 1.00, 0.60),   # lime
    (0.85, 0.70, 0.30),   # gold
]


class ScenePanel(QWidget):
    """Scene browser with mesh/material/light tree and selective mesh loading."""

    scene_loaded = pyqtSignal(dict)  # Emitted when scene data is ready
    mesh_selection_changed = pyqtSignal(list)  # Emitted when user changes mesh selection (list of guids)
    texture_optimize_requested = pyqtSignal(dict)  # Emitted when user right-clicks a texture to optimize

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_data = None
        self._engine_mesh_list = []   # all meshes from resolver (ResourceInfo list)
        self._mesh_check_items = {}   # guid → QTreeWidgetItem (with checkbox)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        self.btn_load = QPushButton("\U0001F4C2  Load Scene")
        self.btn_load.clicked.connect(self._on_load)
        toolbar.addWidget(self.btn_load)

        self.btn_reload = QPushButton("\U0001F504  Reload")
        self.btn_reload.clicked.connect(self._on_reload)
        self.btn_reload.setEnabled(False)
        toolbar.addWidget(self.btn_reload)
        layout.addLayout(toolbar)

        # Mesh selection toolbar (hidden until engine scene is loaded)
        self._mesh_toolbar = QHBoxLayout()
        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.clicked.connect(self._on_select_all)
        self.btn_select_all.setVisible(False)
        self._mesh_toolbar.addWidget(self.btn_select_all)

        self.btn_deselect_all = QPushButton("全不选")
        self.btn_deselect_all.clicked.connect(self._on_deselect_all)
        self.btn_deselect_all.setVisible(False)
        self._mesh_toolbar.addWidget(self.btn_deselect_all)

        self.btn_load_selected = QPushButton("▶ 加载选中网格")
        self.btn_load_selected.clicked.connect(self._on_load_selected)
        self.btn_load_selected.setVisible(False)
        self.btn_load_selected.setStyleSheet(
            "QPushButton { background-color: #0e639c; color: white; "
            "font-weight: bold; padding: 4px 8px; }")
        self._mesh_toolbar.addWidget(self.btn_load_selected)
        layout.addLayout(self._mesh_toolbar)

        # Scene info
        self.lbl_info = QLabel("No scene loaded")
        self.lbl_info.setProperty("cssClass", "dim")
        layout.addWidget(self.lbl_info)

        # Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type", "Info"])
        self.tree.setColumnWidth(0, 220)
        self.tree.setColumnWidth(1, 80)
        self.tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.tree.setMinimumHeight(200)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        layout.addWidget(self.tree, stretch=1)

        self._scene_path = None

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Scene",
            filter="Scene Files (*.gltf *.glb *.fbx *.obj *.json);;All Files (*)"
        )
        if path:
            self.load_scene(path)

    def _on_reload(self):
        if self._scene_path:
            self.load_scene(self._scene_path)

    def load_scene(self, path: str):
        """Load scene from file and populate tree."""
        self._scene_path = path
        self.tree.clear()
        ext = os.path.splitext(path)[1].lower()

        if ext == '.json':
            self._load_json_scene(path)
        elif ext in ('.gltf', '.glb'):
            self._load_gltf_scene(path)
        elif ext in ('.fbx', '.obj'):
            self._load_mesh_scene(path)

        self.btn_reload.setEnabled(True)

    def _load_json_scene(self, path: str):
        """Load scene exported by Messiah plugin (scene.json)."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.scene_data = data

        # Meshes
        meshes_item = QTreeWidgetItem(self.tree, ["Meshes", "Group",
                                                   f"{len(data.get('meshes', []))} items"])
        for i, m in enumerate(data.get('meshes', [])):
            name = m.get('name', f'Mesh_{i}')
            verts = m.get('vertex_count', '?')
            QTreeWidgetItem(meshes_item, [name, "Mesh", f"{verts} verts"])

        # Materials
        mats_item = QTreeWidgetItem(self.tree, ["Materials", "Group",
                                                 f"{len(data.get('materials', []))} items"])
        for i, m in enumerate(data.get('materials', [])):
            name = m.get('name', f'Material_{i}')
            model = m.get('shading_model', 'DefaultLit')
            QTreeWidgetItem(mats_item, [name, "Material", model])

        # Cameras
        cams_item = QTreeWidgetItem(self.tree, ["Cameras", "Group",
                                                 f"{len(data.get('cameras', []))} items"])
        for i, c in enumerate(data.get('cameras', [])):
            fov = c.get('fov', '?')
            QTreeWidgetItem(cams_item, [f"Camera_{i}", "Camera", f"FOV: {fov}"])

        # Lights
        lights_item = QTreeWidgetItem(self.tree, ["Lights", "Group",
                                                   f"{len(data.get('lights', []))} items"])
        for i, l in enumerate(data.get('lights', [])):
            ltype = l.get('type', 'directional')
            QTreeWidgetItem(lights_item, [f"Light_{i}", "Light", ltype])

        self.tree.expandAll()

        info = f"Scene: {os.path.basename(path)}"
        self.lbl_info.setText(info)
        self.scene_loaded.emit(data)

    def _load_gltf_scene(self, path: str):
        """Load glTF scene."""
        try:
            from io_utils.mesh_io import load_gltf
            data = load_gltf(path)
            self.scene_data = data

            item = QTreeWidgetItem(self.tree, [
                os.path.basename(path), "glTF",
                f"{data.get('vertex_count', '?')} verts"
            ])
            self.tree.expandAll()
            self.lbl_info.setText(f"glTF: {os.path.basename(path)}")
            self.scene_loaded.emit(data)
        except ImportError:
            self.lbl_info.setText("pygltflib not installed")

    def _load_mesh_scene(self, path: str):
        """Load generic mesh (FBX/OBJ) and emit scene data."""
        ext = os.path.splitext(path)[1].lower()

        if ext == '.fbx':
            try:
                from io_utils.fbx_loader import load_fbx
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                scene_data = load_fbx(path, device=device)
            except Exception as e:
                self.lbl_info.setText(f"FBX load failed: {e}")
                import traceback
                traceback.print_exc()
                return
        elif ext == '.obj':
            try:
                import trimesh
                mesh = trimesh.load(path, force='mesh')
                import torch
                import numpy as np
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                verts = np.array(mesh.vertices, dtype=np.float32)
                faces = np.array(mesh.faces, dtype=np.int32)
                normals = np.array(mesh.vertex_normals, dtype=np.float32)
                # Normalize to [-1,1]
                vmin, vmax = verts.min(0), verts.max(0)
                center = (vmin + vmax) / 2
                scale = (vmax - vmin).max() / 2 + 1e-8
                verts = (verts - center) / scale
                # UVs
                if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                    uvs = np.array(mesh.visual.uv, dtype=np.float32)
                else:
                    uvs = np.zeros((len(verts), 2), dtype=np.float32)
                    uvs[:, 0] = verts[:, 0] * 0.5 + 0.5
                    uvs[:, 1] = verts[:, 2] * 0.5 + 0.5
                scene_data = {
                    'vertices': torch.tensor(verts, dtype=torch.float32, device=device),
                    'triangles': torch.tensor(faces, dtype=torch.int32, device=device),
                    'normals': torch.tensor(normals, dtype=torch.float32, device=device),
                    'uvs': torch.tensor(uvs, dtype=torch.float32, device=device),
                    'vertex_count': len(verts),
                    'triangle_count': len(faces),
                    'meshes': [{'name': os.path.basename(path),
                                'vertex_count': len(verts),
                                'triangle_count': len(faces)}],
                    'materials': [{'name': 'OBJ_Default', 'shading_model': 'DefaultLit',
                                   'base_color': [0.8, 0.8, 0.82],
                                   'roughness': 0.5, 'metallic': 0.0}],
                    'source_file': path,
                }
            except Exception as e:
                self.lbl_info.setText(f"OBJ load failed: {e}")
                import traceback
                traceback.print_exc()
                return
        else:
            self.lbl_info.setText(f"Unsupported format: {ext}")
            return

        self.scene_data = scene_data

        # Populate tree
        mesh_list = scene_data.get('meshes', [])
        total_v = scene_data.get('vertex_count', 0)
        total_t = scene_data.get('triangle_count', 0)

        root_item = QTreeWidgetItem(self.tree, [
            os.path.basename(path), ext.upper().lstrip('.'),
            f"{total_v:,} verts, {total_t:,} tris"
        ])
        for m in mesh_list:
            QTreeWidgetItem(root_item, [
                m.get('name', 'Mesh'), "Mesh",
                f"{m.get('vertex_count', '?')} verts"
            ])

        self.tree.expandAll()
        self.lbl_info.setText(
            f"{ext.upper().lstrip('.')}: {os.path.basename(path)}  |  "
            f"{total_v:,} verts, {total_t:,} tris"
        )
        self.scene_loaded.emit(scene_data)

    def load_engine_scene(self, scene_data: dict, world_name: str = ''):
        """
        Populate the scene tree from engine-resolved resources.
        Shows per-mesh textures as child nodes for easy identification.

        Args:
            scene_data: dict from engine_scene_loader.load_engine_scene()
            world_name: display name of the world
        """
        import logging
        logger = logging.getLogger(__name__)

        self.tree.clear()
        self.scene_data = scene_data

        title = world_name or 'Engine Scene'
        total_v = scene_data.get('vertex_count', 0)
        total_t = scene_data.get('triangle_count', 0)

        try:
            # Root info
            root_item = QTreeWidgetItem(self.tree, [
                title, "World", f"{total_v:,} verts, {total_t:,} tris"
            ])

            # Meshes — with per-mesh textures as children
            mesh_list = scene_data.get('meshes', [])
            mesh_tex_map = scene_data.get('mesh_textures', {})
            meshes_item = QTreeWidgetItem(root_item, [
                "Meshes", "Group", f"{len(mesh_list)} loaded"
            ])

            ROLE_LABELS = {
                'base_color': '🎨 Base Color',
                'normal': '🔵 Normal Map',
                'roughness': '⬛ Roughness',
                'metallic': '⚙ Metallic',
                'ao': '☁ AO',
                'emissive': '💡 Emissive',
                'unknown': '❓ Texture',
            }

            for i, m in enumerate(mesh_list):
                name = m.get('name', 'Unknown')
                verts = m.get('vertex_count', '?')
                color = m.get('color')
                mesh_item = QTreeWidgetItem(meshes_item,
                                            [name, "Mesh", f"{verts} verts"])
                # Apply highlight color
                if color and i < len(MESH_HIGHLIGHT_COLORS):
                    try:
                        r, g, b = color
                        mesh_item.setForeground(0, QColor(int(r*255), int(g*255), int(b*255)))
                    except (ValueError, TypeError):
                        pass

                # Per-mesh textures
                mesh_guid = m.get('guid', '')
                tex_list = m.get('textures', mesh_tex_map.get(mesh_guid, []))
                if not isinstance(tex_list, list):
                    tex_list = []
                for tex in tex_list:
                    if not isinstance(tex, dict):
                        continue
                    role = tex.get('role', 'unknown')
                    label = ROLE_LABELS.get(role, f'❓ {role}')
                    tex_name = tex.get('name', 'Unknown')
                    mat_name = tex.get('material_name', '')
                    info_str = f"via {mat_name}" if mat_name else tex.get('guid', '')[:12]
                    tex_item = QTreeWidgetItem(mesh_item,
                                               [f"  {label}: {tex_name}", "Texture", info_str])
                    # Store texture data for right-click menu
                    tex_item.setData(0, Qt.ItemDataRole.UserRole, {
                        'guid': tex.get('guid', ''),
                        'name': tex_name,
                        'role': role,
                        'path': tex.get('path', ''),
                        'mesh_guid': mesh_guid,
                        'mesh_name': name,
                        'material_guid': tex.get('material_guid', ''),
                        'material_name': mat_name,
                    })

            # Materials
            mat_list = scene_data.get('materials', [])
            mats_item = QTreeWidgetItem(root_item, [
                "Materials", "Group", f"{len(mat_list)} items"
            ])
            for m in mat_list:
                name = m.get('name', 'Unknown')
                model = m.get('shading_model', 'DefaultLit')
                QTreeWidgetItem(mats_item, [name, "Material", model])

            # Textures (global list)
            tex_all_list = scene_data.get('textures_loaded', [])
            tex_group = QTreeWidgetItem(root_item, [
                "Textures", "Group", f"{len(tex_all_list)} items"
            ])
            for t in tex_all_list:
                name = t.get('name', 'Unknown')
                QTreeWidgetItem(tex_group, [name, "Texture", t.get('guid', '')])

            self.tree.expandAll()
        except Exception as e:
            logger.error(f"Failed to populate scene tree: {e}", exc_info=True)
            self.tree.clear()
            QTreeWidgetItem(self.tree, [f"Tree build failed: {e}", "Error", ""])

        self.lbl_info.setText(
            f"Engine: {title}  |  {len(scene_data.get('meshes', []))} meshes, "
            f"{len(scene_data.get('materials', []))} materials, "
            f"{len(scene_data.get('textures_loaded', []))} textures  "
            f"(右键贴图可发起优化)"
        )
        self.btn_reload.setEnabled(False)

        # Emit signal so main_window can init the render pipeline
        self.scene_loaded.emit(scene_data)

    def populate_engine_mesh_list(self, mesh_resources: list, world_name: str = ''):
        """
        Show all available engine meshes with checkboxes for selective loading.

        Called by main_window after world is parsed, BEFORE loading any mesh data.

        Args:
            mesh_resources: list of ResourceInfo objects from resolver.get_meshes()
            world_name: display name of the world
        """
        self.tree.clear()
        self._engine_mesh_list = mesh_resources
        self._mesh_check_items = {}

        title = world_name or 'Engine Scene'
        root_item = QTreeWidgetItem(self.tree, [
            title, "World", f"{len(mesh_resources)} meshes available"
        ])

        # Meshes with checkboxes
        meshes_group = QTreeWidgetItem(root_item, [
            "Meshes", "Group", f"{len(mesh_resources)} items — check to load"
        ])

        # Block signals while populating to avoid per-item signal spam
        self.tree.blockSignals(True)
        for i, res in enumerate(mesh_resources):
            name = getattr(res, 'name', f'Mesh_{i}')
            guid = getattr(res, 'guid', '')
            res_type = getattr(res, 'type', 'Mesh')

            item = QTreeWidgetItem(meshes_group, [name, res_type, guid[:12] + '...'])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Unchecked)
            item.setData(0, Qt.ItemDataRole.UserRole, guid)

            # Show predicted color swatch for first 10
            if i < len(MESH_HIGHLIGHT_COLORS):
                r, g, b = MESH_HIGHLIGHT_COLORS[i]
                item.setForeground(0, QColor(int(r*255), int(g*255), int(b*255)))

            self._mesh_check_items[guid] = item
        self.tree.blockSignals(False)

        self.tree.expandAll()
        self.lbl_info.setText(
            f"Engine: {title}  |  {len(mesh_resources)} meshes available  "
            f"(勾选后点击「加载选中网格」)")

        # Show mesh selection buttons
        self.btn_select_all.setVisible(True)
        self.btn_deselect_all.setVisible(True)
        self.btn_load_selected.setVisible(True)

    def get_selected_mesh_guids(self) -> list:
        """Return GUIDs of all checked meshes."""
        selected = []
        for guid, item in self._mesh_check_items.items():
            if item.checkState(0) == Qt.CheckState.Checked:
                selected.append(guid)
        return selected

    def get_selected_count(self) -> int:
        return len(self.get_selected_mesh_guids())

    def _on_item_changed(self, item, column):
        """Handle checkbox toggle — update button label with count."""
        if column == 0 and self._mesh_check_items:
            n = self.get_selected_count()
            self.btn_load_selected.setText(f"▶ 加载选中网格 ({n})")

    def _on_select_all(self):
        self.tree.blockSignals(True)
        for item in self._mesh_check_items.values():
            item.setCheckState(0, Qt.CheckState.Checked)
        self.tree.blockSignals(False)
        n = len(self._mesh_check_items)
        self.btn_load_selected.setText(f"▶ 加载选中网格 ({n})")

    def _on_deselect_all(self):
        self.tree.blockSignals(True)
        for item in self._mesh_check_items.values():
            item.setCheckState(0, Qt.CheckState.Unchecked)
        self.tree.blockSignals(False)
        self.btn_load_selected.setText(f"▶ 加载选中网格 (0)")

    def _on_load_selected(self):
        """Emit signal with selected mesh GUIDs to trigger loading."""
        guids = self.get_selected_mesh_guids()
        if not guids:
            self.lbl_info.setText("请先勾选要加载的网格")
            return
        self.mesh_selection_changed.emit(guids)

    def _on_tree_context_menu(self, pos):
        """Right-click context menu on tree items — optimize texture."""
        item = self.tree.itemAt(pos)
        if item is None:
            return

        tex_data = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(tex_data, dict) or 'role' not in tex_data:
            return

        menu = QMenu(self)
        role = tex_data.get('role', 'unknown')
        tex_name = tex_data.get('name', 'Unknown')
        mesh_name = tex_data.get('mesh_name', '')

        # Optimize as Base Map
        act_base = QAction(f"🎨 优化为 Base Map — {tex_name}", self)
        act_base.triggered.connect(lambda: self._emit_optimize(tex_data, 'base_color'))
        menu.addAction(act_base)

        # Optimize as Normal Map
        act_normal = QAction(f"🔵 优化为 Normal Map — {tex_name}", self)
        act_normal.triggered.connect(lambda: self._emit_optimize(tex_data, 'normal'))
        menu.addAction(act_normal)

        menu.addSeparator()

        # Info action (non-functional, just shows details)
        info_text = f"Mesh: {mesh_name}  |  检测角色: {role}"
        act_info = QAction(info_text, self)
        act_info.setEnabled(False)
        menu.addAction(act_info)

        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _emit_optimize(self, tex_data: dict, optimize_as: str):
        """Emit texture_optimize_requested with full context."""
        request = {
            **tex_data,
            'optimize_as': optimize_as,
        }
        self.texture_optimize_requested.emit(request)
        self.lbl_info.setText(
            f"已选择优化: {tex_data.get('name', '?')} "
            f"(作为 {optimize_as}) — Mesh: {tex_data.get('mesh_name', '?')}"
        )
