"""Quick smoke test: app startup → scene load → render pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import torch, numpy as np
from PyQt6.QtWidgets import QApplication
from editor.main import _global_exception_handler

sys.excepthook = _global_exception_handler
app = QApplication(sys.argv)

from editor.main_window import OptimizerMainWindow
window = OptimizerMainWindow()
print(f'App OK: device={window._render_device}, sw={window._use_software_renderer}')

V, T = 2000, 1000
device = window._render_device
scene_data = {
    'vertices': torch.randn(V, 3, dtype=torch.float32, device=device),
    'triangles': torch.randint(0, V, (T, 3), dtype=torch.int32, device=device),
    'normals': torch.randn(V, 3, dtype=torch.float32, device=device),
    'uvs': torch.rand(V, 2, dtype=torch.float32, device=device),
    'vertex_colors': torch.rand(V, 3, dtype=torch.float32, device=device),
    'vertex_count': V, 'triangle_count': T,
    'meshes': [{'name': 'Test', 'vertex_count': V, 'triangle_count': T,
                'guid': 'g1', 'color': (0.2, 0.6, 1.0)}],
    'mesh_ranges': [], 'materials': [], 'textures_loaded': [],
    'mesh_textures': {}, 'source': 'engine_selective',
}

window._on_scene_loaded(scene_data)
print(f'Scene loaded: pipeline={type(window.pipeline).__name__}, mesh={window._render_mesh is not None}')
print('FULL FLOW: OK')
