"""
Messiah Bridge - Client-side communication with Messiah Editor.

Connects to the OptimizerServer running inside Messiah Editor
via TCP JSON-RPC to exchange scene data and optimization results.
"""

import socket
import logging
from typing import Optional

from .protocol import Protocol

logger = logging.getLogger(__name__)


class MessiahBridge:
    """
    TCP client for communicating with Messiah Editor.

    Usage:
        bridge = MessiahBridge('127.0.0.1', 9527)
        bridge.connect()
        scene = bridge.pull_scene('/tmp/scene_export')
        bridge.push_texture('optimized.png', 'Assets/Textures/base_color.dds')
        bridge.trigger_hot_reload()
        bridge.disconnect()
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9527,
                 timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self._msg_id = 0

    def connect(self):
        """Connect to Messiah Editor's OptimizerServer."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))
        logger.info(f"Connected to Messiah at {self.host}:{self.port}")

        # Verify with ping
        result = self._send('ping')
        if not isinstance(result, dict) or result.get('status') != 'ok':
            logger.warning(f"Unexpected ping response: {result}")

    def disconnect(self):
        """Close connection."""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
            logger.info("Disconnected from Messiah")

    def is_connected(self) -> bool:
        return self.sock is not None

    def _send(self, method: str, params: dict = None):
        """Send a JSON-RPC request and wait for response."""
        if not self.sock:
            raise ConnectionError("Not connected to Messiah")

        self._msg_id += 1
        Protocol.write_message(self.sock, {
            'jsonrpc': '2.0',
            'method': method,
            'params': params or {},
            'id': self._msg_id,
        })

        response = Protocol.read_message(self.sock)
        if response is None:
            raise ConnectionError("Connection closed by Messiah")

        if 'error' in response:
            raise RuntimeError(f"Messiah error: {response['error']}")

        return response.get('result')

    # ============ Scene Operations ============

    def pull_scene(self, output_dir: str) -> dict:
        """
        Request Messiah to export current scene data.

        Args:
            output_dir: Directory where Messiah should write exported files

        Returns:
            dict with scene metadata (meshes, materials, cameras, lights)
        """
        return self._send('export_scene', {'output_dir': output_dir})

    def capture_frame(self, resolution: tuple = (1024, 1024),
                      camera_params: dict = None) -> dict:
        """
        Request Messiah to capture a rendered frame.

        Args:
            resolution: (width, height) of the capture
            camera_params: Optional camera override

        Returns:
            dict with 'image_path' pointing to the captured PNG
        """
        return self._send('capture_frame', {
            'resolution': list(resolution),
            'camera': camera_params,
        })

    def capture_multiview(self, num_views: int, resolution: tuple = (1024, 1024),
                          output_dir: str = None) -> dict:
        """
        Request Messiah to render multiple views around the selected object.

        Returns:
            dict with 'images': list of image paths, 'cameras': list of camera params
        """
        return self._send('capture_multiview', {
            'num_views': num_views,
            'resolution': list(resolution),
            'output_dir': output_dir,
        })

    # ============ Push Results ============

    def push_texture(self, local_path: str, asset_path: str) -> dict:
        """
        Send optimized texture to Messiah.

        Args:
            local_path: Path to optimized texture file on disk
            asset_path: Target asset path in Messiah project

        Returns:
            dict with status
        """
        return self._send('import_texture', {
            'source': local_path,
            'target': asset_path,
        })

    def push_material(self, material_data: dict, material_name: str) -> dict:
        """
        Send optimized material parameters to Messiah.

        Args:
            material_data: dict of material parameters
            material_name: Name of the material to update

        Returns:
            dict with status
        """
        return self._send('update_material', {
            'name': material_name,
            'params': material_data,
        })

    def trigger_hot_reload(self) -> dict:
        """
        Trigger Messiah to hot-reload shaders and textures.
        Uses ShaderWatcher mechanism.
        """
        return self._send('hot_reload')

    # ============ Camera Sync ============

    def sync_camera(self, camera_params: dict) -> dict:
        """Send camera parameters to sync Messiah viewport."""
        return self._send('camera_update', camera_params)

    def get_camera(self) -> dict:
        """Get current Messiah camera parameters."""
        return self._send('get_camera')
