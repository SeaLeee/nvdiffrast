"""
Messiah Plugin: JSON-RPC server running inside the Messiah Editor.

Confirmed Python APIs used (from C++ binding source code):
  - MExecuter.sync(code: str, returnType: int, returnHint: str) -> str
  - MEditor.RefreshResources(paths)
  - MEditor.RunCommand(command)
  - MRender.RefreshShaderSource()
  - MResource.RefreshResourceByPath(path)

For scene/render operations without direct Python bindings, Lua code
is executed via MExecuter.sync(). Lua templates are marked with
[ADAPT] comments — they MUST be filled in with actual Messiah Lua API
calls for your engine version.

To discover available Lua APIs, run in the engine Python console:
  import MExecuter
  print(MExecuter.sync("local t={} for k,v in pairs(_G) do t[#t+1]=k end return table.concat(t,'\\n')", 2, ""))
"""

import json
import os
import shutil
import socket
import struct
import threading
import traceback
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Lua bridge helpers — MExecuter.sync() is the ONLY fully confirmed API
# ---------------------------------------------------------------------------

def _exec_lua(code: str, return_hint: str = '') -> str:
    """Execute Lua code and return a string result.

    Confirmed signature: MExecuter.sync(code, returnType, returnHint)
      returnType: 0=void, 1=int, 2=string, 3=float
    """
    import MExecuter
    return MExecuter.sync(code, 2, return_hint)


def _exec_lua_void(code: str):
    """Execute Lua code without a return value."""
    import MExecuter
    MExecuter.sync(code, 0, '')


# ---------------------------------------------------------------------------
# RPC Server
# ---------------------------------------------------------------------------

class OptimizerRPCServer:
    """JSON-RPC server for optimizer <-> Messiah Editor communication."""

    def __init__(self, host: str = '127.0.0.1', port: int = 9800):
        self.host = host
        self.port = port
        self._server_socket: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handlers: dict[str, Callable] = {}
        self._register_default_handlers()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()
        print(f"[OptimizerServer] Listening on {self.host}:{self.port}")

    def stop(self):
        self._running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[OptimizerServer] Stopped.")

    def register_handler(self, method: str, handler: Callable):
        """Register or override an RPC method handler."""
        self._handlers[method] = handler

    # ---- TCP infrastructure (pure Python stdlib) ----

    def _serve_loop(self):
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.settimeout(1.0)
        try:
            self._server_socket.bind((self.host, self.port))
            self._server_socket.listen(1)
            while self._running:
                try:
                    conn, addr = self._server_socket.accept()
                    print(f"[OptimizerServer] Client connected: {addr}")
                    t = threading.Thread(
                        target=self._handle_client, args=(conn,), daemon=True
                    )
                    t.start()
                except socket.timeout:
                    continue
                except OSError:
                    break
        except Exception as e:
            print(f"[OptimizerServer] Error: {e}")
        finally:
            try:
                self._server_socket.close()
            except Exception:
                pass

    def _handle_client(self, conn: socket.socket):
        try:
            conn.settimeout(5.0)
            while self._running:
                try:
                    length_data = self._recv_exact(conn, 4)
                    if not length_data:
                        break
                    msg_len = struct.unpack('<I', length_data)[0]
                    if msg_len > 10 * 1024 * 1024:
                        break
                    payload = self._recv_exact(conn, msg_len)
                    if not payload:
                        break
                    request = json.loads(payload.decode('utf-8'))
                    response = self._dispatch(request)
                    resp_bytes = json.dumps(response).encode('utf-8')
                    conn.sendall(struct.pack('<I', len(resp_bytes)))
                    conn.sendall(resp_bytes)
                except socket.timeout:
                    continue
                except ConnectionError:
                    break
        except Exception as e:
            print(f"[OptimizerServer] Client error: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _recv_exact(self, conn: socket.socket, n: int) -> Optional[bytes]:
        data = b''
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _dispatch(self, request: dict) -> dict:
        req_id = request.get('id')
        method = request.get('method', '')
        params = request.get('params', {})
        handler = self._handlers.get(method)
        if handler is None:
            return {
                'jsonrpc': '2.0', 'id': req_id,
                'error': {'code': -32601, 'message': f'Method not found: {method}'}
            }
        try:
            result = handler(params)
            return {'jsonrpc': '2.0', 'id': req_id, 'result': result}
        except Exception as e:
            traceback.print_exc()
            return {
                'jsonrpc': '2.0', 'id': req_id,
                'error': {'code': -32000, 'message': str(e)}
            }

    # ================================================================
    # Handlers — method names match client-side MessiahBridge._send()
    # ================================================================

    def _register_default_handlers(self):
        self._handlers.update({
            'ping':             self._handle_ping,
            'export_scene':     self._handle_export_scene,
            'capture_frame':    self._handle_capture_frame,
            'capture_multiview': self._handle_capture_multiview,
            'import_texture':   self._handle_import_texture,
            'update_material':  self._handle_update_material,
            'hot_reload':       self._handle_hot_reload,
            'get_camera':       self._handle_get_camera,
            'camera_update':    self._handle_camera_update,
        })

    # ---- ping ----

    @staticmethod
    def _handle_ping(params):
        return {'status': 'ok', 'engine': 'Messiah'}

    # ---- export_scene ----

    @staticmethod
    def _handle_export_scene(params):
        """Export scene data via MExecuter.sync() Lua bridge.

        The Lua code below is a TEMPLATE. Lines marked [ADAPT] must be
        replaced with actual Messiah Lua API calls.
        """
        import tempfile

        output_dir = params.get('output_dir',
                                tempfile.mkdtemp(prefix='opt_scene_'))
        os.makedirs(output_dir, exist_ok=True)
        safe_dir = output_dir.replace(os.sep, '/')

        lua_code = f'''
            local json = require("rapidjson")
            local scene = {{
                meshes = {{}},
                materials = {{}},
                cameras = {{}},
                lights = {{}},
            }}

            -- [ADAPT] Replace this block with your engine's Lua scene API.
            -- To discover available globals, run:
            --   for k,v in pairs(_G) do print(k, type(v)) end
            -- Example patterns (NOT real API — fill in after discovery):
            --   local objs = SomeSceneManager:GetAllObjects()
            --   for _, obj in ipairs(objs) do ... end

            scene.note = "Template — replace [ADAPT] blocks with real Lua API"

            local out_path = "{safe_dir}/scene.json"
            local f = io.open(out_path, "w")
            if f then
                f:write(json.encode(scene))
                f:close()
            end
            return out_path
        '''
        try:
            result = _exec_lua(lua_code, 'scene_path')
            manifest = result.strip() if result else ''
            if manifest and os.path.exists(manifest):
                with open(manifest, 'r', encoding='utf-8') as f:
                    scene_data = json.load(f)
                scene_data['export_dir'] = output_dir
                return scene_data
        except Exception as e:
            print(f"[OptimizerServer] export_scene failed: {e}")

        return {
            'export_dir': output_dir,
            'meshes': [], 'materials': [], 'cameras': [], 'lights': [],
            'error': str(e) if 'e' in dir() else 'unknown',
            'note': 'Adapt Lua template in optimizer_server.py _handle_export_scene',
        }

    # ---- capture_frame ----

    @staticmethod
    def _handle_capture_frame(params):
        """Capture rendered frame via MPlatform.ScreenShot.

        Confirmed API (from Editor/QtScript/qtmain.py):
          MPlatform.ScreenShot(callback)
          callback(path: str, width: int, height: int)

        The engine saves the screenshot to a temp path and invokes the
        callback with (path, width, height).  We use a threading.Event
        to wait synchronously for the async callback.
        """
        import tempfile
        import threading
        import time

        resolution = params.get('resolution', [1024, 1024])
        camera = params.get('camera')

        # Sync camera first if provided
        if camera:
            try:
                OptimizerRPCServer._handle_camera_update(camera)
            except Exception:
                pass

        # Use MPlatform.ScreenShot with a synchronizing callback
        result_holder = {'path': None, 'width': 0, 'height': 0}
        done_event = threading.Event()

        def _on_screenshot(path, width, height):
            result_holder['path'] = path
            result_holder['width'] = width
            result_holder['height'] = height
            done_event.set()

        try:
            import MPlatform
            MPlatform.ScreenShot(_on_screenshot)

            # Wait up to 10 seconds for the screenshot callback
            if done_event.wait(timeout=10.0):
                shot_path = result_holder['path']
                if shot_path and os.path.exists(shot_path):
                    # Copy to a stable temp location (engine may reuse the path)
                    tmp = tempfile.NamedTemporaryFile(
                        suffix='.png', delete=False, prefix='opt_cap_')
                    tmp.close()
                    shutil.copy2(shot_path, tmp.name)
                    return {
                        'image_path': tmp.name,
                        'width': result_holder['width'],
                        'height': result_holder['height'],
                        'source': 'MPlatform.ScreenShot',
                    }
                else:
                    return {'error': f'Screenshot file not found: {shot_path}'}
            else:
                return {'error': 'Screenshot timed out (10s)'}
        except ImportError:
            print("[OptimizerServer] MPlatform not available — not running inside Messiah Editor")
            return {
                'error': 'MPlatform not available — must run inside Messiah Editor',
                'hint': 'This handler only works when the plugin runs inside the engine process',
            }
        except Exception as e:
            print(f"[OptimizerServer] capture_frame: {e}")
            return {'error': str(e)}

    # ---- capture_multiview ----

    @staticmethod
    def _handle_capture_multiview(params):
        """Capture multiple views around an object using MPlatform.ScreenShot.

        Rotates the camera around Y-axis and captures at each angle.
        """
        import math
        import tempfile
        import threading

        num_views = params.get('num_views', 8)
        resolution = params.get('resolution', [1024, 1024])
        elevation = params.get('elevation', 20.0)
        distance = params.get('distance', 5.0)
        output_dir = params.get('output_dir',
                                tempfile.mkdtemp(prefix='opt_mv_'))
        os.makedirs(output_dir, exist_ok=True)

        images = []
        cameras = []

        for i in range(num_views):
            angle = 360.0 * i / num_views
            rad = math.radians(angle)
            elev_rad = math.radians(elevation)

            # Camera position on orbit
            cam_x = distance * math.cos(elev_rad) * math.sin(rad)
            cam_y = distance * math.sin(elev_rad)
            cam_z = distance * math.cos(elev_rad) * math.cos(rad)

            cam_params = {
                'position': [cam_x, cam_y, cam_z],
                'target': [0, 0, 0],
                'fov': 45.0,
            }

            # Set camera via Lua
            try:
                safe_pos = f"{cam_x},{cam_y},{cam_z}"
                lua_code = f'''
                    -- [ADAPT] Set camera position/target for multiview capture
                    -- Example: SetViewportCamera({safe_pos}, 0,0,0)
                '''
                _exec_lua_void(lua_code)
            except Exception:
                pass

            # Capture this view
            result = OptimizerRPCServer._handle_capture_frame({
                'resolution': resolution,
            })
            if 'image_path' in result:
                # Move to output_dir with stable name
                dest = os.path.join(output_dir, f'view_{i:03d}.png')
                shutil.move(result['image_path'], dest)
                images.append(dest)
                cameras.append(cam_params)

        return {
            'images': images,
            'cameras': cameras,
            'output_dir': output_dir,
            'num_captured': len(images),
        }

    # ---- import_texture ----

    @staticmethod
    def _handle_import_texture(params):
        """Copy optimized texture into project and trigger refresh.

        Uses confirmed API: MEditor.RefreshResources(paths)
        """
        source = params.get('source', '')
        target = params.get('target', '')

        if not source or not os.path.exists(source):
            return {'status': 'error', 'message': f'Source not found: {source}'}
        if not target:
            return {'status': 'error', 'message': 'No target path specified'}

        # Copy file to project location
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(source, target)

        # Trigger resource refresh — confirmed API
        try:
            import MEditor
            MEditor.RefreshResources([target])
        except Exception as e:
            print(f"[OptimizerServer] RefreshResources failed: {e}")
            # Also try single-resource refresh
            try:
                import MResource
                MResource.RefreshResourceByPath(target)
            except Exception:
                pass

        return {'status': 'ok', 'target': target}

    # ---- update_material ----

    @staticmethod
    def _handle_update_material(params):
        """Update material parameters via Lua bridge.

        No confirmed Python API for material property access.
        All material operations go through MExecuter.sync().
        """
        mat_name = params.get('name', '')
        mat_params = params.get('params', {})

        if not mat_name:
            return {'status': 'error', 'message': 'No material name specified'}

        # Build Lua property-setting code from params
        set_lines = []
        for key, value in mat_params.items():
            if isinstance(value, (list, tuple)):
                vals = ', '.join(str(v) for v in value)
                set_lines.append(f'-- [ADAPT] mat:Set{key}({vals})')
            else:
                set_lines.append(f'-- [ADAPT] mat:Set{key}({value})')

        sets_block = '\n            '.join(set_lines) if set_lines else '-- no params'

        lua_code = f'''
            -- [ADAPT] Replace with actual material access API
            -- local mat = SomeMaterialManager:FindMaterial("{mat_name}")
            {sets_block}
            error("update_material: [ADAPT] not yet implemented")
        '''
        try:
            _exec_lua(lua_code, 'mat_result')
            return {'status': 'ok', 'material': mat_name}
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'note': 'Adapt Lua template in _handle_update_material',
            }

    # ---- hot_reload ----

    @staticmethod
    def _handle_hot_reload(params):
        """Trigger resource hot reload.

        Uses confirmed APIs:
          - MEditor.RefreshResources(paths)
          - MRender.RefreshShaderSource()
          - MResource.RefreshResourceByPath(path)
        """
        paths = params.get('paths', [])
        errors = []

        # Refresh specific resources
        if paths:
            try:
                import MEditor
                MEditor.RefreshResources(paths)
            except Exception as e:
                errors.append(f'MEditor.RefreshResources: {e}')

        # Refresh shaders
        if params.get('shaders', True):
            try:
                import MRender
                MRender.RefreshShaderSource()
            except Exception as e:
                errors.append(f'MRender.RefreshShaderSource: {e}')

        # Refresh individual resource paths
        for p in paths:
            try:
                import MResource
                MResource.RefreshResourceByPath(p)
            except Exception as e:
                errors.append(f'MResource.RefreshResourceByPath({p}): {e}')
                break  # likely same error for all, stop

        if errors:
            return {'status': 'partial', 'errors': errors}
        return {'status': 'ok'}

    # ---- get_camera ----

    @staticmethod
    def _handle_get_camera(params):
        """Get camera transform.

        Confirmed API exists: MEditor.GetCameraTransformFromAffiliatedResourceView
        Exact signature unknown — wrapped in try/except.
        """
        # Try confirmed API (signature guarded by try/except)
        try:
            import MEditor
            transform = MEditor.GetCameraTransformFromAffiliatedResourceView(0)
            return {'transform': transform, 'source': 'MEditor'}
        except Exception as e:
            print(f"[OptimizerServer] GetCameraTransform: {e}")

        # Fallback: Lua bridge
        lua_code = '''
            -- [ADAPT] Get camera transform via Lua API
            -- local cam = GetMainCamera()
            -- return string.format("%f,%f,%f", cam:GetPosition():Get())
            error("get_camera: [ADAPT] not yet implemented")
        '''
        try:
            result = _exec_lua(lua_code, 'camera_data')
            return {'raw': result, 'source': 'lua'}
        except Exception as e2:
            return {'error': str(e2), 'hint': 'Adapt get_camera Lua template'}

    # ---- camera_update ----

    @staticmethod
    def _handle_camera_update(params):
        """Set camera transform.

        Confirmed API exists: MEditor.SetCameraTransformFromAffiliatedResourceView
        Exact signature unknown — wrapped in try/except.
        """
        try:
            import MEditor
            transform = params.get('transform')
            if transform:
                MEditor.SetCameraTransformFromAffiliatedResourceView(0, transform)
                return {'status': 'ok', 'source': 'MEditor'}
        except Exception as e:
            print(f"[OptimizerServer] SetCameraTransform: {e}")

        # Fallback: Lua bridge
        lua_code = '''
            -- [ADAPT] Set camera transform via Lua API
            error("camera_update: [ADAPT] not yet implemented")
        '''
        try:
            _exec_lua_void(lua_code)
            return {'status': 'ok', 'source': 'lua'}
        except Exception as e2:
            return {'status': 'error', 'message': str(e2)}
