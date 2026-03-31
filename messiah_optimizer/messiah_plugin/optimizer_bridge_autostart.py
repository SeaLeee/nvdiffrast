"""
Auto-start the Optimizer RPC bridge when the engine's embedded Python initializes.

This module is placed in Engine/Editor/PythonLib/Lib/site-packages/ and triggered
by optimizer_bridge_autostart.pth during Python interpreter startup.

Timeline:
  1. EditorPythonModule constructor → Py_InitializeFromConfig → site module
     processes .pth → this module starts a background thread
  2. EditorPythonModule::Initialize() → initMExecuter() → MExecuter available
  3. Background thread detects MExecuter → starts RPC server on port 9800

No engine C++ modification required.
"""

import os
import sys
import threading
import time

_bridge_server = None
_bridge_started = False
_bridge_port = 9800


def _log_path():
    """Resolve log path: Editor/QtScript/optimizer_bridge.log"""
    for p in sys.path:
        if p.replace('\\', '/').endswith('/QtScript') or p.replace('\\', '/').endswith('/QtScript/'):
            return os.path.join(p, 'optimizer_bridge.log')
    # Fallback: derive from site-packages → ../../QtScript
    sp = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(sp, '..', '..', '..', 'QtScript', 'optimizer_bridge.log'))


_LOG_PATH = _log_path()


def _log(msg):
    try:
        with open(_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(f'[{time.strftime("%H:%M:%S")}] {msg}\n')
    except Exception:
        pass


def _wait_and_start():
    global _bridge_server, _bridge_started

    # Clear previous log
    try:
        with open(_LOG_PATH, 'w', encoding='utf-8') as f:
            f.write(f'[{time.strftime("%H:%M:%S")}] autostart thread started\n')
            f.write(f'sys.path: {sys.path}\n')
    except Exception:
        pass

    # Wait for MExecuter to become available (up to 120 seconds)
    for attempt in range(1200):
        try:
            import MExecuter  # noqa: F401
            _log(f'MExecuter available after {attempt * 0.1:.1f}s')
            break
        except ImportError:
            time.sleep(0.1)
    else:
        _log('TIMEOUT: MExecuter not available after 120s, giving up')
        return

    # Small extra delay for full initialization
    time.sleep(0.5)

    # Check if server already running (e.g. started by qtmain.py injection)
    import socket as _sock
    try:
        _s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        _s.settimeout(0.5)
        result = _s.connect_ex(('127.0.0.1', _bridge_port))
        _s.close()
        if result == 0:
            _log(f'port {_bridge_port} already in use, skipping')
            _bridge_started = True
            return
    except Exception:
        pass

    # Start RPC server
    try:
        from optimizer_plugin.optimizer_server import OptimizerRPCServer
        _log('import OptimizerRPCServer OK')
        _bridge_server = OptimizerRPCServer(host='127.0.0.1', port=_bridge_port)
        _bridge_server.start()
        _bridge_started = True
        _log(f'RPC server started on port {_bridge_port}')
    except Exception as e:
        import traceback
        _log(f'ERROR: {e}')
        _log(traceback.format_exc())


# Launch background thread immediately on import
_thread = threading.Thread(
    target=_wait_and_start,
    daemon=True,
    name='optimizer-bridge-autostart',
)
_thread.start()
