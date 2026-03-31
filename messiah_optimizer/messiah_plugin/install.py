"""
Messiah Plugin: Install script.

Copies the plugin files into the Messiah Editor scripts directory
and registers the optimizer bridge in the editor startup.
"""

import os
import sys
import shutil
from typing import Optional


PLUGIN_FILES = [
    '__init__.py',
    'export_for_optimizer.py',
    'import_optimized.py',
    'optimizer_server.py',
]

# Files installed to site-packages for auto-start at Python init time
AUTOSTART_FILES = [
    'optimizer_bridge_autostart.py',
    'optimizer_bridge_autostart.pth',
]

STARTUP_SNIPPET_INDENT = '''
    # === nvdiffrast Optimizer Bridge ===
    # Fallback: only starts if autostart (site-packages) did not already bind the port
    try:
        import socket as _sock
        _s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
        _s.settimeout(0.5)
        _already = _s.connect_ex(('127.0.0.1', {port})) == 0
        _s.close()
        if _already:
            print("[Messiah] Optimizer bridge already running on port {port}")
        else:
            global _optimizer_server
            from optimizer_plugin.optimizer_server import OptimizerRPCServer
            _optimizer_server = OptimizerRPCServer(host='127.0.0.1', port={port})
            _optimizer_server.start()
            print("[Messiah] Optimizer bridge started on port {port}")
    except Exception as _e:
        print(f"[Messiah] Optimizer bridge failed: {{_e}}")
    # === End Optimizer Bridge ===
'''


def install(engine_root: str, port: int = 9800,
            auto_start: bool = True) -> bool:
    """
    Install optimizer plugin into Messiah Editor.

    Args:
        engine_root: Path to engine root (e.g., D:\\NewTrunk\\Engine\\src\\Engine)
        port:        RPC server port
        auto_start:  Whether to add auto-start to editor's qtmain.py

    Returns:
        True if installation was successful
    """
    scripts_root = os.path.join(engine_root, 'Editor', 'QtScript')
    if not os.path.isdir(scripts_root):
        print(f"[Install] Editor scripts directory not found: {scripts_root}")
        return False

    # Create plugin directory
    plugin_dir = os.path.join(scripts_root, 'optimizer_plugin')
    os.makedirs(plugin_dir, exist_ok=True)

    # Copy plugin files
    src_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in PLUGIN_FILES:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(plugin_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[Install] Copied {fname}")
        else:
            print(f"[Install] Warning: {fname} not found")

    # PRIMARY: Install autostart files into site-packages
    # The .pth file triggers Python to import optimizer_bridge_autostart.py
    # at interpreter startup, which polls for MExecuter and starts RPC.
    if auto_start:
        site_packages = os.path.join(
            engine_root, 'Editor', 'PythonLib', 'Lib', 'site-packages'
        )
        if os.path.isdir(site_packages):
            for fname in AUTOSTART_FILES:
                src = os.path.join(src_dir, fname)
                dst = os.path.join(site_packages, fname)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    print(f"[Install] Installed {fname} -> site-packages")
                else:
                    print(f"[Install] Warning: {fname} not found")
        else:
            print(f"[Install] Warning: site-packages not found: {site_packages}")
            print(f"[Install]   Falling back to qtmain.py injection only.")

    # FALLBACK: Also inject into qtmain.py init() for when user opens Python panel
    if auto_start:
        startup_file = os.path.join(scripts_root, 'qtmain.py')
        if os.path.exists(startup_file):
            with open(startup_file, 'r', encoding='utf-8') as f:
                content = f.read()

            marker = '# === nvdiffrast Optimizer Bridge ==='
            if marker not in content:
                # Ensure 'import os' exists at top of qtmain.py
                if 'import os' not in content:
                    content = content.replace('# coding:utf-8\n', '# coding:utf-8\nimport os\n', 1)
                snippet = STARTUP_SNIPPET_INDENT.format(port=port)
                # Insert at the end of init() — find 'def tick()' and inject before it
                lines = content.split('\n')
                insert_idx = None
                for i, line in enumerate(lines):
                    if line.startswith('def tick('):
                        insert_idx = i
                        break
                if insert_idx is not None:
                    # Walk backwards to skip blank lines between init() and tick()
                    inject_at = insert_idx
                    lines.insert(inject_at, snippet.rstrip('\n'))
                    content = '\n'.join(lines)
                else:
                    # Fallback: append at end of init() body
                    # Find the last line of init() by looking for dedent after 'def init()'
                    content += '\n' + snippet
                    print("[Install] Warning: 'def tick()' not found, appended at file end")

                with open(startup_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print("[Install] Auto-start injected into init() in qtmain.py")
            else:
                print("[Install] Auto-start already configured.")
        else:
            print(f"[Install] Startup script not found: {startup_file}")

    print("[Install] Plugin installation complete.")
    return True


def uninstall(engine_root: str) -> bool:
    """
    Remove optimizer plugin from Messiah Editor.

    Args:
        engine_root: Path to engine root

    Returns:
        True if uninstallation was successful
    """
    scripts_root = os.path.join(engine_root, 'Editor', 'QtScript')

    # Remove plugin directory
    plugin_dir = os.path.join(scripts_root, 'optimizer_plugin')
    if os.path.isdir(plugin_dir):
        shutil.rmtree(plugin_dir)
        print("[Uninstall] Removed plugin directory.")

    # Remove autostart files from site-packages
    site_packages = os.path.join(
        engine_root, 'Editor', 'PythonLib', 'Lib', 'site-packages'
    )
    for fname in AUTOSTART_FILES:
        fpath = os.path.join(site_packages, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"[Uninstall] Removed {fname} from site-packages")

    # Remove startup snippet from qtmain.py
    startup_file = os.path.join(scripts_root, 'qtmain.py')
    if os.path.exists(startup_file):
        with open(startup_file, 'r', encoding='utf-8') as f:
            content = f.read()

        start_marker = '# === nvdiffrast Optimizer Bridge ==='
        end_marker = '# === End Optimizer Bridge ==='

        if start_marker in content:
            start_idx = content.index(start_marker)
            end_idx = content.index(end_marker) + len(end_marker)
            # Remove the block plus surrounding newlines
            content = content[:start_idx].rstrip('\n') + content[end_idx:]
            with open(startup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print("[Uninstall] Removed auto-start from qtmain.py")

    print("[Uninstall] Plugin uninstallation complete.")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python install.py <engine_root> [--port PORT] [--no-autostart] [--uninstall]")
        sys.exit(1)

    engine = sys.argv[1]

    if '--uninstall' in sys.argv:
        uninstall(engine)
    else:
        p = 9800
        if '--port' in sys.argv:
            idx = sys.argv.index('--port')
            p = int(sys.argv[idx + 1])
        auto = '--no-autostart' not in sys.argv
        install(engine, port=p, auto_start=auto)
