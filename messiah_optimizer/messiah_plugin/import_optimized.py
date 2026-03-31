"""
Messiah Plugin: Import optimized results back into the engine.

Runs inside Messiah Editor to apply optimizer outputs (textures, materials).

Confirmed APIs used:
  - MEditor.RefreshResources(paths)       — resource hot reload
  - MResource.RefreshResourceByPath(path) — single resource reload (Editor-only)
  - MRender.RefreshShaderSource()         — shader reload
  - MExecuter.sync(code, type, hint)      — Lua bridge for material updates
"""

import os
import json
import shutil


def _exec_lua(code: str, return_hint: str = '') -> str:
    """Execute Lua code via MExecuter.sync()."""
    import MExecuter
    return MExecuter.sync(code, 2, return_hint)


def import_optimized_results(results_dir: str, options: dict = None):
    """
    Import optimizer results into the current Messiah scene.

    Args:
        results_dir: Directory containing optimizer output (results.json + files)
        options:     Import options:
                     - 'apply_textures': bool (default True)
                     - 'apply_materials': bool (default True)
                     - 'backup_original': bool (default True)
                     - 'trigger_hot_reload': bool (default True)
    """
    if options is None:
        options = {}

    manifest_path = os.path.join(results_dir, 'results.json')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Results manifest not found: {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # ---- Backup originals ----
    if options.get('backup_original', True) and 'textures' in results:
        _backup_originals(results['textures'], results_dir)

    applied_paths = []

    # ---- Apply textures (file copy) ----
    if options.get('apply_textures', True) and 'textures' in results:
        for tex_info in results['textures']:
            path = _apply_texture(tex_info, results_dir)
            if path:
                applied_paths.append(path)

    # ---- Apply material params (via Lua bridge) ----
    if options.get('apply_materials', True) and 'materials' in results:
        for mat_info in results['materials']:
            _apply_material_via_lua(mat_info)

    # ---- Trigger hot reload with confirmed APIs ----
    if options.get('trigger_hot_reload', True):
        _hot_reload(applied_paths)

    print(f"[Optimizer] Results imported from {results_dir}")


def _backup_originals(textures: list, results_dir: str):
    """Backup original textures before replacing."""
    backup_dir = os.path.join(results_dir, 'backup')
    os.makedirs(backup_dir, exist_ok=True)

    for tex_info in textures:
        original_path = tex_info.get('original_path')
        if original_path and os.path.exists(original_path):
            dst = os.path.join(backup_dir, os.path.basename(original_path))
            if not os.path.exists(dst):
                shutil.copy2(original_path, dst)
                print(f"[Optimizer] Backed up: {original_path}")


def _apply_texture(tex_info: dict, results_dir: str) -> str:
    """Copy optimized texture file to its target location.

    Returns the target path if successful, empty string otherwise.
    """
    tex_file = tex_info.get('file')
    target_path = tex_info.get('target_path', '')

    if not tex_file or not target_path:
        return ''

    source = os.path.join(results_dir, tex_file)
    if not os.path.exists(source):
        print(f"[Optimizer] Texture file not found: {source}")
        return ''

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy2(source, target_path)
    print(f"[Optimizer] Copied texture to {target_path}")
    return target_path


def _apply_material_via_lua(mat_info: dict):
    """Apply material parameter changes via MExecuter.sync() Lua bridge.

    The Lua code is a TEMPLATE — [ADAPT] sections must be replaced
    with actual Messiah Lua API calls for material property access.
    """
    mat_name = mat_info.get('name', '')
    if not mat_name:
        return

    # Build a description of what parameters to set
    params_desc = json.dumps(mat_info, ensure_ascii=False)

    lua_code = f'''
        -- [ADAPT] Material parameter update for: {mat_name}
        -- Params to apply: {params_desc}
        --
        -- Replace with actual Messiah Lua API, for example:
        --   local mat = SomeMaterialManager:FindMaterial("{mat_name}")
        --   mat:SetProperty("roughness", value)
        --   mat:SetProperty("metallic", value)
        --
        -- Until adapted, this is a no-op that logs intent.
        print("[Optimizer] update_material template called for: {mat_name}")
        return "template"
    '''
    try:
        _exec_lua(lua_code, 'mat_update')
    except Exception as e:
        print(f"[Optimizer] Material update failed for {mat_name}: {e}")


def _hot_reload(paths: list):
    """Trigger resource hot reload using confirmed APIs."""
    # MEditor.RefreshResources — confirmed to exist
    if paths:
        try:
            import MEditor
            MEditor.RefreshResources(paths)
            print(f"[Optimizer] RefreshResources called for {len(paths)} paths")
        except Exception as e:
            print(f"[Optimizer] MEditor.RefreshResources failed: {e}")

    # MRender.RefreshShaderSource — confirmed to exist
    try:
        import MRender
        MRender.RefreshShaderSource()
        print("[Optimizer] RefreshShaderSource called")
    except Exception as e:
        print(f"[Optimizer] MRender.RefreshShaderSource failed: {e}")

    # MResource.RefreshResourceByPath — confirmed (Editor-only)
    for p in paths:
        try:
            import MResource
            MResource.RefreshResourceByPath(p)
        except Exception:
            break  # likely same error for all
