"""
Messiah Plugin: Export scene data for the optimizer.

Runs inside Messiah Editor's Python environment.
Uses ONLY confirmed APIs — all engine data access goes through
MExecuter.sync() to the Lua runtime.

Confirmed APIs used:
  - MExecuter.sync(code, returnType, returnHint)  — Lua bridge
  - MEngine.GetFileSystemBasePath(name)           — project paths
  - MResource.QueryResourceByType(type)           — enumerate resources (Editor-only)
"""

import os
import json
from typing import Optional


def _exec_lua(code: str, return_hint: str = '') -> str:
    """Execute Lua code and return string result via MExecuter.sync()."""
    import MExecuter
    return MExecuter.sync(code, 2, return_hint)


def export_scene_for_optimizer(output_dir: str, options: dict = None) -> str:
    """
    Export the current Messiah scene for use by the optimizer.

    Must be called from within Messiah Editor's Python environment.

    Args:
        output_dir:  Directory to write exported data
        options:     Export options dict

    Returns:
        Path to the scene.json manifest file.
    """
    if options is None:
        options = {}

    os.makedirs(output_dir, exist_ok=True)
    safe_dir = output_dir.replace(os.sep, '/')

    # The actual scene export is delegated to Lua via MExecuter.sync().
    # The Lua code below is a TEMPLATE — lines marked [ADAPT] must be
    # replaced with actual Messiah Lua API calls.
    lua_code = f'''
        local json = require("rapidjson")
        local scene = {{
            format_version = "1.0",
            meshes = {{}},
            materials = {{}},
            cameras = {{}},
            lights = {{}},
        }}

        -- [ADAPT] Scene object enumeration
        -- Replace with actual Messiah Lua API for iterating scene objects.
        -- To discover available APIs, run:
        --   for k,v in pairs(_G) do print(k, type(v)) end
        --
        -- Example pattern (NOT real API):
        --   local scene_mgr = GetSceneManager()
        --   local objects = scene_mgr:GetAllObjects()
        --   for _, obj in ipairs(objects) do
        --       local mesh_info = {{
        --           name = obj:GetName(),
        --           position = {{obj:GetWorldPosition():Get()}},
        --           material = obj:GetMaterialName(),
        --       }}
        --       table.insert(scene.meshes, mesh_info)
        --   end

        -- [ADAPT] Camera export
        -- Example pattern:
        --   local cam = GetMainCamera()
        --   table.insert(scene.cameras, {{
        --       name = "MainCamera",
        --       position = {{cam:GetPosition():Get()}},
        --       fov = cam:GetFOV(),
        --   }})

        scene.note = "Template — replace [ADAPT] blocks with real Messiah Lua API"

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
        manifest_path = result.strip() if result else ''
        if manifest_path and os.path.exists(manifest_path):
            print(f"[Optimizer] Scene exported to {output_dir}")
            return manifest_path
    except Exception as e:
        print(f"[Optimizer] Lua scene export failed: {e}")

    # Fallback: write a minimal manifest so the caller has something to work with
    manifest_path = os.path.join(output_dir, 'scene.json')
    fallback = {
        'format_version': '1.0',
        'meshes': [],
        'materials': [],
        'cameras': [],
        'lights': [],
        'note': 'Lua scene export template not yet adapted — see export_for_optimizer.py',
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(fallback, f, indent=2, ensure_ascii=False)

    print(f"[Optimizer] Wrote fallback manifest to {manifest_path}")
    return manifest_path
