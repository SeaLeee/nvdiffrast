"""
Engine Scene Loader — loads Messiah engine resources (binary mesh data, TGA
textures, material XML) resolved by ResourceResolver into renderable scene data.

Parses resource.xml + resource.data directly (vertex streams + indices) instead
of requiring FBX loaders.

Produces a dict compatible with _on_scene_loaded / _init_render_pipeline.
"""

import os
import struct
import logging
import numpy as np
import torch
from typing import Optional, List, Dict, Tuple
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

# ── Vertex format component byte sizes ──────────────────────────────────
COMPONENT_SIZES: Dict[str, int] = {
    'P3F': 12,   # Position   3 × float32
    'P3H': 6,    # Position   3 × float16
    'C4B': 4,    # Color      4 × uint8
    'N4B': 4,    # Normal     4 × uint8  (packed, [0‥255] → [-1,1])
    'T2F': 8,    # TexCoord   2 × float32
    'T2H': 4,    # TexCoord   2 × float16
    'T4F': 16,   # TexCoord   4 × float32 (extended UV)
    'T4H': 8,    # Tangent    4 × float16
    'B4H': 8,    # Binormal   4 × float16
}

# Maximum meshes to load (to avoid OOM on huge worlds)
MAX_MESHES_TO_LOAD = 50
# Maximum total vertices  (safety cap)
MAX_TOTAL_VERTICES = 2_000_000


def load_engine_scene(resolver, world, device: str = 'cuda',
                      max_meshes: int = MAX_MESHES_TO_LOAD) -> dict:
    """
    Load engine resources from a resolved WorldInfo into a renderable scene dict.

    Args:
        resolver: ResourceResolver instance
        world: WorldInfo with all_resources populated
        device: torch device
        max_meshes: max meshes to load

    Returns:
        dict compatible with scene_panel.scene_loaded signal:
        {
            'vertices': [V, 3] tensor,
            'triangles': [T, 3] tensor,
            'normals': [V, 3] tensor,
            'uvs': [V, 2] tensor,
            'vertex_count': int,
            'triangle_count': int,
            'materials': [{'name', 'base_color', 'roughness', 'metallic', ...}],
            'meshes': [{'name', 'vertex_count', 'guid'}],
            'textures_loaded': [{'name', 'guid', 'path'}],
            'base_color_texture': [1, H, W, 3] tensor or None,
            'source': 'engine',
        }
    """
    meshes = resolver.get_meshes(world)
    materials = resolver.get_materials(world)
    textures = resolver.get_textures(world)

    logger.info(f"Loading engine scene: {len(meshes)} meshes, "
                f"{len(materials)} materials, {len(textures)} textures")

    # --- Load meshes ---
    all_verts = []
    all_tris = []
    all_normals = []
    all_uvs = []
    mesh_info_list = []
    vertex_offset = 0
    total_verts = 0

    loaded_count = 0
    skipped_flat = 0
    for res in meshes:
        if loaded_count >= max_meshes:
            break
        if total_verts >= MAX_TOTAL_VERTICES:
            logger.warning(f"Vertex cap reached ({MAX_TOTAL_VERTICES}), skipping remaining meshes")
            break

        mesh_data = _load_mesh_resource(resolver, res)
        if mesh_data is None:
            continue

        # --- Skip flat / degenerate meshes (particle billboards, screen quads) ---
        verts_np = mesh_data['vertices']
        extents = verts_np.max(axis=0) - verts_np.min(axis=0)
        max_ext = extents.max()
        if max_ext > 1e-6:
            # Flatness ratio: if the smallest axis extent is < 1% of the largest,
            # the mesh is essentially a flat plane (particle quad, screen overlay).
            min_ext = extents.min()
            if min_ext < 0.01 * max_ext:
                skipped_flat += 1
                logger.debug(f"Skipped flat mesh '{res.name}': "
                             f"extents=[{extents[0]:.2f}, {extents[1]:.2f}, {extents[2]:.2f}]")
                continue
        else:
            # All extents near zero — point-like, skip
            continue

        nv = mesh_data['vertex_count']
        nt = mesh_data['triangle_count']

        all_verts.append(verts_np)
        all_normals.append(mesh_data['normals'])
        all_uvs.append(mesh_data['uvs'])
        # Offset triangle indices
        all_tris.append(mesh_data['triangles'] + vertex_offset)

        mesh_info_list.append({
            'name': res.name,
            'vertex_count': nv,
            'triangle_count': nt,
            'guid': res.guid,
        })

        vertex_offset += nv
        total_verts += nv
        loaded_count += 1

    if skipped_flat > 0:
        logger.info(f"Skipped {skipped_flat} flat/degenerate meshes (particle quads, screen overlays)")

    if not all_verts:
        logger.warning("No meshes loaded, using placeholder sphere")
        return _placeholder_scene(device)

    # Concatenate all mesh data
    vertices = np.concatenate(all_verts, axis=0)
    triangles = np.concatenate(all_tris, axis=0)
    normals = np.concatenate(all_normals, axis=0)
    uvs = np.concatenate(all_uvs, axis=0)

    # --- Sanitize: replace NaN/Inf with 0 ---
    nan_mask = ~np.isfinite(vertices)
    if nan_mask.any():
        n_bad = nan_mask.any(axis=1).sum()
        logger.warning(f"Replaced NaN/Inf in {n_bad} vertices")
        vertices = np.nan_to_num(vertices, nan=0.0, posinf=0.0, neginf=0.0)
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    uvs = np.nan_to_num(uvs, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Validate triangle indices ---
    valid_mask = (triangles >= 0).all(axis=1) & (triangles < len(vertices)).all(axis=1)
    if not valid_mask.all():
        n_bad = (~valid_mask).sum()
        logger.warning(f"Removed {n_bad} triangles with out-of-range indices")
        triangles = triangles[valid_mask]

    # Center and normalize the scene
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2.0
    extent = (vmax - vmin).max()
    if extent > 0:
        vertices = (vertices - center) / extent * 2.0  # fit in [-1, 1]

    result = {
        'vertices': torch.from_numpy(vertices.astype(np.float32)).to(device),
        'triangles': torch.from_numpy(triangles.astype(np.int32)).to(device),
        'normals': torch.from_numpy(normals.astype(np.float32)).to(device),
        'uvs': torch.from_numpy(uvs.astype(np.float32)).to(device),
        'vertex_count': len(vertices),
        'triangle_count': len(triangles),
        'meshes': mesh_info_list,
        'source': 'engine',
    }

    # --- Load first available texture as base_color ---
    base_color_tex = _load_first_texture(resolver, textures, device)
    if base_color_tex is not None:
        result['base_color_texture'] = base_color_tex

    # --- Parse material metadata ---
    mat_info_list = []
    for res in materials[:50]:  # cap
        mat_data = _parse_material_resource(resolver, res)
        if mat_data:
            mat_info_list.append(mat_data)
    result['materials'] = mat_info_list

    # --- Texture info for scene panel ---
    tex_info_list = []
    for res in textures[:100]:  # cap
        res_dir = resolver.get_resource_dir(res)
        tex_info_list.append({
            'name': res.name,
            'guid': res.guid,
            'path': res_dir or '',
        })
    result['textures_loaded'] = tex_info_list

    logger.info(f"Engine scene loaded: {result['vertex_count']} verts, "
                f"{result['triangle_count']} tris, {loaded_count} meshes, "
                f"{len(mat_info_list)} materials")

    return result


# ── Per-mesh highlight colors for selective loading ─────────────────────
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


def resolve_mesh_textures(resolver, world) -> Dict[str, List[dict]]:
    """
    Build a mapping: mesh_guid → list of associated textures.

    Traverses the dependency chain:
      Model/LodModel → deps → [Mesh, Material, ...]
      Material → deps → [Texture, ...]

    Returns:
        dict of mesh_guid → [{'guid', 'name', 'type', 'role', 'path'}, ...]
        where role is guessed from name/type (e.g. 'base_color', 'normal', 'unknown')
    """
    all_res = world.all_resources

    # Build reverse lookup: child_guid → set of parent ResourceInfo
    child_to_parents: Dict[str, List] = {}
    for info in all_res.values():
        for dep_guid in info.deps:
            child_to_parents.setdefault(dep_guid, []).append(info)

    mesh_guids = set()
    material_guids = set()
    texture_guids = set()
    for info in all_res.values():
        t = info.type.lower()
        if t in ('mesh', 'staticmesh', 'skeletalmesh'):
            mesh_guids.add(info.guid)
        elif t == 'material':
            material_guids.add(info.guid)
        elif t == 'texture':
            texture_guids.add(info.guid)

    result: Dict[str, List[dict]] = {}

    for mesh_guid in mesh_guids:
        textures = []
        seen_tex = set()

        # Find parent Models that reference this mesh
        parents = child_to_parents.get(mesh_guid, [])
        for parent in parents:
            # Parent should be a Model/LodModel — look at its other deps for Materials
            for sibling_guid in parent.deps:
                sib = all_res.get(sibling_guid)
                if not sib or sib.type.lower() != 'material':
                    continue
                # This material is a sibling of our mesh under same Model
                # Follow material's deps to find textures
                for tex_guid in sib.deps:
                    if tex_guid in seen_tex:
                        continue
                    tex_info = all_res.get(tex_guid)
                    if not tex_info or tex_info.type.lower() != 'texture':
                        continue
                    seen_tex.add(tex_guid)

                    role = _guess_texture_role(
                        tex_info.name,
                        getattr(tex_info, 'source_path', '') or '')
                    tex_dir = resolver.get_resource_dir(tex_info)
                    textures.append({
                        'guid': tex_guid,
                        'name': tex_info.name,
                        'type': tex_info.res_class or tex_info.type,
                        'role': role,
                        'material_guid': sib.guid,
                        'material_name': sib.name,
                        'path': tex_dir or '',
                    })

        result[mesh_guid] = textures

    return result


def _guess_texture_role(name: str, source_path: str) -> str:
    """Guess texture role from name/path conventions."""
    combined = (name + ' ' + source_path).lower()
    if any(k in combined for k in ('_n.', '_normal', '_nrm', 'normalmap', '_nm')):
        return 'normal'
    if any(k in combined for k in ('_d.', '_diff', '_base', '_albedo', '_color', 'basemap', 'diffuse')):
        return 'base_color'
    if any(k in combined for k in ('_r.', '_rough', 'roughness')):
        return 'roughness'
    if any(k in combined for k in ('_m.', '_metal', 'metallic', 'metalness')):
        return 'metallic'
    if any(k in combined for k in ('_ao', '_occlusion', 'ambient')):
        return 'ao'
    if any(k in combined for k in ('_e.', '_emissive', 'emission', 'glow')):
        return 'emissive'
    return 'unknown'


def load_engine_scene_selective(resolver, world, selected_guids: List[str],
                                 device: str = 'cuda') -> dict:
    """
    Load only the meshes whose GUIDs are in *selected_guids*.

    Returns the same dict as load_engine_scene(), plus:
      - 'vertex_colors': [V, 3] tensor — per-vertex highlight color
      - 'mesh_ranges': list of {'guid', 'name', 'vert_start', 'vert_end',
                                 'tri_start', 'tri_end', 'color'}
    """
    all_meshes = resolver.get_meshes(world)
    materials = resolver.get_materials(world)
    textures = resolver.get_textures(world)

    # Build GUID set for fast lookup
    guid_set = set(selected_guids)

    # Keep order from selected_guids for deterministic colors
    guid_to_res = {}
    for res in all_meshes:
        if res.guid in guid_set:
            guid_to_res[res.guid] = res

    # Order by selection order
    ordered = []
    for guid in selected_guids:
        if guid in guid_to_res:
            ordered.append(guid_to_res[guid])

    logger.info(f"Selective load: {len(ordered)}/{len(all_meshes)} meshes requested")

    all_verts = []
    all_tris = []
    all_normals = []
    all_uvs = []
    all_colors = []
    mesh_info_list = []
    mesh_ranges = []
    vertex_offset = 0
    tri_offset = 0

    for i, res in enumerate(ordered):
        mesh_data = _load_mesh_resource(resolver, res)
        if mesh_data is None:
            logger.debug(f"Failed to load mesh {res.name} ({res.guid})")
            continue

        nv = mesh_data['vertex_count']
        nt = mesh_data['triangle_count']

        # Assign highlight color (cycle if > palette size)
        color = MESH_HIGHLIGHT_COLORS[i % len(MESH_HIGHLIGHT_COLORS)]
        vtx_color = np.full((nv, 3), color, dtype=np.float32)

        all_verts.append(mesh_data['vertices'])
        all_normals.append(mesh_data['normals'])
        all_uvs.append(mesh_data['uvs'])
        all_tris.append(mesh_data['triangles'] + vertex_offset)
        all_colors.append(vtx_color)

        mesh_info_list.append({
            'name': res.name,
            'vertex_count': nv,
            'triangle_count': nt,
            'guid': res.guid,
            'color': color,
        })
        mesh_ranges.append({
            'guid': res.guid,
            'name': res.name,
            'vert_start': vertex_offset,
            'vert_end': vertex_offset + nv,
            'tri_start': tri_offset,
            'tri_end': tri_offset + nt,
            'color': color,
        })

        vertex_offset += nv
        tri_offset += nt

    if not all_verts:
        logger.warning("No selected meshes loaded, using placeholder sphere")
        return _placeholder_scene(device)

    vertices = np.concatenate(all_verts, axis=0)
    triangles = np.concatenate(all_tris, axis=0)
    normals = np.concatenate(all_normals, axis=0)
    uvs = np.concatenate(all_uvs, axis=0)
    vertex_colors = np.concatenate(all_colors, axis=0)

    # Sanitize
    vertices = np.nan_to_num(vertices, nan=0.0, posinf=0.0, neginf=0.0)
    normals = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    uvs = np.nan_to_num(uvs, nan=0.0, posinf=0.0, neginf=0.0)

    valid_mask = (triangles >= 0).all(axis=1) & (triangles < len(vertices)).all(axis=1)
    if not valid_mask.all():
        triangles = triangles[valid_mask]

    # Center and normalize
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2.0
    extent = (vmax - vmin).max()
    if extent > 0:
        vertices = (vertices - center) / extent * 2.0

    result = {
        'vertices': torch.from_numpy(vertices.astype(np.float32)).to(device),
        'triangles': torch.from_numpy(triangles.astype(np.int32)).to(device),
        'normals': torch.from_numpy(normals.astype(np.float32)).to(device),
        'uvs': torch.from_numpy(uvs.astype(np.float32)).to(device),
        'vertex_colors': torch.from_numpy(vertex_colors).to(device),
        'vertex_count': len(vertices),
        'triangle_count': len(triangles),
        'meshes': mesh_info_list,
        'mesh_ranges': mesh_ranges,
        'source': 'engine_selective',
    }

    # Load first available texture as base_color
    base_color_tex = _load_first_texture(resolver, textures, device)
    if base_color_tex is not None:
        result['base_color_texture'] = base_color_tex

    # Parse material metadata
    mat_info_list = []
    for res in materials[:50]:
        mat_data = _parse_material_resource(resolver, res)
        if mat_data:
            mat_info_list.append(mat_data)
    result['materials'] = mat_info_list

    # Texture info
    tex_info_list = []
    for res in textures[:100]:
        res_dir = resolver.get_resource_dir(res)
        tex_info_list.append({
            'name': res.name,
            'guid': res.guid,
            'path': res_dir or '',
        })
    result['textures_loaded'] = tex_info_list

    # Build per-mesh texture associations via dependency chain
    try:
        mesh_tex_map = resolve_mesh_textures(resolver, world)
        for m_info in mesh_info_list:
            m_info['textures'] = mesh_tex_map.get(m_info['guid'], [])
        result['mesh_textures'] = mesh_tex_map
    except Exception as e:
        logger.warning(f"Failed to resolve mesh-texture associations: {e}")
        result['mesh_textures'] = {}

    logger.info(f"Selective scene loaded: {result['vertex_count']} verts, "
                f"{result['triangle_count']} tris, {len(mesh_info_list)} meshes")

    return result


def _load_mesh_resource(resolver, res) -> Optional[dict]:
    """
    Load a single Mesh resource by parsing resource.xml + resource.data
    (engine binary vertex/index streams).  Falls back to source.fbx if
    the binary parser fails.

    Returns dict with numpy arrays or None on failure.
    """
    res_dir = resolver.get_resource_dir(res)
    if not res_dir:
        return None

    xml_path = os.path.join(res_dir, 'resource.xml')
    data_path = os.path.join(res_dir, 'resource.data')

    # Primary path: binary parser
    if os.path.isfile(xml_path) and os.path.isfile(data_path):
        result = _parse_mesh_binary(xml_path, data_path)
        if result is not None:
            return result

    # Fallback: try source.fbx
    fbx_path = os.path.join(res_dir, 'source.fbx')
    if os.path.isfile(fbx_path):
        try:
            from io_utils.fbx_loader import load_fbx
            scene = load_fbx(fbx_path, device='cpu', max_meshes=1)
            # Convert torch tensors back to numpy for per-mesh dict
            return {
                'vertices': scene['vertices'].cpu().numpy(),
                'triangles': scene['triangles'].cpu().numpy(),
                'normals': scene['normals'].cpu().numpy(),
                'uvs': scene['uvs'].cpu().numpy(),
                'vertex_count': scene['vertex_count'],
                'triangle_count': scene['triangle_count'],
            }
        except Exception as e:
            logger.warning(f'FBX fallback failed for {res.guid}: {e}')

    return None


# ── Vertex format parsing ───────────────────────────────────────────────

def _parse_vertex_format(fmt_str: str) -> List[Tuple[str, int]]:
    """
    Parse a vertex format string like 'P3F_C4B_N4B_T2F' into a list of
    (component_name, byte_size) tuples.
    """
    components = []
    for part in fmt_str.split('_'):
        if not part or part == 'None':
            continue
        size = COMPONENT_SIZES.get(part)
        if size is None:
            logger.warning(f"Unknown vertex component '{part}', guessing 4 bytes")
            size = 4
        components.append((part, size))
    return components


def _parse_mesh_binary(xml_path: str, data_path: str) -> Optional[dict]:
    """
    Parse engine mesh binary (resource.xml metadata + resource.data payload).

    resource.data layout: [Indices][Stream0][Stream1][...]

    Returns dict with vertices [V,3], triangles [T,3], normals [V,3], uvs [V,2].
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        logger.debug(f"Malformed resource.xml {xml_path}: {e}")
        return None

    entity = tree.getroot().find('.//Entity')
    if entity is None:
        return None

    vertex_count = int(entity.findtext('VertexCount', '0'))
    index_count = int(entity.findtext('IndexCount', '0'))
    if vertex_count == 0 or index_count == 0:
        return None

    # --- vertex format for stream 0 ---
    vf_str = entity.findtext('.//VertexFormat/Element[@index="0"]', 'None')
    if vf_str == 'None':
        return None
    components = _parse_vertex_format(vf_str)
    vertex_stride = sum(sz for _, sz in components)
    if vertex_stride == 0:
        return None

    # --- stream sizes ---
    stream_sizes: List[int] = []
    for elem in entity.findall('.//Streams/Element'):
        stream_sizes.append(int(elem.findtext('StreamSize', '0')))
    index_byte_size = int(entity.findtext('.//Indices/StreamSize', '0'))

    # --- read binary payload ---
    with open(data_path, 'rb') as f:
        data = f.read()

    # Layout: [Indices][Stream0][Stream1][Stream2][...]
    index_offset = 0
    stream0_offset = index_byte_size
    # Further streams follow stream0
    stream_offsets = [stream0_offset]
    off = stream0_offset
    for sz in stream_sizes:
        stream_offsets.append(off + sz)
        off += sz

    # Sanity check
    if stream0_offset + vertex_stride * vertex_count > len(data):
        logger.debug(f"resource.data too small for stream0 in {xml_path}")
        return None
    if index_byte_size > len(data):
        logger.debug(f"resource.data too small for indices in {xml_path}")
        return None

    # --- parse stream 0 into numpy arrays --------------------------------
    stream0 = np.frombuffer(
        data, dtype=np.uint8, count=vertex_stride * vertex_count,
        offset=stream0_offset,
    ).reshape(vertex_count, vertex_stride)

    # Build component→offset map (first occurrence only for T2F)
    comp_offset: Dict[str, int] = {}
    off = 0
    for name, sz in components:
        if name not in comp_offset:
            comp_offset[name] = off
        off += sz

    # Positions (P3F = 3 × float32)
    positions: np.ndarray
    if 'P3F' in comp_offset:
        p = comp_offset['P3F']
        positions = np.ascontiguousarray(stream0[:, p:p + 12]).view(np.float32).reshape(vertex_count, 3)
    elif 'P3H' in comp_offset:
        p = comp_offset['P3H']
        positions = np.ascontiguousarray(stream0[:, p:p + 6]).view(np.float16).astype(np.float32).reshape(vertex_count, 3)
    else:
        logger.debug(f"No position component in {vf_str}")
        return None

    # Replace any NaN/Inf in positions early
    positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)

    # Normals (N4B = 4 × uint8, decode [0..255] → [-1,1])
    if 'N4B' in comp_offset:
        n = comp_offset['N4B']
        raw_n = stream0[:, n:n + 3].astype(np.float32)
        normals = raw_n / 127.5 - 1.0
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-8)
    else:
        normals = None  # will compute later

    # UVs (first T2F = 2 × float32)
    if 'T2F' in comp_offset:
        t = comp_offset['T2F']
        uvs = np.ascontiguousarray(stream0[:, t:t + 8]).view(np.float32).reshape(vertex_count, 2)
    elif 'T2H' in comp_offset:
        t = comp_offset['T2H']
        uvs = np.ascontiguousarray(stream0[:, t:t + 4]).view(np.float16).astype(np.float32).reshape(vertex_count, 2)
    else:
        uvs = _generate_planar_uvs(positions)

    # --- parse indices ---------------------------------------------------
    bytes_per_index = index_byte_size // index_count if index_count > 0 else 2
    if bytes_per_index == 4:
        indices = np.frombuffer(data, dtype=np.uint32, count=index_count, offset=index_offset)
    else:
        indices = np.frombuffer(data, dtype=np.uint16, count=index_count, offset=index_offset)

    # Trim to multiple of 3 (Tri_List)
    tri_count = index_count // 3
    indices = indices[:tri_count * 3]
    triangles = indices.astype(np.int32).reshape(-1, 3)

    # Only warn about out-of-range indices, clamp instead of filtering
    max_idx = triangles.max()
    if max_idx >= vertex_count:
        logger.debug(f"Index out of range: max={max_idx} vc={vertex_count}, clamping")
        triangles = np.clip(triangles, 0, vertex_count - 1)

    # Compute normals if not available from vertex format
    if normals is None:
        normals = _compute_normals_np(positions, triangles)

    return {
        'vertices': positions.astype(np.float32),
        'triangles': triangles,
        'normals': normals.astype(np.float32),
        'uvs': uvs.astype(np.float32),
        'vertex_count': int(vertex_count),
        'triangle_count': int(len(triangles)),
    }


def _load_first_texture(resolver, textures, device: str) -> Optional[torch.Tensor]:
    """
    Load the first available RGB source texture (TGA/PNG) as a torch tensor
    [1, H, W, 3] with power-of-two dimensions (required by nvdiffrast mipmap).

    Skips textures that are too small (< 16 pixels in any dimension).
    """
    import torch.nn.functional as F

    for res in textures[:50]:  # try more candidates to find a good one
        res_dir = resolver.get_resource_dir(res)
        if not res_dir:
            continue
        # Look for source image files
        for fname in ('source.tga', 'source.png', 'source.jpg'):
            img_path = os.path.join(res_dir, fname)
            if os.path.exists(img_path):
                try:
                    from io_utils.texture_io import load_texture
                    tex = load_texture(img_path, device=device, srgb_to_linear=True)
                    # [1, H, W, C]

                    # Skip tiny textures (density maps, 1D LUTs, etc.)
                    if tex.shape[1] < 16 or tex.shape[2] < 16:
                        continue

                    # Ensure at least 3 channels (grayscale → RGB)
                    if tex.shape[-1] == 1:
                        tex = tex.expand(-1, -1, -1, 3).contiguous()
                    elif tex.shape[-1] == 4:
                        tex = tex[..., :3].contiguous()

                    # Resize to power-of-two for nvdiffrast mipmapping
                    h, w = tex.shape[1], tex.shape[2]
                    h2 = _next_power_of_two(h)
                    w2 = _next_power_of_two(w)
                    if h2 != h or w2 != w:
                        # F.interpolate expects [B, C, H, W]
                        tex = tex.permute(0, 3, 1, 2)
                        tex = F.interpolate(tex, size=(h2, w2), mode='bilinear',
                                            align_corners=False)
                        tex = tex.permute(0, 2, 3, 1).contiguous()

                    logger.info(f"Loaded texture: {res.name} ({fname}) "
                                f"{tex.shape[1]}x{tex.shape[2]} ch={tex.shape[3]}")
                    return tex
                except Exception as e:
                    logger.debug(f"Failed to load texture {img_path}: {e}")
    return None


def _next_power_of_two(n: int) -> int:
    """Round up to the nearest power of two."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def _parse_material_resource(resolver, res) -> Optional[dict]:
    """
    Parse a material's resource.xml for metadata (name, shading model, etc.).
    """
    res_dir = resolver.get_resource_dir(res)
    if not res_dir:
        return None

    xml_path = os.path.join(res_dir, 'resource.xml')
    if not os.path.exists(xml_path):
        return {'name': res.name, 'guid': res.guid, 'shading_model': 'Unknown'}

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        mat_info = {
            'name': res.name,
            'guid': res.guid,
            'shading_model': 'DefaultLit',
        }

        # Try to find shading model
        for elem in root.iter():
            tag_lower = elem.tag.lower()
            if 'shadingmodel' in tag_lower or 'shading_model' in tag_lower:
                mat_info['shading_model'] = elem.text or 'DefaultLit'
                break

        return mat_info
    except Exception:
        return {'name': res.name, 'guid': res.guid, 'shading_model': 'Unknown'}


def _compute_normals_np(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals from faces (numpy)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)

    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)

    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (normals / norms).astype(np.float32)


def _generate_planar_uvs(vertices: np.ndarray) -> np.ndarray:
    """Generate simple planar UV mapping from XZ coordinates."""
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extent = vmax - vmin
    extent = np.maximum(extent, 1e-8)

    # Project X,Z to U,V
    u = (vertices[:, 0] - vmin[0]) / extent[0]
    v = (vertices[:, 2] - vmin[2]) / extent[2]
    return np.stack([u, v], axis=-1).astype(np.float32)


def _placeholder_scene(device: str) -> dict:
    """Return a minimal placeholder scene when no meshes could be loaded."""
    from pipeline.procedural import create_uv_sphere
    mesh = create_uv_sphere(radius=1.0, rings=32, sectors=64, device=device)
    mesh['materials'] = []
    mesh['meshes'] = [{'name': 'placeholder_sphere', 'vertex_count': mesh['vertex_count']}]
    mesh['textures_loaded'] = []
    mesh['source'] = 'engine_placeholder'
    return mesh
