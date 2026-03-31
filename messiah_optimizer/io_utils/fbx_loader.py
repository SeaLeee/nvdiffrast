"""
FBX Loader — zero-dependency FBX binary parser.

Reads Autodesk FBX binary files (version 7100–7500+) and extracts mesh
geometry (vertices, triangles, normals, UVs) into numpy arrays compatible
with the Messiah optimizer render pipeline.

No external libraries required — parses the binary format directly with
zlib decompression for compressed array properties.
"""

import os
import struct
import zlib
import logging
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

# ── FBX binary constants ────────────────────────────────────────────────
FBX_MAGIC = b'Kaydara FBX Binary  \x00'
FBX_HEADER_SIZE = 27  # 21 magic + 2 padding + 4 version


# ── Low-level FBX binary reader ─────────────────────────────────────────

class FbxNode:
    """Lightweight FBX node with name, properties, and children."""
    __slots__ = ('name', 'props', 'children')

    def __init__(self, name: str, props: list, children: list):
        self.name = name
        self.props = props
        self.children = children

    def find(self, name: str) -> Optional['FbxNode']:
        """Find first child node by name."""
        for c in self.children:
            if c.name == name:
                return c
        return None

    def find_all(self, name: str) -> List['FbxNode']:
        """Find all child nodes by name."""
        return [c for c in self.children if c.name == name]

    def __repr__(self):
        return f'FbxNode({self.name!r}, {len(self.props)} props, {len(self.children)} children)'


def _read_array_property(f, elem_fmt: str, elem_size: int) -> np.ndarray:
    """Read an FBX array property (possibly zlib-compressed)."""
    count, encoding, comp_len = struct.unpack('<III', f.read(12))
    raw = f.read(comp_len)
    if encoding == 1:  # zlib compressed
        raw = zlib.decompress(raw)
    dtype_map = {'d': np.float64, 'f': np.float32, 'i': np.int32, 'l': np.int64}
    return np.frombuffer(raw, dtype=dtype_map[elem_fmt], count=count)


def _read_property(f) -> Any:
    """Read a single FBX property value."""
    tc = f.read(1)
    if not tc:
        return None
    tc = chr(tc[0])

    if tc == 'S':  # string
        length = struct.unpack('<I', f.read(4))[0]
        return f.read(length).decode('utf-8', errors='replace')
    elif tc == 'R':  # raw binary
        length = struct.unpack('<I', f.read(4))[0]
        return f.read(length)
    elif tc == 'I':
        return struct.unpack('<i', f.read(4))[0]
    elif tc == 'L':
        return struct.unpack('<q', f.read(8))[0]
    elif tc == 'F':
        return struct.unpack('<f', f.read(4))[0]
    elif tc == 'D':
        return struct.unpack('<d', f.read(8))[0]
    elif tc == 'Y':
        return struct.unpack('<h', f.read(2))[0]
    elif tc == 'C':
        return struct.unpack('<B', f.read(1))[0]
    elif tc == 'd':
        return _read_array_property(f, 'd', 8)
    elif tc == 'f':
        return _read_array_property(f, 'f', 4)
    elif tc == 'i':
        return _read_array_property(f, 'i', 4)
    elif tc == 'l':
        return _read_array_property(f, 'l', 8)
    else:
        raise ValueError(f'Unknown FBX property type: {tc!r}')


def _read_node(f, version: int) -> Optional[FbxNode]:
    """Read a single FBX node record."""
    if version >= 7500:
        header = f.read(24)
        if len(header) < 24:
            return None
        end_offset, num_props, prop_list_len = struct.unpack('<QQQ', header)
    else:
        header = f.read(12)
        if len(header) < 12:
            return None
        end_offset, num_props, prop_list_len = struct.unpack('<III', header)

    if end_offset == 0:
        return None  # NULL sentinel

    name_len = struct.unpack('<B', f.read(1))[0]
    name = f.read(name_len).decode('ascii', errors='replace')

    # Read properties
    props_start = f.tell()
    props = []
    for _ in range(num_props):
        props.append(_read_property(f))

    # Read children
    f.seek(props_start + prop_list_len)
    children = []
    while f.tell() < end_offset:
        child = _read_node(f, version)
        if child is None:
            break
        children.append(child)

    f.seek(end_offset)
    return FbxNode(name, props, children)


def _parse_fbx_binary(filepath: str) -> List[FbxNode]:
    """Parse FBX binary file and return top-level nodes."""
    with open(filepath, 'rb') as f:
        magic = f.read(21)
        if magic != FBX_MAGIC:
            raise ValueError(f'Not an FBX binary file: {filepath}')

        f.read(2)  # padding
        version = struct.unpack('<I', f.read(4))[0]
        logger.info(f'FBX version: {version}')

        nodes = []
        while True:
            node = _read_node(f, version)
            if node is None:
                break
            nodes.append(node)

    return nodes


# ── Geometry extraction ──────────────────────────────────────────────────

def _triangulate_polygon_indices(poly_vertex_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert FBX PolygonVertexIndex to per-vertex indices and triangle faces.

    FBX uses negative values to mark polygon boundaries: the last index of
    each polygon is stored as ~index (bitwise NOT).  Polygons are fan-
    triangulated into triangles.

    Returns:
        vertex_indices: int32 array of per-polygon-vertex indices (decoded)
        triangles: [T, 3] int32 array of triangle face indices into the
                   polygon-vertex list (for ByPolygonVertex mapping)
    """
    # Decode negative markers
    decoded = poly_vertex_index.copy()
    neg_mask = decoded < 0
    decoded[neg_mask] = ~decoded[neg_mask]  # bitwise NOT to get actual index

    # Find polygon boundaries (each negative value ends a polygon)
    ends = np.where(neg_mask)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])

    triangles = []
    for s, e in zip(starts, ends):
        n = e - s + 1  # number of vertices in polygon
        if n < 3:
            continue
        # Fan triangulation: (s, s+1, s+2), (s, s+2, s+3), ...
        for k in range(1, n - 1):
            triangles.append([s, s + k, s + k + 1])

    return decoded, np.array(triangles, dtype=np.int32) if triangles else np.zeros((0, 3), dtype=np.int32)


def _extract_geometry(geom_node: FbxNode) -> Optional[Dict]:
    """
    Extract mesh data from an FBX Geometry node.

    Returns dict with vertices, triangles, normals, uvs (numpy arrays)
    or None on failure.
    """
    # ── Vertices ─────────────────────────────────────────────────────
    verts_node = geom_node.find('Vertices')
    if verts_node is None or len(verts_node.props) == 0:
        logger.warning('Geometry node has no Vertices')
        return None

    raw_verts = verts_node.props[0]
    if not isinstance(raw_verts, np.ndarray):
        logger.warning('Vertices property is not an array')
        return None

    vertices = raw_verts.astype(np.float32).reshape(-1, 3)
    num_verts = len(vertices)
    logger.info(f'  Vertices: {num_verts}')

    # ── PolygonVertexIndex → triangles ──────────────────────────────
    pvi_node = geom_node.find('PolygonVertexIndex')
    if pvi_node is None or len(pvi_node.props) == 0:
        logger.warning('Geometry node has no PolygonVertexIndex')
        return None

    raw_pvi = pvi_node.props[0]
    if not isinstance(raw_pvi, np.ndarray):
        logger.warning('PolygonVertexIndex is not an array')
        return None

    decoded_pvi, poly_triangles = _triangulate_polygon_indices(raw_pvi)
    num_poly_verts = len(decoded_pvi)
    num_tris = len(poly_triangles)
    logger.info(f'  Polygon vertices: {num_poly_verts}, Triangles: {num_tris}')

    if num_tris == 0:
        return None

    # Build per-polygon-vertex position array
    # poly_triangles indexes into the polygon-vertex list; we need
    # actual vertex positions for each polygon-vertex.
    pv_positions = vertices[decoded_pvi]  # [num_poly_verts, 3]

    # ── Normals ──────────────────────────────────────────────────────
    pv_normals = None
    norm_layer = geom_node.find('LayerElementNormal')
    if norm_layer is not None:
        normals_node = norm_layer.find('Normals')
        mapping_node = norm_layer.find('MappingInformationType')
        ref_node = norm_layer.find('ReferenceInformationType')

        mapping = mapping_node.props[0] if mapping_node and mapping_node.props else ''
        ref_type = ref_node.props[0] if ref_node and ref_node.props else ''

        if normals_node and isinstance(normals_node.props[0], np.ndarray):
            raw_normals = normals_node.props[0].astype(np.float32).reshape(-1, 3)

            if mapping == 'ByPolygonVertex':
                if ref_type == 'IndexToDirect':
                    idx_node = norm_layer.find('NormalsIndex')
                    if idx_node and isinstance(idx_node.props[0], np.ndarray):
                        nidx = idx_node.props[0].astype(np.int32)
                        nidx = np.clip(nidx, 0, len(raw_normals) - 1)
                        pv_normals = raw_normals[nidx]
                elif ref_type == 'Direct':
                    if len(raw_normals) >= num_poly_verts:
                        pv_normals = raw_normals[:num_poly_verts]
            elif mapping == 'ByVertice' or mapping == 'ByVertex':
                # Per-control-point normals — expand to per-polygon-vertex
                if len(raw_normals) >= num_verts:
                    pv_normals = raw_normals[decoded_pvi]

    # ── UVs ──────────────────────────────────────────────────────────
    pv_uvs = None
    uv_layer = geom_node.find('LayerElementUV')
    if uv_layer is not None:
        uv_node = uv_layer.find('UV')
        mapping_node = uv_layer.find('MappingInformationType')
        ref_node = uv_layer.find('ReferenceInformationType')

        mapping = mapping_node.props[0] if mapping_node and mapping_node.props else ''
        ref_type = ref_node.props[0] if ref_node and ref_node.props else ''

        if uv_node and isinstance(uv_node.props[0], np.ndarray):
            raw_uvs = uv_node.props[0].astype(np.float32).reshape(-1, 2)

            if mapping == 'ByPolygonVertex':
                if ref_type == 'IndexToDirect':
                    idx_node = uv_layer.find('UVIndex')
                    if idx_node and isinstance(idx_node.props[0], np.ndarray):
                        uvidx = idx_node.props[0].astype(np.int32)
                        uvidx = np.clip(uvidx, 0, len(raw_uvs) - 1)
                        pv_uvs = raw_uvs[uvidx]
                elif ref_type == 'Direct':
                    if len(raw_uvs) >= num_poly_verts:
                        pv_uvs = raw_uvs[:num_poly_verts]

    # ── Assemble final vertex buffer ─────────────────────────────────
    # Since normals/UVs may be ByPolygonVertex (unique per face-vertex),
    # we use the expanded polygon-vertex layout for everything.
    final_positions = pv_positions[poly_triangles.ravel()].reshape(-1, 3)

    if pv_normals is not None:
        final_normals = pv_normals[poly_triangles.ravel()].reshape(-1, 3)
    else:
        final_normals = None

    if pv_uvs is not None:
        final_uvs = pv_uvs[poly_triangles.ravel()].reshape(-1, 2)
    else:
        final_uvs = None

    # Now each triangle has 3 unique vertices (unindexed)
    num_final_verts = len(final_positions)
    final_triangles = np.arange(num_final_verts, dtype=np.int32).reshape(-1, 3)

    return {
        'vertices': final_positions,
        'triangles': final_triangles,
        'normals': final_normals,
        'uvs': final_uvs,
        'vertex_count': num_final_verts,
        'triangle_count': len(final_triangles),
    }


# ── Public API ────────────────────────────────────────────────────────

def load_fbx(filepath: str, device: str = 'cuda',
             max_meshes: int = 50,
             max_vertices: int = 2_000_000) -> dict:
    """
    Load an FBX file and return a scene dict compatible with the render pipeline.

    Args:
        filepath: path to .fbx file
        device: torch device
        max_meshes: max geometry nodes to load
        max_vertices: safety cap on total vertices

    Returns:
        dict with keys: vertices, triangles, normals, uvs (torch tensors),
        vertex_count, triangle_count, meshes (list of per-mesh info),
        materials (list), source_file
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f'FBX file not found: {filepath}')

    logger.info(f'Loading FBX: {filepath}')
    nodes = _parse_fbx_binary(filepath)

    # Find the Objects node
    objects_node = None
    for n in nodes:
        if n.name == 'Objects':
            objects_node = n
            break

    if objects_node is None:
        raise ValueError('FBX file has no Objects node')

    # Collect all Geometry nodes
    geom_nodes = objects_node.find_all('Geometry')
    logger.info(f'Found {len(geom_nodes)} Geometry nodes')

    all_verts = []
    all_tris = []
    all_normals = []
    all_uvs = []
    mesh_infos = []
    total_v = 0

    for i, gnode in enumerate(geom_nodes[:max_meshes]):
        # Get mesh name from node properties
        name = 'Unknown'
        if len(gnode.props) >= 2:
            name = str(gnode.props[1]).replace('\x00\x01', '_')

        logger.info(f'Parsing geometry [{i}]: {name}')
        mesh = _extract_geometry(gnode)
        if mesh is None:
            logger.warning(f'  Skipped (no valid geometry)')
            continue

        vc = mesh['vertex_count']
        if total_v + vc > max_vertices:
            logger.warning(f'  Vertex cap reached ({total_v + vc} > {max_vertices}), stopping')
            break

        # Sanitize NaN/Inf
        mesh['vertices'] = np.nan_to_num(mesh['vertices'], nan=0.0, posinf=0.0, neginf=0.0)
        if mesh['normals'] is not None:
            mesh['normals'] = np.nan_to_num(mesh['normals'], nan=0.0, posinf=0.0, neginf=0.0)
        if mesh['uvs'] is not None:
            mesh['uvs'] = np.nan_to_num(mesh['uvs'], nan=0.0, posinf=0.0, neginf=0.0)

        offset = total_v
        all_verts.append(mesh['vertices'])
        all_tris.append(mesh['triangles'] + offset)

        if mesh['normals'] is not None:
            all_normals.append(mesh['normals'])
        else:
            all_normals.append(np.zeros_like(mesh['vertices']))

        if mesh['uvs'] is not None:
            all_uvs.append(mesh['uvs'])
        else:
            # Planar UV fallback
            v = mesh['vertices']
            uvs = np.zeros((len(v), 2), dtype=np.float32)
            uvs[:, 0] = v[:, 0]
            uvs[:, 1] = v[:, 2]
            rng = uvs.max() - uvs.min() + 1e-8
            uvs = (uvs - uvs.min()) / rng
            all_uvs.append(uvs)

        total_v += vc
        mesh_infos.append({
            'name': name,
            'vertex_count': vc,
            'triangle_count': mesh['triangle_count'],
        })
        logger.info(f'  OK: {vc} verts, {mesh["triangle_count"]} tris')

    if not all_verts:
        raise ValueError('No valid meshes found in FBX file')

    # Concatenate
    cat_verts = np.concatenate(all_verts, axis=0)
    cat_tris = np.concatenate(all_tris, axis=0)
    cat_normals = np.concatenate(all_normals, axis=0)
    cat_uvs = np.concatenate(all_uvs, axis=0)

    # Final NaN sanitization
    cat_verts = np.nan_to_num(cat_verts, nan=0.0, posinf=0.0, neginf=0.0)
    cat_normals = np.nan_to_num(cat_normals, nan=0.0, posinf=0.0, neginf=0.0)
    cat_uvs = np.nan_to_num(cat_uvs, nan=0.0, posinf=0.0, neginf=0.0)

    # Recompute normals if all-zero
    norm_mag = np.linalg.norm(cat_normals, axis=1)
    if np.mean(norm_mag < 1e-6) > 0.5:
        logger.info('Recomputing normals (most were zero)')
        cat_normals = _compute_normals_np(cat_verts, cat_tris)

    # Normalize to [-1, 1] bounding box
    vmin = cat_verts.min(axis=0)
    vmax = cat_verts.max(axis=0)
    center = (vmin + vmax) / 2
    scale = (vmax - vmin).max() / 2 + 1e-8
    cat_verts = (cat_verts - center) / scale

    total_t = len(cat_tris)
    logger.info(f'FBX loaded: {total_v} verts, {total_t} tris from {len(mesh_infos)} meshes')

    # Convert to torch tensors
    scene = {
        'vertices': torch.tensor(cat_verts, dtype=torch.float32, device=device),
        'triangles': torch.tensor(cat_tris, dtype=torch.int32, device=device),
        'normals': torch.tensor(cat_normals, dtype=torch.float32, device=device),
        'uvs': torch.tensor(cat_uvs, dtype=torch.float32, device=device),
        'vertex_count': total_v,
        'triangle_count': total_t,
        'meshes': mesh_infos,
        'materials': [{'name': 'FBX_Default', 'shading_model': 'DefaultLit',
                       'base_color': [0.8, 0.8, 0.82], 'roughness': 0.5, 'metallic': 0.0}],
        'source_file': filepath,
    }
    return scene


def _compute_normals_np(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals from face-weighted cross products."""
    normals = np.zeros_like(vertices)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return (normals / norms).astype(np.float32)
