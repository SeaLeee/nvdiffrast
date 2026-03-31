"""
Mesh I/O - Load and export 3D mesh data from various formats.
"""

import os
import numpy as np
import torch
from typing import Optional


def load_gltf(path: str, device: str = 'cuda') -> dict:
    """
    Load a glTF/glb file and return mesh data tensors.

    Returns:
        dict with:
          'vertices':   [V, 3] float32 tensor
          'triangles':  [T, 3] int32 tensor
          'normals':    [V, 3] float32 tensor
          'uvs':        [V, 2] float32 tensor
          'tangents':   [V, 4] float32 tensor (if available)
          'vertex_count': int
          'triangle_count': int
    """
    from pygltflib import GLTF2
    import struct

    gltf = GLTF2.load(path)
    mesh = gltf.meshes[0]
    primitive = mesh.primitives[0]

    result = {'vertex_count': 0, 'triangle_count': 0}

    # Helper to read accessor data
    def read_accessor(accessor_idx):
        accessor = gltf.accessors[accessor_idx]
        buffer_view = gltf.bufferViews[accessor.bufferView]
        buffer = gltf.buffers[buffer_view.buffer]

        # Get binary data
        data = gltf.binary_blob()

        offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)

        type_map = {
            'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4,
            'MAT2': 4, 'MAT3': 9, 'MAT4': 16,
        }
        component_map = {
            5120: ('b', 1), 5121: ('B', 1), 5122: ('h', 2),
            5123: ('H', 2), 5125: ('I', 4), 5126: ('f', 4),
        }

        num_components = type_map[accessor.type]
        fmt_char, byte_size = component_map[accessor.componentType]
        count = accessor.count

        stride = buffer_view.byteStride or (num_components * byte_size)
        values = []

        for i in range(count):
            start = offset + i * stride
            for j in range(num_components):
                val = struct.unpack_from(fmt_char, data, start + j * byte_size)[0]
                values.append(val)

        arr = np.array(values, dtype=np.float32 if fmt_char == 'f' else np.int32)
        arr = arr.reshape(count, num_components)
        return arr

    # Read indices
    if primitive.indices is not None:
        indices = read_accessor(primitive.indices).flatten().astype(np.int32)
        triangles = indices.reshape(-1, 3)
        result['triangles'] = torch.from_numpy(triangles).to(device)
        result['triangle_count'] = len(triangles)

    # Read attributes
    attrs = primitive.attributes

    if attrs.POSITION is not None:
        verts = read_accessor(attrs.POSITION)
        result['vertices'] = torch.from_numpy(verts).float().to(device)
        result['vertex_count'] = len(verts)

    if attrs.NORMAL is not None:
        normals = read_accessor(attrs.NORMAL)
        result['normals'] = torch.from_numpy(normals).float().to(device)

    if attrs.TEXCOORD_0 is not None:
        uvs = read_accessor(attrs.TEXCOORD_0)
        result['uvs'] = torch.from_numpy(uvs).float().to(device)

    if hasattr(attrs, 'TANGENT') and attrs.TANGENT is not None:
        tangents = read_accessor(attrs.TANGENT)
        result['tangents'] = torch.from_numpy(tangents).float().to(device)

    return result


def load_obj(path: str, device: str = 'cuda') -> dict:
    """
    Load a Wavefront OBJ file (basic parser).

    Returns:
        dict with 'vertices', 'triangles', 'normals', 'uvs'
    """
    vertices = []
    normals = []
    uvs = []
    faces_v = []
    faces_vt = []
    faces_vn = []

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v' and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vn' and len(parts) >= 4:
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vt' and len(parts) >= 3:
                uvs.append([float(parts[1]), float(parts[2])])
            elif parts[0] == 'f':
                face_verts = []
                face_uvs = []
                face_norms = []
                for vert in parts[1:]:
                    indices = vert.split('/')
                    face_verts.append(int(indices[0]) - 1)
                    if len(indices) > 1 and indices[1]:
                        face_uvs.append(int(indices[1]) - 1)
                    if len(indices) > 2 and indices[2]:
                        face_norms.append(int(indices[2]) - 1)

                # Triangulate (fan triangulation for n-gons)
                for i in range(1, len(face_verts) - 1):
                    faces_v.append([face_verts[0], face_verts[i], face_verts[i + 1]])
                    if face_uvs:
                        faces_vt.append([face_uvs[0], face_uvs[i], face_uvs[i + 1]])
                    if face_norms:
                        faces_vn.append([face_norms[0], face_norms[i], face_norms[i + 1]])

    result = {
        'vertices': torch.tensor(vertices, dtype=torch.float32, device=device),
        'triangles': torch.tensor(faces_v, dtype=torch.int32, device=device),
        'vertex_count': len(vertices),
        'triangle_count': len(faces_v),
    }

    if normals:
        result['normals'] = torch.tensor(normals, dtype=torch.float32, device=device)
    if uvs:
        result['uvs'] = torch.tensor(uvs, dtype=torch.float32, device=device)

    return result


def prepare_vtx_attr(mesh_data: dict) -> dict:
    """
    Prepare vtx_attr dict from loaded mesh data for the pipeline.

    Returns:
        dict suitable for MessiahDiffPipeline.render()
    """
    vtx_attr = {}

    if 'normals' in mesh_data:
        vtx_attr['normal'] = mesh_data['normals']
    else:
        # Compute flat normals
        vtx_attr['normal'] = _compute_normals(
            mesh_data['vertices'], mesh_data['triangles']
        )

    if 'uvs' in mesh_data:
        vtx_attr['uv'] = mesh_data['uvs']
    else:
        vtx_attr['uv'] = torch.zeros(
            mesh_data['vertex_count'], 2,
            device=mesh_data['vertices'].device
        )

    if 'tangents' in mesh_data:
        vtx_attr['tangent'] = mesh_data['tangents']

    vtx_attr['pos_world'] = mesh_data['vertices']

    return vtx_attr


def _compute_normals(vertices: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Compute per-vertex normals from face normals."""
    v0 = vertices[triangles[:, 0].long()]
    v1 = vertices[triangles[:, 1].long()]
    v2 = vertices[triangles[:, 2].long()]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

    normals = torch.zeros_like(vertices)
    for i in range(3):
        normals.index_add_(0, triangles[:, i].long(), face_normals)

    normals = torch.nn.functional.normalize(normals, dim=-1)
    return normals
