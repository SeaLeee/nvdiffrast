"""
Procedural geometry generation for preview rendering.
Creates simple meshes (sphere, cube, plane) with UVs and normals.
"""

import math
import torch
import numpy as np


def create_uv_sphere(radius: float = 1.0, rings: int = 32, sectors: int = 64,
                     device: str = 'cuda') -> dict:
    """
    Create a UV sphere mesh.

    Returns:
        dict with 'vertices' [V,3], 'triangles' [T,3], 'normals' [V,3], 'uvs' [V,2]
    """
    verts = []
    normals = []
    uvs = []

    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(sectors + 1):
            theta = 2.0 * math.pi * s / sectors

            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)

            verts.append([x * radius, y * radius, z * radius])
            normals.append([x, y, z])
            uvs.append([s / sectors, r / rings])

    tris = []
    for r in range(rings):
        for s in range(sectors):
            a = r * (sectors + 1) + s
            b = a + sectors + 1

            if r != 0:
                tris.append([a, b, a + 1])
            if r != rings - 1:
                tris.append([a + 1, b, b + 1])

    verts_t = torch.tensor(verts, dtype=torch.float32, device=device)
    normals_t = torch.tensor(normals, dtype=torch.float32, device=device)
    uvs_t = torch.tensor(uvs, dtype=torch.float32, device=device)
    tris_t = torch.tensor(tris, dtype=torch.int32, device=device)

    return {
        'vertices': verts_t,
        'triangles': tris_t,
        'normals': normals_t,
        'uvs': uvs_t,
        'vertex_count': len(verts),
        'triangle_count': len(tris),
    }


def create_default_textures(resolution: int = 256, device: str = 'cuda') -> dict:
    """
    Create default PBR texture set for preview.

    Returns:
        dict with 'base_color' [1,H,W,3], 'roughness' [1,H,W,1], 'metallic' [1,H,W,1]
    """
    H = W = resolution

    # Checkerboard base color
    checker = np.zeros((H, W, 3), dtype=np.float32)
    block = max(H // 8, 1)
    for y in range(H):
        for x in range(W):
            if ((y // block) + (x // block)) % 2 == 0:
                checker[y, x] = [0.8, 0.8, 0.82]
            else:
                checker[y, x] = [0.3, 0.35, 0.4]

    base_color = torch.tensor(checker, dtype=torch.float32, device=device).unsqueeze(0)
    roughness = torch.full((1, H, W, 1), 0.4, dtype=torch.float32, device=device)
    metallic = torch.full((1, H, W, 1), 0.0, dtype=torch.float32, device=device)

    return {
        'base_color': base_color,
        'roughness': roughness,
        'metallic': metallic,
    }


def create_solid_color_textures(color, roughness_val=0.5, metallic_val=0.0,
                                resolution=64, device='cuda') -> dict:
    """
    Create solid-color PBR textures from scalar values.

    Args:
        color: [3] list/tuple of RGB values in [0,1]
        roughness_val: scalar roughness
        metallic_val: scalar metallic
    """
    H = W = resolution
    base_color = torch.tensor(color, dtype=torch.float32, device=device)
    base_color = base_color.view(1, 1, 1, 3).expand(1, H, W, 3).contiguous()
    roughness = torch.full((1, H, W, 1), roughness_val, dtype=torch.float32, device=device)
    metallic = torch.full((1, H, W, 1), metallic_val, dtype=torch.float32, device=device)

    return {
        'base_color': base_color,
        'roughness': roughness,
        'metallic': metallic,
    }
