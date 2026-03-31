"""
Lightweight CPU software rasterizer for preview rendering.
Falls back to this when nvdiffrast/CUDA are not available.
Uses numpy for basic triangle rasterization with z-buffer and Lambert shading.
"""

import numpy as np
import torch


def _look_at(eye, target, up=None):
    """Build 4x4 view matrix."""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up or [0, 1, 0], dtype=np.float32)

    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up2 = np.cross(right, forward)

    mat = np.eye(4, dtype=np.float32)
    mat[0, :3] = right
    mat[1, :3] = up2
    mat[2, :3] = -forward
    mat[0, 3] = -np.dot(right, eye)
    mat[1, 3] = -np.dot(up2, eye)
    mat[2, 3] = np.dot(forward, eye)
    return mat


def _perspective(fov_deg, aspect, near, far):
    """Build 4x4 perspective projection matrix."""
    import math
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2 * far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat


def render_preview(vertices, triangles, normals, uvs=None, textures=None,
                   camera_params=None, resolution=(512, 512),
                   light_dir=None, light_color=None, ambient=0.15):
    """
    CPU software rasterizer for preview.

    Args:
        vertices:  [V, 3] numpy or tensor
        triangles: [T, 3] numpy or tensor (int)
        normals:   [V, 3] numpy or tensor
        uvs:       [V, 2] numpy or tensor (optional)
        textures:  dict with 'base_color' [1,H,W,3] tensor (optional)
        camera_params: dict with position, target, fov
        resolution: (H, W) output size
        light_dir: [3] light direction
        light_color: [3] light RGB
        ambient: ambient light intensity

    Returns:
        image: [H, W, 3] numpy uint8 RGB
    """
    # Convert to numpy
    def _to_np(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    verts = _to_np(vertices).astype(np.float32)
    tris = _to_np(triangles).astype(np.int32)
    norms = _to_np(normals).astype(np.float32)

    H, W = resolution

    # Camera
    cam = camera_params or {}
    eye = np.array(cam.get('position', [3, 2, 3]), dtype=np.float32)
    target = np.array(cam.get('target', [0, 0, 0]), dtype=np.float32)
    fov = cam.get('fov', 60.0)

    view = _look_at(eye, target)
    proj = _perspective(fov, W / H, 0.1, 100.0)
    mvp = proj @ view

    # Project vertices to clip space
    ones = np.ones((verts.shape[0], 1), dtype=np.float32)
    verts_h = np.hstack([verts, ones])  # [V, 4]
    clip = (mvp @ verts_h.T).T  # [V, 4]

    # Perspective divide -> NDC
    w_clip = clip[:, 3:4]
    w_clip = np.where(np.abs(w_clip) < 1e-6, 1e-6, w_clip)
    ndc = clip[:, :3] / w_clip  # [V, 3]

    # NDC to screen
    screen_x = ((ndc[:, 0] + 1.0) * 0.5 * W).astype(np.float32)
    screen_y = ((1.0 - ndc[:, 1]) * 0.5 * H).astype(np.float32)  # flip Y
    screen_z = ndc[:, 2]  # depth

    # Light direction
    if light_dir is None:
        light_dir = np.array([-0.5, -1.0, -0.5], dtype=np.float32)
    else:
        light_dir = np.asarray(light_dir, dtype=np.float32)
    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-8)

    if light_color is None:
        light_color = np.array([1.0, 0.98, 0.95], dtype=np.float32)
    else:
        light_color = np.asarray(light_color, dtype=np.float32)

    # Base color from texture or default
    base_color_per_vert = np.full((verts.shape[0], 3), 0.7, dtype=np.float32)
    if textures is not None and 'base_color' in textures:
        bc_tex = _to_np(textures['base_color'])  # [1, TH, TW, 3]
        if bc_tex.ndim == 4:
            bc_tex = bc_tex[0]
        if uvs is not None:
            uv = _to_np(uvs).astype(np.float32)
            th, tw = bc_tex.shape[:2]
            u_px = np.clip((uv[:, 0] * tw).astype(int), 0, tw - 1)
            v_px = np.clip(((1.0 - uv[:, 1]) * th).astype(int), 0, th - 1)
            base_color_per_vert = bc_tex[v_px, u_px, :3]
        else:
            # Use center pixel as uniform color
            base_color_per_vert[:] = bc_tex[bc_tex.shape[0]//2, bc_tex.shape[1]//2, :3]

    # Initialize buffers
    color_buf = np.full((H, W, 3), 0.05, dtype=np.float32)  # dark background
    z_buf = np.full((H, W), 1e10, dtype=np.float32)

    # Rasterize triangles
    for ti in range(tris.shape[0]):
        i0, i1, i2 = tris[ti]

        # Screen-space triangle vertices
        sx = np.array([screen_x[i0], screen_x[i1], screen_x[i2]])
        sy = np.array([screen_y[i0], screen_y[i1], screen_y[i2]])
        sz = np.array([screen_z[i0], screen_z[i1], screen_z[i2]])

        # Skip triangles behind camera
        if np.any(sz < -1.0) or np.any(sz > 1.0):
            continue

        # Bounding box
        min_x = max(int(np.floor(sx.min())), 0)
        max_x = min(int(np.ceil(sx.max())), W - 1)
        min_y = max(int(np.floor(sy.min())), 0)
        max_y = min(int(np.ceil(sy.max())), H - 1)

        if min_x > max_x or min_y > max_y:
            continue

        # Triangle edge vectors for barycentric
        v0x, v0y = sx[1] - sx[0], sy[1] - sy[0]
        v1x, v1y = sx[2] - sx[0], sy[2] - sy[0]
        denom = v0x * v1y - v1x * v0y
        if abs(denom) < 1e-8:
            continue
        inv_denom = 1.0 / denom

        # Per-vertex data
        n0, n1, n2 = norms[i0], norms[i1], norms[i2]
        c0, c1, c2 = base_color_per_vert[i0], base_color_per_vert[i1], base_color_per_vert[i2]

        # Vectorized rasterization over bbox
        py, px = np.mgrid[min_y:max_y+1, min_x:max_x+1]
        px_f = px.astype(np.float32) + 0.5
        py_f = py.astype(np.float32) + 0.5

        v2x = px_f - sx[0]
        v2y = py_f - sy[0]

        u = (v2x * v1y - v1x * v2y) * inv_denom
        v = (v0x * v2y - v2x * v0y) * inv_denom
        w = 1.0 - u - v

        # Inside triangle mask
        mask = (u >= 0) & (v >= 0) & (w >= 0)

        if not np.any(mask):
            continue

        # Interpolate depth
        z_interp = w * sz[0] + u * sz[1] + v * sz[2]

        # Z-buffer test
        rows = py[mask]
        cols = px[mask]
        z_vals = z_interp[mask]
        u_m, v_m, w_m = u[mask], v[mask], w[mask]

        closer = z_vals < z_buf[rows, cols]
        if not np.any(closer):
            continue

        rows = rows[closer]
        cols = cols[closer]
        z_vals = z_vals[closer]
        u_m = u_m[closer]
        v_m = v_m[closer]
        w_m = w_m[closer]

        z_buf[rows, cols] = z_vals

        # Interpolate normal
        nx_i = w_m[:, None] * n0 + u_m[:, None] * n1 + v_m[:, None] * n2
        n_len = np.linalg.norm(nx_i, axis=1, keepdims=True)
        nx_i = nx_i / (n_len + 1e-8)

        # Interpolate base color
        bc_i = w_m[:, None] * c0 + u_m[:, None] * c1 + v_m[:, None] * c2

        # Lambert shading
        NdotL = np.sum(nx_i * (-light_dir), axis=1, keepdims=True)
        NdotL = np.clip(NdotL, 0, 1)

        # Simple specular (Blinn-Phong)
        view_dir = eye - (w_m[:, None] * verts[i0] + u_m[:, None] * verts[i1] + v_m[:, None] * verts[i2])
        view_dir = view_dir / (np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-8)
        half_vec = (-light_dir) + view_dir
        half_vec = half_vec / (np.linalg.norm(half_vec, axis=1, keepdims=True) + 1e-8)
        NdotH = np.sum(nx_i * half_vec, axis=1, keepdims=True)
        NdotH = np.clip(NdotH, 0, 1)
        spec = np.power(NdotH, 32.0) * 0.3

        shaded = bc_i * (ambient + NdotL * light_color) + spec * light_color
        color_buf[rows, cols] = np.clip(shaded, 0, 1)

    # Apply simple tone mapping (gamma)
    color_buf = np.power(np.clip(color_buf, 0, 1), 1.0 / 2.2)

    return (color_buf * 255).astype(np.uint8)
