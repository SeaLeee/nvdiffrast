"""
Texture I/O - Load and save textures in various formats.
"""

import os
import numpy as np
import torch
from typing import Optional


def load_texture(path: str, device: str = 'cuda',
                 srgb_to_linear: bool = True) -> torch.Tensor:
    """
    Load a texture image as a PyTorch tensor.

    Args:
        path:           File path (.png, .jpg, .tga, .dds, .exr, .hdr)
        device:         Target device
        srgb_to_linear: Convert sRGB to linear color space

    Returns:
        [1, H, W, C] float32 tensor in [0, 1] range
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in ('.exr', '.hdr'):
        arr = _load_hdr(path)
    else:
        from PIL import Image
        img = Image.open(path)
        if img.mode == 'RGBA':
            arr = np.array(img, dtype=np.float32) / 255.0
        elif img.mode == 'L':
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr[:, :, np.newaxis]
        else:
            img = img.convert('RGB')
            arr = np.array(img, dtype=np.float32) / 255.0

    tex = torch.from_numpy(arr).unsqueeze(0).to(device)  # [1, H, W, C]

    if srgb_to_linear and tex.shape[-1] >= 3:
        tex[..., :3] = _srgb_to_linear_torch(tex[..., :3])

    return tex.contiguous()


def save_texture(arr, path: str, srgb: bool = True):
    """
    Save texture array to file.

    Args:
        arr:  numpy [H, W, C] uint8 or float32, or torch tensor
        path: Output file path (.png, .tga, .dds)
        srgb: Apply linear-to-sRGB before saving
    """
    if isinstance(arr, torch.Tensor):
        if arr.dim() == 4:
            arr = arr[0]
        arr = arr.cpu().numpy()

    if arr.dtype == np.float32 or arr.dtype == np.float64:
        if srgb and arr.shape[-1] >= 3:
            arr_srgb = arr.copy()
            arr_srgb[..., :3] = _linear_to_srgb_np(arr_srgb[..., :3])
            arr = arr_srgb
        arr = (arr * 255).clip(0, 255).astype(np.uint8)

    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    if ext in ('.png', '.jpg', '.jpeg', '.bmp'):
        from PIL import Image
        if arr.shape[-1] == 1:
            img = Image.fromarray(arr[:, :, 0], mode='L')
        elif arr.shape[-1] == 4:
            img = Image.fromarray(arr, mode='RGBA')
        else:
            img = Image.fromarray(arr, mode='RGB')
        img.save(path)

    elif ext == '.tga':
        _save_tga(arr, path)

    elif ext == '.exr':
        _save_exr(arr, path)

    else:
        # Default to PNG
        from PIL import Image
        img = Image.fromarray(arr[:, :, :3], mode='RGB')
        img.save(path)


def create_uniform_texture(value, resolution=(256, 256), channels=1,
                           device='cuda') -> torch.Tensor:
    """Create a uniform-color texture [1, H, W, C]."""
    if isinstance(value, (int, float)):
        value = [value] * channels
    H, W = resolution
    tex = torch.zeros(1, H, W, len(value), device=device)
    for i, v in enumerate(value):
        tex[..., i] = v
    return tex


# ============ Internal ============

def _srgb_to_linear_torch(c: torch.Tensor) -> torch.Tensor:
    low = c / 12.92
    high = torch.pow((c + 0.055) / 1.055, 2.4)
    return torch.where(c <= 0.04045, low, high)


def _linear_to_srgb_np(c: np.ndarray) -> np.ndarray:
    low = c * 12.92
    high = 1.055 * np.power(np.maximum(c, 1e-8), 1.0 / 2.4) - 0.055
    return np.where(c <= 0.0031308, low, high)


def _load_hdr(path: str) -> np.ndarray:
    """Load HDR/EXR image."""
    try:
        import imageio
        arr = imageio.imread(path)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr
    except ImportError:
        raise ImportError("imageio required for HDR/EXR loading: pip install imageio")


def _save_tga(arr: np.ndarray, path: str):
    """Save as TGA (uncompressed)."""
    h, w = arr.shape[:2]
    c = arr.shape[2] if arr.ndim == 3 else 1

    with open(path, 'wb') as f:
        # TGA header
        f.write(bytes([0, 0, 2]))  # uncompressed true-color
        f.write(bytes(9))          # color map + origin
        f.write(w.to_bytes(2, 'little'))
        f.write(h.to_bytes(2, 'little'))
        f.write(bytes([c * 8, 0x20]))  # bits per pixel, top-left origin

        # Convert RGB → BGR for TGA
        if c >= 3:
            data = arr.copy()
            data[..., 0], data[..., 2] = arr[..., 2], arr[..., 0]
            f.write(data.tobytes())
        else:
            f.write(arr.tobytes())


def _save_exr(arr: np.ndarray, path: str):
    """Save as EXR."""
    try:
        import imageio
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        imageio.imwrite(path, arr)
    except ImportError:
        raise ImportError("imageio required for EXR saving")
