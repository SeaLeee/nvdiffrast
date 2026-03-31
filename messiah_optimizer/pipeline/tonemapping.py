"""
Tone mapping operators matching Messiah Engine's post-processing.
"""

import torch


def reinhard_tonemap(color: torch.Tensor) -> torch.Tensor:
    """Simple Reinhard tone mapping."""
    return color / (1.0 + color)


def aces_tonemap(color: torch.Tensor) -> torch.Tensor:
    """ACES filmic tone mapping (fitted curve)."""
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    mapped = (color * (a * color + b)) / (color * (c * color + d) + e)
    return torch.clamp(mapped, 0.0, 1.0)


def linear_to_srgb(color: torch.Tensor) -> torch.Tensor:
    """Linear to sRGB gamma correction."""
    low = color * 12.92
    high = 1.055 * torch.pow(torch.clamp(color, min=1e-8), 1.0 / 2.4) - 0.055
    return torch.where(color <= 0.0031308, low, high)


def srgb_to_linear(color: torch.Tensor) -> torch.Tensor:
    """sRGB to linear."""
    low = color / 12.92
    high = torch.pow((color + 0.055) / 1.055, 2.4)
    return torch.where(color <= 0.04045, low, high)
