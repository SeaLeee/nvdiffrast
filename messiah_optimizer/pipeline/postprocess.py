"""
Differentiable post-processing stack matching Messiah Engine effects.

All operations preserve gradient flow so they can be used during optimization.
When comparing with engine screenshots (which include bloom, color grading etc.),
enable matching post-processing here so the optimizer doesn't waste capacity
trying to replicate post-processing artifacts in textures.

Key principle:
  - Self-supervised mode: SAME pipeline for ref & rendered → post-proc cancels out
  - Engine reference mode: NEED matching post-proc to reduce domain gap
"""

import torch
import torch.nn.functional as F
from typing import Optional


class DiffBloom(torch.nn.Module):
    """
    Differentiable bloom approximation.
    
    Extracts bright pixels above threshold, applies multi-scale gaussian blur,
    then blends back. Fully differentiable for gradient flow.
    """

    def __init__(self, threshold: float = 0.8, intensity: float = 0.3,
                 blur_sizes: tuple = (5, 9, 17)):
        super().__init__()
        self.threshold = threshold
        self.intensity = intensity
        self.blur_sizes = blur_sizes

    def forward(self, color: torch.Tensor) -> torch.Tensor:
        """
        Args:
            color: [B, H, W, 3] in [0, 1] range (after tonemapping)
        Returns:
            [B, H, W, 3] with bloom applied
        """
        # Extract bright regions (soft threshold for differentiability)
        brightness = color.mean(dim=-1, keepdim=True)
        # Soft threshold: sigmoid ramp instead of hard cutoff
        bloom_mask = torch.sigmoid((brightness - self.threshold) * 10.0)
        bright = color * bloom_mask

        # Convert to NCHW for conv2d
        bright_nchw = bright.permute(0, 3, 1, 2)

        # Multi-scale gaussian blur and accumulate
        bloom_accum = torch.zeros_like(bright_nchw)
        for ks in self.blur_sizes:
            sigma = ks / 3.0
            bloom_accum = bloom_accum + self._gaussian_blur(bright_nchw, ks, sigma)

        bloom_accum = bloom_accum / len(self.blur_sizes)

        # Blend back (NCHW → NHWC)
        bloom_nhwc = bloom_accum.permute(0, 2, 3, 1)
        result = color + bloom_nhwc * self.intensity
        return torch.clamp(result, 0.0, 1.0)

    @staticmethod
    def _gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """Apply separable gaussian blur (differentiable)."""
        pad = kernel_size // 2
        # 1D kernel
        coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - pad
        kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Horizontal pass
        kh = kernel_1d.view(1, 1, 1, -1).expand(x.shape[1], -1, -1, -1)
        x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode='reflect'),
                      kh, groups=x.shape[1])
        # Vertical pass
        kv = kernel_1d.view(1, 1, -1, 1).expand(x.shape[1], -1, -1, -1)
        x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode='reflect'),
                      kv, groups=x.shape[1])
        return x


class DiffColorGrading(torch.nn.Module):
    """
    Differentiable color grading approximation.
    
    Adjustable: exposure, contrast, saturation, gamma,
    color temperature (warm/cool shift).
    """

    def __init__(self, exposure: float = 0.0, contrast: float = 1.0,
                 saturation: float = 1.0, gamma: float = 1.0,
                 temperature: float = 0.0):
        super().__init__()
        # Use buffers (not parameters) — these are matching targets, not optimized
        self.register_buffer('exposure', torch.tensor(exposure))
        self.register_buffer('contrast', torch.tensor(contrast))
        self.register_buffer('saturation', torch.tensor(saturation))
        self.register_buffer('gamma', torch.tensor(gamma))
        self.register_buffer('temperature', torch.tensor(temperature))

    def forward(self, color: torch.Tensor) -> torch.Tensor:
        """
        Args:
            color: [B, H, W, 3] in [0, 1]
        Returns:
            [B, H, W, 3] color-graded
        """
        c = color

        # Exposure (multiply in linear space)
        if self.exposure != 0.0:
            c = c * (2.0 ** self.exposure)

        # Contrast (around mid-gray 0.5)
        if self.contrast != 1.0:
            c = (c - 0.5) * self.contrast + 0.5

        # Saturation (lerp with luminance)
        if self.saturation != 1.0:
            luma = 0.2126 * c[..., 0:1] + 0.7152 * c[..., 1:2] + 0.0722 * c[..., 2:3]
            c = luma + (c - luma) * self.saturation

        # Temperature shift (warm = +R -B, cool = -R +B)
        if self.temperature != 0.0:
            t = self.temperature * 0.1
            shift = torch.tensor([t, 0.0, -t], device=c.device).view(1, 1, 1, 3)
            c = c + shift

        # Gamma
        if self.gamma != 1.0:
            c = torch.pow(torch.clamp(c, min=1e-6), 1.0 / self.gamma)

        return torch.clamp(c, 0.0, 1.0)


class DiffVignette(torch.nn.Module):
    """Simple vignette darkening toward edges."""

    def __init__(self, strength: float = 0.3):
        super().__init__()
        self.strength = strength

    def forward(self, color: torch.Tensor) -> torch.Tensor:
        B, H, W, C = color.shape
        y = torch.linspace(-1, 1, H, device=color.device)
        x = torch.linspace(-1, 1, W, device=color.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(xx ** 2 + yy ** 2)
        vignette = 1.0 - self.strength * torch.clamp(dist - 0.5, min=0.0)
        vignette = vignette.view(1, H, W, 1)
        return color * vignette


class PostProcessStack(torch.nn.Module):
    """
    Configurable post-processing stack.
    
    Usage modes:
      1. DISABLED (self-supervised): no post-proc, both ref & rendered are raw
      2. MATCH_ENGINE: apply bloom + color grading to match engine output
      3. CUSTOM: user-configured effects
    
    All effects are differentiable → gradients flow through normally.
    """

    def __init__(self, mode: str = 'disabled', config: dict = None):
        """
        Args:
            mode: 'disabled' | 'match_engine' | 'custom'
            config: dict with effect parameters
        """
        super().__init__()
        self.mode = mode
        cfg = config or {}

        self.bloom = None
        self.color_grading = None
        self.vignette = None

        if mode == 'disabled':
            return

        if mode in ('match_engine', 'custom'):
            bloom_cfg = cfg.get('bloom', {})
            if bloom_cfg.get('enabled', mode == 'match_engine'):
                self.bloom = DiffBloom(
                    threshold=bloom_cfg.get('threshold', 0.8),
                    intensity=bloom_cfg.get('intensity', 0.3),
                    blur_sizes=tuple(bloom_cfg.get('blur_sizes', [5, 9, 17])),
                )

            cg_cfg = cfg.get('color_grading', {})
            if cg_cfg.get('enabled', mode == 'match_engine'):
                self.color_grading = DiffColorGrading(
                    exposure=cg_cfg.get('exposure', 0.0),
                    contrast=cg_cfg.get('contrast', 1.05),
                    saturation=cg_cfg.get('saturation', 1.1),
                    gamma=cg_cfg.get('gamma', 1.0),
                    temperature=cg_cfg.get('temperature', 0.0),
                )

            vig_cfg = cfg.get('vignette', {})
            if vig_cfg.get('enabled', False):
                self.vignette = DiffVignette(
                    strength=vig_cfg.get('strength', 0.3),
                )

    def forward(self, color: torch.Tensor) -> torch.Tensor:
        """
        Apply post-processing stack.
        
        Args:
            color: [B, H, W, 3] tonemapped sRGB image in [0,1]
        Returns:
            [B, H, W, 3] post-processed image
        """
        if self.mode == 'disabled':
            return color

        if self.bloom is not None:
            color = self.bloom(color)
        if self.color_grading is not None:
            color = self.color_grading(color)
        if self.vignette is not None:
            color = self.vignette(color)

        return color

    def describe(self) -> str:
        """Return human-readable description of active effects."""
        if self.mode == 'disabled':
            return "后处理: 关闭 (自监督模式，无域差)"
        parts = []
        if self.bloom is not None:
            parts.append(f"Bloom(阈值={self.bloom.threshold:.1f}, "
                         f"强度={self.bloom.intensity:.1f})")
        if self.color_grading is not None:
            cg = self.color_grading
            parts.append(f"ColorGrade(曝光={cg.exposure:.1f}, "
                         f"对比度={cg.contrast:.2f}, "
                         f"饱和度={cg.saturation:.2f})")
        if self.vignette is not None:
            parts.append(f"Vignette({self.vignette.strength:.1f})")
        return "后处理: " + " → ".join(parts) if parts else "后处理: 无效果"
