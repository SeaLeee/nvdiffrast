"""
GGX PBR BRDF implementation matching Messiah Engine's BRDF.fxh.

Supports:
  - Distribution: GGX (Trowbridge-Reitz)
  - Geometry: Schlick-Disney (Smith approx)
  - Fresnel: Schlick
  - Diffuse: Lambert / Burley
"""

import math
import torch
import torch.nn.functional as F


MIN_ROUGHNESS = 0.08


def safe_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Clamped dot product, keepdim on last axis."""
    return torch.clamp(torch.sum(a * b, dim=-1, keepdim=True), min=0.0, max=1.0)


class GGX_BRDF:
    """GGX-based PBR BRDF matching Messiah Engine parameters."""

    @staticmethod
    def D_GGX(NdotH: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
        """GGX/Trowbridge-Reitz Normal Distribution Function."""
        a2 = roughness * roughness
        a4 = a2 * a2
        d = NdotH * NdotH * (a4 - 1.0) + 1.0
        return a4 / (math.pi * d * d + 1e-7)

    @staticmethod
    def F_Schlick(F0: torch.Tensor, VdotH: torch.Tensor) -> torch.Tensor:
        """Schlick Fresnel approximation."""
        return F0 + (1.0 - F0) * torch.pow(torch.clamp(1.0 - VdotH, 0.0, 1.0), 5.0)

    @staticmethod
    def Vis_Schlick_Disney(NdotV: torch.Tensor, NdotL: torch.Tensor,
                           roughness: torch.Tensor) -> torch.Tensor:
        """Schlick-Disney geometry / visibility term."""
        k = (roughness + 1.0) ** 2 / 8.0
        g_v = NdotV / (NdotV * (1.0 - k) + k + 1e-7)
        g_l = NdotL / (NdotL * (1.0 - k) + k + 1e-7)
        return g_v * g_l

    @staticmethod
    def Diff_Lambert(base_color: torch.Tensor) -> torch.Tensor:
        """Lambertian diffuse."""
        return base_color / math.pi

    @staticmethod
    def Diff_Burley(base_color: torch.Tensor, roughness: torch.Tensor,
                    NdotV: torch.Tensor, NdotL: torch.Tensor,
                    VdotH: torch.Tensor) -> torch.Tensor:
        """Disney/Burley diffuse model."""
        fd90 = 0.5 + 2.0 * VdotH * VdotH * roughness
        light_scatter = 1.0 + (fd90 - 1.0) * torch.pow(1.0 - NdotL, 5.0)
        view_scatter = 1.0 + (fd90 - 1.0) * torch.pow(1.0 - NdotV, 5.0)
        return base_color * light_scatter * view_scatter / math.pi

    @classmethod
    def evaluate(cls, normal: torch.Tensor, view_dir: torch.Tensor,
                 light_dir: torch.Tensor, base_color: torch.Tensor,
                 metallic: torch.Tensor, roughness: torch.Tensor,
                 diffuse_model: str = "lambert") -> torch.Tensor:
        """
        Evaluate full PBR BRDF * NdotL.

        Args:
            normal:     [..., 3] normalized surface normal
            view_dir:   [..., 3] normalized view direction
            light_dir:  [..., 3] normalized light direction
            base_color: [..., 3] or [..., C] albedo
            metallic:   [..., 1] metallic factor
            roughness:  [..., 1] roughness factor

        Returns:
            [..., 3] BRDF * NdotL (ready to multiply by light_color)
        """
        roughness = torch.clamp(roughness, min=MIN_ROUGHNESS)

        H = F.normalize(view_dir + light_dir, dim=-1)
        NdotL = safe_dot(normal, light_dir)
        NdotV = safe_dot(normal, view_dir)
        NdotH = safe_dot(normal, H)
        VdotH = safe_dot(view_dir, H)

        # F0: dielectric = 0.04, metallic blends toward base_color
        F0 = 0.04 * (1.0 - metallic) + base_color * metallic

        D = cls.D_GGX(NdotH, roughness)
        G = cls.Vis_Schlick_Disney(NdotV, NdotL, roughness)
        Fr = cls.F_Schlick(F0, VdotH)

        # Specular
        specular = (D * G * Fr) / (4.0 * NdotV * NdotL + 1e-7)

        # Diffuse
        kD = (1.0 - Fr) * (1.0 - metallic)
        if diffuse_model == "burley":
            diffuse = kD * cls.Diff_Burley(base_color, roughness, NdotV, NdotL, VdotH)
        else:
            diffuse = kD * cls.Diff_Lambert(base_color)

        return (diffuse + specular) * NdotL
