"""
Shading model implementations corresponding to Messiah Engine's ShadingModel.fxh.
Currently implements DefaultLit; other models are stubs for future expansion.
"""

import math
import torch
import torch.nn.functional as F

from .brdf import GGX_BRDF, safe_dot, MIN_ROUGHNESS


class ShadingModel:
    """Base class for shading models."""
    model_id: int = 0
    name: str = "Base"

    def shade(self, shading_data: dict, light_dir: torch.Tensor,
              light_color: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DefaultLitModel(ShadingModel):
    """
    DefaultLit (ID=1) - Standard PBR shading.
    Matches Messiah Deferred.fx DefaultLit branch.
    """
    model_id = 1
    name = "DefaultLit"

    def shade(self, sd: dict, light_dir: torch.Tensor,
              light_color: torch.Tensor) -> torch.Tensor:
        brdf_val = GGX_BRDF.evaluate(
            normal=sd['normal'],
            view_dir=sd['view_dir'],
            light_dir=light_dir,
            base_color=sd['base_color'],
            metallic=sd['metallic'],
            roughness=sd['roughness'],
        )
        color = brdf_val * light_color
        if 'ao' in sd and sd['ao'] is not None:
            color = color * sd['ao']
        return color


class UnlitModel(ShadingModel):
    """Unlit (ID=0) - Emission only, no lighting."""
    model_id = 0
    name = "Unlit"

    def shade(self, sd: dict, light_dir: torch.Tensor,
              light_color: torch.Tensor) -> torch.Tensor:
        return sd.get('emission', sd['base_color'])


class SubsurfaceScatteringModel(ShadingModel):
    """
    SSS (ID=3) - Subsurface scattering approximation.
    Simplified differentiable version of Messiah's SSS.
    """
    model_id = 3
    name = "SSS"

    def __init__(self, scatter_width: float = 0.3):
        self.scatter_width = scatter_width

    def shade(self, sd: dict, light_dir: torch.Tensor,
              light_color: torch.Tensor) -> torch.Tensor:
        # Standard specular/diffuse
        brdf_val = GGX_BRDF.evaluate(
            normal=sd['normal'], view_dir=sd['view_dir'],
            light_dir=light_dir, base_color=sd['base_color'],
            metallic=sd['metallic'], roughness=sd['roughness'],
        )

        # Wrap-around diffuse for subsurface approximation
        NdotL_wrap = torch.sum(sd['normal'] * light_dir, dim=-1, keepdim=True)
        scatter = torch.clamp(
            (NdotL_wrap + self.scatter_width) / (1.0 + self.scatter_width), 0.0, 1.0
        )
        sss_color = sd.get('subsurface_color', sd['base_color'])
        subsurface = sss_color * scatter / math.pi

        color = (brdf_val + subsurface * (1.0 - sd['metallic'])) * light_color
        return color


class ClothModel(ShadingModel):
    """
    Cloth (ID=10) - Cloth/fabric shading.
    Uses modified GGX with sheen term.
    """
    model_id = 10
    name = "Cloth"

    def shade(self, sd: dict, light_dir: torch.Tensor,
              light_color: torch.Tensor) -> torch.Tensor:
        normal = sd['normal']
        view_dir = sd['view_dir']
        roughness = torch.clamp(sd['roughness'], min=MIN_ROUGHNESS)

        H = F.normalize(view_dir + light_dir, dim=-1)
        NdotL = safe_dot(normal, light_dir)
        NdotV = safe_dot(normal, view_dir)
        NdotH = safe_dot(normal, H)
        VdotH = safe_dot(view_dir, H)

        # Cloth diffuse (wrapped Lambert)
        diffuse = sd['base_color'] / math.pi

        # Cloth specular - modified D term with Ashikhmin distribution
        sin_theta = torch.sqrt(torch.clamp(1.0 - NdotH * NdotH, 0.0, 1.0))
        a2 = roughness * roughness
        D_cloth = (1.0 / (math.pi * (1.0 + a2 * 4.0))) * (1.0 + a2 * 4.0 * torch.exp(-sin_theta * sin_theta / a2))

        # Sheen color
        sheen_color = sd.get('sheen_color', torch.ones_like(sd['base_color']) * 0.04)
        specular = D_cloth * sheen_color

        color = (diffuse + specular) * NdotL * light_color
        return color


# Registry
SHADING_MODELS = {
    0: UnlitModel,
    1: DefaultLitModel,
    3: SubsurfaceScatteringModel,
    10: ClothModel,
}


def get_shading_model(model_id: int) -> ShadingModel:
    """Get shading model instance by ID."""
    cls = SHADING_MODELS.get(model_id, DefaultLitModel)
    return cls() if isinstance(cls, type) else cls
