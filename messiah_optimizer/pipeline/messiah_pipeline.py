"""
Differentiable rendering pipeline simulating Messiah Engine's HybridPipeline.

Uses nvdiffrast for rasterize → interpolate → texture → shade → antialias,
with BRDF matching Messiah's BRDF.fxh (GGX + Schlick_Disney + Lambert).
"""

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from .brdf import GGX_BRDF
from .camera import Camera, transform_pos
from .shading_models import get_shading_model
from .tonemapping import aces_tonemap, linear_to_srgb
from .postprocess import PostProcessStack


class MessiahDiffPipeline:
    """
    Differentiable rendering pipeline simulating Messiah HybridPipeline.

    Supports full gradient flow from rendered pixels back to:
      - Texture pixels (base_color, normal, roughness, metallic)
      - Material scalar parameters
      - Vertex positions

    Post-processing modes:
      - 'disabled': No post-proc (self-supervised mode, ref & rendered same pipeline)
      - 'match_engine': Apply bloom + color grading to approximate engine output
      - 'custom': User-configured effects
    """

    def __init__(self, resolution=(1024, 1024), device='cuda',
                 postprocess_mode='disabled', postprocess_config=None):
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.resolution = list(resolution)
        self.device = device
        self.postprocess = PostProcessStack(
            mode=postprocess_mode,
            config=postprocess_config,
        ).to(device)

    def render(self, pos_clip, tri, vtx_attr, textures, material_params=None,
               camera_pos=None, light_dir=None, light_color=None,
               shading_model_id=1, apply_tonemap=False):
        """
        Full differentiable forward rendering pass.

        Args:
            pos_clip:        [B, V, 4] clip-space vertex positions
            tri:             [T, 3] int32 triangle indices
            vtx_attr:        dict with keys:
                             - 'normal':    [V, 3] vertex normals
                             - 'uv':        [V, 2] texture coordinates
                             - 'tangent':   [V, 4] tangent vectors (optional)
                             - 'pos_world': [V, 3] world-space positions
            textures:        dict with keys:
                             - 'base_color': [1, H, W, 3] base color texture
                             - 'normal':     [1, H, W, 3] normal map (optional)
                             - 'roughness':  [1, H, W, 1] roughness texture
                             - 'metallic':   [1, H, W, 1] metallic texture
                             - 'ao':         [1, H, W, 1] ambient occlusion (optional)
            material_params: dict of optional scalar overrides
            camera_pos:      [B, 3] camera world positions
            light_dir:       [3] normalized directional light direction
            light_color:     [3] light color * intensity
            shading_model_id: int, Messiah shading model ID
            apply_tonemap:   bool, apply ACES tonemapping

        Returns:
            color: [B, H, W, 3] rendered image
            mask:  [B, H, W, 1] coverage mask
            rast_out: rasterization output (for antialias)
        """
        B = pos_clip.shape[0]

        # Default lighting - direction FROM surface TO light
        if light_dir is None:
            light_dir = torch.tensor([0.5, 1.0, 1.0], device=self.device)
            light_dir = F.normalize(light_dir, dim=0)
        if light_color is None:
            light_color = torch.ones(3, device=self.device) * 3.0  # Brighter for PBR + ACES tonemap

        # ====== 1. Rasterize ======
        rast_out, rast_db = dr.rasterize(
            self.glctx, pos_clip, tri, resolution=self.resolution
        )

        # Coverage mask
        mask = (rast_out[..., 3:4] > 0).float()

        # ====== 2. Interpolate vertex attributes ======
        # Normals
        normal_interp, _ = dr.interpolate(
            vtx_attr['normal'][None, ...], rast_out, tri
        )
        normal_interp = F.normalize(normal_interp, dim=-1)

        # UV coordinates (with screen-space derivatives for mipmapping)
        uv_interp, uv_da = dr.interpolate(
            vtx_attr['uv'][None, ...], rast_out, tri,
            rast_db=rast_db, diff_attrs='all'
        )

        # World positions
        pos_world = None
        if 'pos_world' in vtx_attr:
            pos_world, _ = dr.interpolate(
                vtx_attr['pos_world'][None, ...], rast_out, tri
            )

        # Tangent vectors (for normal mapping)
        tangent_interp = None
        if 'tangent' in vtx_attr:
            tangent_interp, _ = dr.interpolate(
                vtx_attr['tangent'][None, ...], rast_out, tri
            )

        # ====== 3. Texture sampling ======
        base_color = dr.texture(
            textures['base_color'], uv_interp, uv_da,
            filter_mode='linear-mipmap-linear'
        )

        roughness_tex = dr.texture(
            textures['roughness'], uv_interp, uv_da,
            filter_mode='linear-mipmap-linear'
        )

        metallic_tex = dr.texture(
            textures['metallic'], uv_interp, uv_da,
            filter_mode='linear-mipmap-linear'
        )

        ao_tex = None
        if 'ao' in textures:
            ao_tex = dr.texture(
                textures['ao'], uv_interp, uv_da,
                filter_mode='linear-mipmap-linear'
            )

        # Normal mapping (tangent space → world space)
        if 'normal' in textures and tangent_interp is not None:
            normal_map = dr.texture(
                textures['normal'], uv_interp, uv_da,
                filter_mode='linear-mipmap-linear'
            )
            normal_map = normal_map[..., :3] * 2.0 - 1.0
            normal_interp = self._apply_normal_map(
                normal_interp, tangent_interp, normal_map
            )
        elif 'normal' in textures:
            # Simplified: perturb interpolated normals
            normal_map = dr.texture(
                textures['normal'], uv_interp, uv_da,
                filter_mode='linear-mipmap-linear'
            )
            normal_map = normal_map[..., :3] * 2.0 - 1.0
            normal_interp = F.normalize(
                normal_interp + normal_map * 0.5, dim=-1
            )

        # Apply scalar overrides
        if material_params:
            if 'roughness_scale' in material_params:
                roughness_tex = roughness_tex * material_params['roughness_scale']
            if 'metallic_scale' in material_params:
                metallic_tex = metallic_tex * material_params['metallic_scale']

        # ====== 4. Shading ======
        if camera_pos is not None and pos_world is not None:
            view_dir = F.normalize(
                camera_pos[:, None, None, :].expand_as(pos_world) - pos_world,
                dim=-1
            )
        else:
            view_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            view_dir = view_dir.expand_as(normal_interp)

        L = F.normalize(light_dir, dim=-1)
        L = L[None, None, None, :].expand_as(normal_interp)

        shading_data = {
            'normal': normal_interp,
            'view_dir': view_dir,
            'base_color': base_color[..., :3],
            'metallic': metallic_tex[..., :1],
            'roughness': roughness_tex[..., :1],
            'ao': ao_tex[..., :1] if ao_tex is not None else None,
        }

        model = get_shading_model(shading_model_id)
        color = model.shade(shading_data, L, light_color)

        # ====== 5. Antialias ======
        color = dr.antialias(color, rast_out, pos_clip, tri)

        # Apply mask
        color = color * mask

        # Optional tone mapping
        if apply_tonemap:
            color = aces_tonemap(color)
            color = linear_to_srgb(color)

        # Differentiable post-processing (bloom, color grading, etc.)
        # Only active in 'match_engine' or 'custom' mode
        if apply_tonemap:
            color = self.postprocess(color)

        return color, mask, rast_out

    def render_from_camera(self, camera: Camera, vertices, tri, vtx_attr,
                           textures, light_dir=None, light_color=None,
                           model_matrix=None, **kwargs):
        """
        Convenience method: render from a Camera object.

        Args:
            camera:       Camera instance
            vertices:     [V, 3] world-space vertex positions
            tri:          [T, 3] triangle indices
            vtx_attr:     dict with 'normal', 'uv', etc.
            textures:     dict with texture tensors
            light_dir:    [3] light direction
            light_color:  [3] light color
            model_matrix: [4, 4] optional model transform
        """
        mvp = camera.mvp_matrix(model_matrix)
        pos_clip = transform_pos(mvp, vertices).unsqueeze(0)

        # Store world positions for view direction
        if 'pos_world' not in vtx_attr:
            vtx_attr = {**vtx_attr, 'pos_world': vertices}

        camera_pos = camera.position.unsqueeze(0)  # [1, 3]

        return self.render(
            pos_clip, tri, vtx_attr, textures,
            camera_pos=camera_pos,
            light_dir=light_dir, light_color=light_color,
            **kwargs
        )

    @staticmethod
    def _apply_normal_map(normal: torch.Tensor, tangent: torch.Tensor,
                          normal_map: torch.Tensor) -> torch.Tensor:
        """
        Transform tangent-space normal map to world space using TBN matrix.

        Args:
            normal:     [B, H, W, 3] interpolated surface normal
            tangent:    [B, H, W, 4] tangent (xyz) + bitangent sign (w)
            normal_map: [B, H, W, 3] tangent-space normal from texture
        """
        T = F.normalize(tangent[..., :3], dim=-1)
        N = normal
        sign = tangent[..., 3:4]
        B = torch.cross(N, T, dim=-1) * sign

        # TBN matrix multiply
        world_normal = (
            T * normal_map[..., 0:1] +
            B * normal_map[..., 1:2] +
            N * normal_map[..., 2:3]
        )
        return F.normalize(world_normal, dim=-1)
