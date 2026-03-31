"""
Texture Optimizer - Optimize textures via differentiable rendering.

Uses nvdiffrast to render from multiple views and compute pixel-space
gradients that flow back to texture pixels.
"""

import torch
import torch.nn.functional as F
from typing import Optional

from pipeline import MessiahDiffPipeline, Camera, create_orbit_cameras
from pipeline.camera import transform_pos
from optimizer.losses import CompositeLoss, compute_psnr


class TextureOptimizer:
    """
    Optimize textures to minimize rendering error vs reference images.

    Typical use cases:
      - Texture resolution reduction (high→low res) while preserving quality
      - Texture baking from high-poly to low-poly
      - Lightmap optimization
    """

    def __init__(self, pipeline: MessiahDiffPipeline,
                 mesh_data: dict, cameras: list,
                 reference_images: torch.Tensor,
                 target_resolution: tuple = (256, 256),
                 config: dict = None):
        """
        Args:
            pipeline:          MessiahDiffPipeline instance
            mesh_data:         dict with keys:
                               'vertices'    [V, 3]
                               'triangles'   [T, 3] int32
                               'vtx_attr'    dict {'normal', 'uv', 'pos_world'}
                               'roughness_tex' [1, H, W, 1]
                               'metallic_tex'  [1, H, W, 1]
                               'normal_tex'    [1, H, W, 3] (optional)
                               'base_color_hires' [1, H, W, 3]
            cameras:           list of Camera objects
            reference_images:  [N_views, H, W, 3] reference renderings
            target_resolution: (H, W) for optimized texture
            config:            optimization config dict (optional)
        """
        self.pipeline = pipeline
        self.mesh = mesh_data
        self.cameras = cameras
        self.refs = reference_images
        self.device = pipeline.device

        cfg = config or {}
        lr = cfg.get('learning_rate', 1e-2)
        max_iter = cfg.get('max_iterations', 5000)

        # Initialize optimizable texture from bilinear downsampled original
        hires = mesh_data['base_color_hires']  # [1, H, W, 3]
        init_tex = F.interpolate(
            hires.permute(0, 3, 1, 2),
            size=target_resolution,
            mode='bilinear',
            align_corners=False,
        ).permute(0, 2, 3, 1).contiguous()

        # Store in logit space for unconstrained optimization
        self.tex_param = torch.nn.Parameter(
            torch.logit(init_tex.clamp(1e-4, 1.0 - 1e-4))
        )

        self.optimizer = torch.optim.Adam([self.tex_param], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_iter
        )

        # Loss
        loss_cfg = cfg.get('loss_weights', {})
        self.criterion = CompositeLoss(
            w_l2=loss_cfg.get('l2', 1.0),
            w_perceptual=loss_cfg.get('perceptual', 0.1),
            w_ssim=loss_cfg.get('ssim', 0.05),
            w_smoothness=loss_cfg.get('smoothness', 0.001),
            device=self.device,
        )

        self.iteration = 0

    def get_texture(self) -> torch.Tensor:
        """Get current optimized texture in [0, 1] range."""
        return torch.sigmoid(self.tex_param)

    def step(self, view_idx: Optional[int] = None) -> dict:
        """
        Execute one optimization step.

        Args:
            view_idx: Camera view index. Random if None.

        Returns:
            dict with 'loss', 'psnr', 'rendered' (detached), 'iteration'
        """
        if view_idx is None:
            view_idx = torch.randint(len(self.cameras), (1,)).item()

        camera = self.cameras[view_idx]
        ref = self.refs[view_idx:view_idx + 1]  # [1, H, W, 3]

        # Build textures dict with current optimized texture
        current_tex = self.get_texture()
        textures = {
            'base_color': current_tex,
            'roughness': self.mesh['roughness_tex'],
            'metallic': self.mesh['metallic_tex'],
        }
        if 'normal_tex' in self.mesh:
            textures['normal'] = self.mesh['normal_tex']

        # Render
        color, mask, _ = self.pipeline.render_from_camera(
            camera, self.mesh['vertices'], self.mesh['triangles'],
            self.mesh['vtx_attr'], textures,
            light_dir=self.mesh.get('light_dir'),
            light_color=self.mesh.get('light_color'),
        )

        # Loss
        loss = self.criterion(
            color, ref, mask,
            optimized_textures=[current_tex],
        )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.iteration += 1

        psnr = compute_psnr(color.detach(), ref, mask.detach())

        return {
            'loss': loss.item(),
            'psnr': psnr,
            'rendered': color.detach(),
            'iteration': self.iteration,
        }

    def export_texture(self) -> torch.Tensor:
        """Get final optimized texture as uint8 numpy array [H, W, 3]."""
        tex = self.get_texture().detach().cpu()
        tex_np = (tex[0].numpy() * 255).clip(0, 255).astype('uint8')
        return tex_np
