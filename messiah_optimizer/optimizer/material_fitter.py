"""
Material Fitter - Fit PBR material parameters from reference images.

Given one or more reference photos/renders of an object, optimizes
PBR textures (BaseColor, Roughness, Metallic) to match appearance.
"""

import torch
import torch.nn.functional as F

from pipeline import MessiahDiffPipeline, Camera
from optimizer.losses import CompositeLoss, compute_psnr


class MaterialFitter:
    """
    Optimize full PBR texture maps to match reference images.

    Unlike TextureOptimizer (which optimizes base_color only), this
    jointly optimizes base_color + roughness + metallic textures.
    """

    def __init__(self, pipeline: MessiahDiffPipeline,
                 mesh_data: dict, cameras: list,
                 reference_images: torch.Tensor,
                 tex_resolution: tuple = (512, 512),
                 config: dict = None):
        self.pipeline = pipeline
        self.mesh = mesh_data
        self.cameras = cameras
        self.refs = reference_images
        self.device = pipeline.device

        cfg = config or {}
        H, W = tex_resolution

        # Optimizable textures (logit space for [0,1] constraint)
        self.base_color_param = torch.nn.Parameter(
            torch.zeros(1, H, W, 3, device=self.device)
        )
        self.roughness_param = torch.nn.Parameter(
            torch.zeros(1, H, W, 1, device=self.device)  # sigmoid(0)=0.5
        )
        self.metallic_param = torch.nn.Parameter(
            torch.full((1, H, W, 1), -2.0, device=self.device)  # ~0.12
        )

        # Initialize from existing textures if available
        if 'base_color_hires' in mesh_data:
            init_bc = F.interpolate(
                mesh_data['base_color_hires'].permute(0, 3, 1, 2),
                size=(H, W), mode='bilinear', align_corners=False,
            ).permute(0, 2, 3, 1)
            self.base_color_param.data.copy_(
                torch.logit(init_bc.clamp(1e-4, 1 - 1e-4))
            )

        lr = cfg.get('learning_rate', 0.01)
        self.optimizer = torch.optim.Adam([
            {'params': [self.base_color_param], 'lr': lr},
            {'params': [self.roughness_param], 'lr': lr * 0.5},
            {'params': [self.metallic_param], 'lr': lr * 0.3},
        ])

        max_iter = cfg.get('max_iterations', 5000)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_iter
        )

        loss_cfg = cfg.get('loss_weights', {})
        self.criterion = CompositeLoss(
            w_l2=loss_cfg.get('l2', 1.0),
            w_perceptual=loss_cfg.get('perceptual', 0.1),
            w_ssim=loss_cfg.get('ssim', 0.05),
            w_smoothness=loss_cfg.get('smoothness', 0.001),
            device=self.device,
        )

        self.iteration = 0

    def get_textures(self) -> dict:
        """Get current optimized textures in [0,1]."""
        return {
            'base_color': torch.sigmoid(self.base_color_param),
            'roughness': torch.sigmoid(self.roughness_param),
            'metallic': torch.sigmoid(self.metallic_param),
        }

    def step(self, view_idx=None) -> dict:
        if view_idx is None:
            view_idx = torch.randint(len(self.cameras), (1,)).item()

        camera = self.cameras[view_idx]
        ref = self.refs[view_idx:view_idx + 1]

        textures = self.get_textures()

        # Add normal map if available
        if 'normal_tex' in self.mesh:
            textures['normal'] = self.mesh['normal_tex']

        color, mask, _ = self.pipeline.render_from_camera(
            camera, self.mesh['vertices'], self.mesh['triangles'],
            self.mesh['vtx_attr'], textures,
            light_dir=self.mesh.get('light_dir'),
            light_color=self.mesh.get('light_color'),
        )

        opt_textures = [
            textures['base_color'],
            textures['roughness'],
            textures['metallic'],
        ]
        loss = self.criterion(color, ref, mask, optimized_textures=opt_textures)

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

    def export_textures(self) -> dict:
        """Export optimized textures as uint8 numpy arrays."""
        import numpy as np
        texs = self.get_textures()
        result = {}
        for name, tex in texs.items():
            arr = (tex[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            result[name] = arr
        return result
