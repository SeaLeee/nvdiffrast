"""
Normal Map Baker - Optimize normal maps via differentiable rendering.

Uses nvdiffrast to bake high-poly geometric detail into a normal map
for a low-poly mesh, optimized for visual consistency across views.
"""

import torch
import torch.nn.functional as F

from pipeline import MessiahDiffPipeline, Camera
from optimizer.losses import CompositeLoss, compute_psnr


class NormalMapBaker:
    """
    Bake optimized normal maps from high-poly reference.

    The normal map is treated as a learnable texture in tangent space,
    optimized so that the low-poly+normal-map rendering matches
    the high-poly rendering across multiple views.
    """

    def __init__(self, pipeline: MessiahDiffPipeline,
                 lowpoly_mesh: dict, cameras: list,
                 reference_images: torch.Tensor,
                 normal_resolution: tuple = (512, 512),
                 config: dict = None):
        self.pipeline = pipeline
        self.mesh = lowpoly_mesh
        self.cameras = cameras
        self.refs = reference_images
        self.device = pipeline.device

        cfg = config or {}
        H, W = normal_resolution

        # Optimizable normal map in tangent space
        # Initialize to flat normal (0, 0, 1) → encoded as (0.5, 0.5, 1.0)
        init_normal = torch.zeros(1, H, W, 3, device=self.device)
        init_normal[..., 2] = 3.0  # after sigmoid → ~0.95 → tangent z points up
        self.normal_param = torch.nn.Parameter(init_normal)

        lr = cfg.get('learning_rate', 0.01)
        self.optimizer = torch.optim.Adam([self.normal_param], lr=lr)

        max_iter = cfg.get('max_iterations', 3000)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_iter
        )

        self.criterion = CompositeLoss(
            w_l2=1.0, w_perceptual=0.1, w_ssim=0.05,
            w_smoothness=0.005,  # Higher smoothness for normal maps
            device=self.device,
        )

        self.iteration = 0

    def get_normal_map(self) -> torch.Tensor:
        """Get current normal map in [0, 1] range (tangent space encoded)."""
        return torch.sigmoid(self.normal_param)

    def step(self, view_idx=None) -> dict:
        if view_idx is None:
            view_idx = torch.randint(len(self.cameras), (1,)).item()

        camera = self.cameras[view_idx]
        ref = self.refs[view_idx:view_idx + 1]

        normal_tex = self.get_normal_map()

        textures = {
            'base_color': self.mesh['base_color_tex'],
            'roughness': self.mesh['roughness_tex'],
            'metallic': self.mesh['metallic_tex'],
            'normal': normal_tex,
        }

        color, mask, _ = self.pipeline.render_from_camera(
            camera, self.mesh['vertices'], self.mesh['triangles'],
            self.mesh['vtx_attr'], textures,
            light_dir=self.mesh.get('light_dir'),
            light_color=self.mesh.get('light_color'),
        )

        loss = self.criterion(
            color, ref, mask,
            optimized_textures=[normal_tex],
        )

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

    def export_normal_map(self):
        """Export optimized normal map as uint8 numpy array [H, W, 3]."""
        import numpy as np
        tex = self.get_normal_map().detach().cpu()
        return (tex[0].numpy() * 255).clip(0, 255).astype(np.uint8)
