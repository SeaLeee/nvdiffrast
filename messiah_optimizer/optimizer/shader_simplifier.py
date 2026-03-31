"""
Shader Simplifier - Fit complex shader effects to simpler PBR parameters.

Given reference images rendered with complex Messiah shaders (SSS, Cloth, etc.),
optimizes a simpler shader's parameters to approximate the visual result.
"""

import torch
import torch.nn.functional as F

from pipeline import MessiahDiffPipeline, Camera
from optimizer.losses import CompositeLoss, compute_psnr


class ShaderSimplifier:
    """
    Fit a DefaultLit shader to approximate complex shader output.

    Optimizable parameters:
      - Roughness (scalar or per-pixel bias)
      - Metallic (scalar or per-pixel bias)
      - Base color bias
      - Emission strength
    """

    def __init__(self, pipeline: MessiahDiffPipeline,
                 mesh_data: dict, cameras: list,
                 reference_images: torch.Tensor,
                 config: dict = None):
        self.pipeline = pipeline
        self.mesh = mesh_data
        self.cameras = cameras
        self.refs = reference_images
        self.device = pipeline.device

        cfg = config or {}

        # Optimizable scalar material parameters (in logit/unconstrained space)
        self.roughness_param = torch.nn.Parameter(
            torch.tensor([0.0], device=self.device)  # sigmoid(0) = 0.5
        )
        self.metallic_param = torch.nn.Parameter(
            torch.tensor([-2.0], device=self.device)  # sigmoid(-2) ≈ 0.12
        )
        self.emission_param = torch.nn.Parameter(
            torch.tensor([0.0], device=self.device)
        )

        # Per-pixel base color bias (added to texture)
        tex_h, tex_w = mesh_data['base_color_hires'].shape[1:3]
        self.color_bias = torch.nn.Parameter(
            torch.zeros(1, tex_h, tex_w, 3, device=self.device)
        )

        lr = cfg.get('learning_rate', 0.005)
        self.optimizer = torch.optim.Adam([
            {'params': [self.roughness_param, self.metallic_param], 'lr': lr},
            {'params': [self.emission_param], 'lr': lr * 0.1},
            {'params': [self.color_bias], 'lr': lr * 0.5},
        ])

        max_iter = cfg.get('max_iterations', 3000)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_iter
        )

        loss_cfg = cfg.get('loss_weights', {})
        self.criterion = CompositeLoss(
            w_l2=loss_cfg.get('l2', 1.0),
            w_perceptual=loss_cfg.get('perceptual', 0.2),
            w_ssim=loss_cfg.get('ssim', 0.1),
            w_smoothness=loss_cfg.get('smoothness', 0.0005),
            device=self.device,
        )

        self.iteration = 0

    def get_params(self) -> dict:
        """Get current optimized parameters in display-friendly format."""
        return {
            'roughness': torch.sigmoid(self.roughness_param).item(),
            'metallic': torch.sigmoid(self.metallic_param).item(),
            'emission_strength': F.softplus(self.emission_param).item(),
        }

    def step(self, view_idx=None) -> dict:
        """Execute one optimization step."""
        if view_idx is None:
            view_idx = torch.randint(len(self.cameras), (1,)).item()

        camera = self.cameras[view_idx]
        ref = self.refs[view_idx:view_idx + 1]

        # Build modified textures
        roughness_val = torch.sigmoid(self.roughness_param)
        metallic_val = torch.sigmoid(self.metallic_param)
        base_color_adj = torch.clamp(
            self.mesh['base_color_hires'] + self.color_bias, 0.0, 1.0
        )

        # Uniform roughness/metallic textures
        r_shape = self.mesh['roughness_tex'].shape
        m_shape = self.mesh['metallic_tex'].shape
        roughness_tex = roughness_val.view(1, 1, 1, 1).expand(r_shape)
        metallic_tex = metallic_val.view(1, 1, 1, 1).expand(m_shape)

        textures = {
            'base_color': base_color_adj,
            'roughness': roughness_tex,
            'metallic': metallic_tex,
        }

        # Render with DefaultLit (simplified target)
        color, mask, _ = self.pipeline.render_from_camera(
            camera, self.mesh['vertices'], self.mesh['triangles'],
            self.mesh['vtx_attr'], textures,
            light_dir=self.mesh.get('light_dir'),
            light_color=self.mesh.get('light_color'),
            shading_model_id=1,  # DefaultLit
        )

        # Add emission
        emission = F.softplus(self.emission_param) * base_color_adj
        emission_interp = color + emission * mask  # Only on visible pixels

        loss = self.criterion(emission_interp, ref, mask)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.iteration += 1
        psnr = compute_psnr(emission_interp.detach(), ref, mask.detach())

        return {
            'loss': loss.item(),
            'psnr': psnr,
            'params': self.get_params(),
            'rendered': emission_interp.detach(),
            'iteration': self.iteration,
        }

    def export_material(self) -> dict:
        """Export optimized material parameters for Messiah."""
        params = self.get_params()
        params['shading_model'] = 'DefaultLit'
        params['color_bias'] = self.color_bias.detach().cpu().numpy().tolist()
        return params
