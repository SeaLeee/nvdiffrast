"""
Loss functions for Messiah texture/shader optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for visual quality comparison."""

    def __init__(self, layers=16):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:layers]
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def to(self, device):
        self.vgg = self.vgg.to(device)
        return super().to(device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, y: [B, H, W, 3] in [0, 1] range

        Returns:
            Scalar perceptual loss
        """
        # [B, H, W, 3] → [B, 3, H, W]
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        y_nchw = y.permute(0, 3, 1, 2).contiguous()

        # VGG expects normalized input
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_norm = (x_nchw - mean) / std
        y_norm = (y_nchw - mean) / std

        return F.mse_loss(self.vgg(x_norm), self.vgg(y_norm))


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss."""

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, y: [B, H, W, C] in [0, 1] range

        Returns:
            1 - SSIM (loss form)
        """
        x_nchw = x.permute(0, 3, 1, 2)
        y_nchw = y.permute(0, 3, 1, 2)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ws = self.window_size
        pad = ws // 2

        mu_x = F.avg_pool2d(x_nchw, ws, stride=1, padding=pad)
        mu_y = F.avg_pool2d(y_nchw, ws, stride=1, padding=pad)

        sigma_x2 = F.avg_pool2d(x_nchw ** 2, ws, stride=1, padding=pad) - mu_x ** 2
        sigma_y2 = F.avg_pool2d(y_nchw ** 2, ws, stride=1, padding=pad) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x_nchw * y_nchw, ws, stride=1, padding=pad) - mu_x * mu_y

        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2))

        return 1.0 - ssim.mean()


class SmoothnessLoss(nn.Module):
    """Total variation / smoothness regularizer for textures."""

    def forward(self, tex: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tex: [B, H, W, C] texture tensor

        Returns:
            Scalar smoothness loss
        """
        dx = tex[:, 1:, :, :] - tex[:, :-1, :, :]
        dy = tex[:, :, 1:, :] - tex[:, :, :-1, :]
        return torch.mean(dx ** 2) + torch.mean(dy ** 2)


class CompositeLoss(nn.Module):
    """Combined loss for Messiah texture/shader optimization."""

    def __init__(self, w_l2=1.0, w_perceptual=0.1, w_ssim=0.05,
                 w_smoothness=0.001, device='cuda'):
        super().__init__()
        self.w_l2 = w_l2
        self.w_perceptual = w_perceptual
        self.w_ssim = w_ssim
        self.w_smoothness = w_smoothness

        self.perceptual = PerceptualLoss().to(device) if w_perceptual > 0 else None
        self.ssim = SSIMLoss() if w_ssim > 0 else None
        self.smoothness = SmoothnessLoss() if w_smoothness > 0 else None

    def forward(self, rendered: torch.Tensor, reference: torch.Tensor,
                mask: torch.Tensor = None,
                optimized_textures: list = None) -> torch.Tensor:
        """
        Args:
            rendered:  [B, H, W, C] rendered image
            reference: [B, H, W, C] reference image
            mask:      [B, H, W, 1] optional mask
            optimized_textures: list of texture tensors for smoothness reg

        Returns:
            Scalar composite loss
        """
        if mask is not None:
            rendered = rendered * mask
            reference = reference * mask

        loss = self.w_l2 * F.mse_loss(rendered, reference)

        if self.perceptual is not None and self.w_perceptual > 0:
            loss = loss + self.w_perceptual * self.perceptual(rendered, reference)

        if self.ssim is not None and self.w_ssim > 0:
            loss = loss + self.w_ssim * self.ssim(rendered, reference)

        if self.smoothness is not None and self.w_smoothness > 0 and optimized_textures:
            for tex in optimized_textures:
                loss = loss + self.w_smoothness * self.smoothness(tex)

        return loss


def compute_psnr(rendered: torch.Tensor, reference: torch.Tensor,
                 mask: torch.Tensor = None) -> float:
    """Compute PSNR in dB."""
    if mask is not None:
        rendered = rendered * mask
        reference = reference * mask
    mse = F.mse_loss(rendered, reference).item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * torch.log10(torch.tensor(mse)).item()
