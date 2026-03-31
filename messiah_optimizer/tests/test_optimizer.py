"""
Tests for the optimization modules.
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class TestLosses:
    def test_ssim_loss_identical(self, device):
        """SSIM loss should be 0 for identical images."""
        from messiah_optimizer.optimizer.losses import SSIMLoss

        loss_fn = SSIMLoss(device=device)
        img = torch.rand(1, 256, 256, 3, device=device)

        loss = loss_fn(img, img)
        assert loss.item() < 0.01, f"SSIM of identical images should be ~0, got {loss.item()}"

    def test_ssim_loss_different(self, device):
        """SSIM loss should be positive for different images."""
        from messiah_optimizer.optimizer.losses import SSIMLoss

        loss_fn = SSIMLoss(device=device)
        img1 = torch.zeros(1, 256, 256, 3, device=device)
        img2 = torch.ones(1, 256, 256, 3, device=device)

        loss = loss_fn(img1, img2)
        assert loss.item() > 0.5

    def test_smoothness_loss(self, device):
        """Smoothness loss should be higher for noisy images."""
        from messiah_optimizer.optimizer.losses import SmoothnessLoss

        loss_fn = SmoothnessLoss()
        smooth = torch.ones(1, 64, 64, 3, device=device) * 0.5
        noisy = torch.rand(1, 64, 64, 3, device=device)

        l_smooth = loss_fn(smooth)
        l_noisy = loss_fn(noisy)

        assert l_smooth < l_noisy, "Noisy image should have higher TV loss"

    def test_composite_loss(self, device):
        """Composite loss should combine sub-losses correctly."""
        from messiah_optimizer.optimizer.losses import CompositeLoss

        loss_fn = CompositeLoss(
            l1_weight=1.0,
            ssim_weight=0.0,
            perceptual_weight=0.0,
            smoothness_weight=0.0,
            device=device,
        )

        img1 = torch.zeros(1, 64, 64, 3, device=device)
        img2 = torch.ones(1, 64, 64, 3, device=device)

        total, breakdown = loss_fn(img1, img2)
        # L1 between 0 and 1 should be 1.0
        assert abs(total.item() - 1.0) < 0.01

    def test_compute_psnr(self, device):
        """PSNR should be inf for identical images, reasonable for others."""
        from messiah_optimizer.optimizer.losses import compute_psnr

        img = torch.rand(1, 64, 64, 3, device=device)
        psnr_same = compute_psnr(img, img)
        assert psnr_same > 50, "PSNR of identical images should be very high"

        noise = img + torch.randn_like(img) * 0.1
        psnr_noisy = compute_psnr(img, noise)
        assert 10 < psnr_noisy < 50


class TestTextureOptimizer:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_optimizer(self):
        try:
            import nvdiffrast.torch as dr
        except ImportError:
            pytest.skip("nvdiffrast not installed")

        from messiah_optimizer.optimizer.texture_optimizer import TextureOptimizer

        opt = TextureOptimizer(
            resolution=(128, 128),
            tex_resolution=(64, 64),
        )
        assert opt is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_optimizer_single_step(self):
        try:
            import nvdiffrast.torch as dr
        except ImportError:
            pytest.skip("nvdiffrast not installed")

        from messiah_optimizer.optimizer.texture_optimizer import TextureOptimizer

        opt = TextureOptimizer(
            resolution=(64, 64),
            tex_resolution=(16, 16),
        )

        # Create simple mesh and reference
        vertices = torch.tensor([
            [-0.5, -0.5, 0.0],
            [ 0.5, -0.5, 0.0],
            [ 0.0,  0.5, 0.0],
        ], dtype=torch.float32, device='cuda')

        triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')

        vtx_attr = {
            'normal': torch.tensor([
                [0, 0, 1], [0, 0, 1], [0, 0, 1]
            ], dtype=torch.float32, device='cuda'),
            'uv': torch.tensor([
                [0, 0], [1, 0], [0.5, 1]
            ], dtype=torch.float32, device='cuda'),
            'pos_world': vertices,
        }

        ref_image = torch.rand(1, 64, 64, 3, device='cuda')

        opt.setup(vertices, triangles, vtx_attr, [ref_image])

        # Run one step
        loss = opt.step()
        assert loss is not None
        assert loss > 0


class TestShaderSimplifier:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_simplifier(self):
        try:
            import nvdiffrast.torch as dr
        except ImportError:
            pytest.skip("nvdiffrast not installed")

        from messiah_optimizer.optimizer.shader_simplifier import ShaderSimplifier

        simplifier = ShaderSimplifier(resolution=(64, 64))
        assert simplifier is not None


class TestMaterialFitter:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_fitter(self):
        try:
            import nvdiffrast.torch as dr
        except ImportError:
            pytest.skip("nvdiffrast not installed")

        from messiah_optimizer.optimizer.material_fitter import MaterialFitter

        fitter = MaterialFitter(resolution=(64, 64), tex_resolution=(16, 16))
        assert fitter is not None
