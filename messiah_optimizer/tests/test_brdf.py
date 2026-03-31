"""
Tests for the BRDF implementation.
Verifies our PyTorch BRDF matches Messiah's BRDF.fxh.
"""

import pytest
import torch
import math


@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class TestGGXDistribution:
    def test_d_ggx_at_zero_angle(self, device):
        """D_GGX should be maximized when NdotH = 1."""
        from messiah_optimizer.pipeline.brdf import GGX_BRDF

        roughness = torch.tensor([0.5], device=device)
        ndoth_peak = torch.tensor([1.0], device=device)
        ndoth_off = torch.tensor([0.5], device=device)

        d_peak = GGX_BRDF.D_GGX(roughness, ndoth_peak)
        d_off = GGX_BRDF.D_GGX(roughness, ndoth_off)

        assert d_peak > d_off, "D_GGX should peak at NdotH=1"

    def test_d_ggx_roughness_effect(self, device):
        """Higher roughness should give wider, lower peak."""
        from messiah_optimizer.pipeline.brdf import GGX_BRDF

        ndoth = torch.tensor([1.0], device=device)
        d_smooth = GGX_BRDF.D_GGX(torch.tensor([0.1], device=device), ndoth)
        d_rough = GGX_BRDF.D_GGX(torch.tensor([0.8], device=device), ndoth)

        assert d_smooth > d_rough, "Smoother surface should have taller peak"

    def test_d_ggx_min_roughness(self, device):
        """Very low roughness should be clamped to MIN_ROUGHNESS."""
        from messiah_optimizer.pipeline.brdf import GGX_BRDF

        ndoth = torch.tensor([0.9], device=device)
        d_zero = GGX_BRDF.D_GGX(torch.tensor([0.0], device=device), ndoth)
        d_min = GGX_BRDF.D_GGX(torch.tensor([0.08], device=device), ndoth)

        # With clamping, D(0.0) should equal D(0.08)
        assert torch.allclose(d_zero, d_min, atol=1e-4)


class TestFresnel:
    def test_f_schlick_grazing_angle(self, device):
        """At grazing angle (VdotH=0), Fresnel should approach 1."""
        from messiah_optimizer.pipeline.brdf import GGX_BRDF

        f0 = torch.tensor([0.04], device=device)
        vdoth_grazing = torch.tensor([0.001], device=device)

        f = GGX_BRDF.F_Schlick(f0, vdoth_grazing)
        assert f.item() > 0.95, "Fresnel should approach 1 at grazing angle"

    def test_f_schlick_normal_incidence(self, device):
        """At normal incidence (VdotH=1), F should equal F0."""
        from messiah_optimizer.pipeline.brdf import GGX_BRDF

        f0 = torch.tensor([0.04], device=device)
        vdoth = torch.tensor([1.0], device=device)

        f = GGX_BRDF.F_Schlick(f0, vdoth)
        assert torch.allclose(f, f0, atol=1e-6)

    def test_f_schlick_metal(self, device):
        """Metal (F0~0.9) should have high Fresnel even at normal."""
        from messiah_optimizer.pipeline.brdf import GGX_BRDF

        f0 = torch.tensor([0.9], device=device)
        vdoth = torch.tensor([1.0], device=device)

        f = GGX_BRDF.F_Schlick(f0, vdoth)
        assert f.item() > 0.85


class TestGeometry:
    def test_vis_range(self, device):
        """Visibility should be in [0, inf) range and finite."""
        from messiah_optimizer.pipeline.brdf import GGX_BRDF

        roughness = torch.tensor([0.5], device=device)
        ndotv = torch.tensor([0.5], device=device)
        ndotl = torch.tensor([0.5], device=device)

        vis = GGX_BRDF.Vis_Schlick_Disney(roughness, ndotv, ndotl)
        assert vis.item() >= 0
        assert torch.isfinite(vis).all()


class TestFullBRDF:
    def test_evaluate_shape(self, device):
        """BRDF evaluate should return correct shape."""
        from messiah_optimizer.pipeline.brdf import GGX_BRDF

        N = 100
        normal = torch.randn(N, 3, device=device)
        normal = torch.nn.functional.normalize(normal, dim=-1)
        view_dir = torch.randn(N, 3, device=device)
        view_dir = torch.nn.functional.normalize(view_dir, dim=-1)
        light_dir = torch.randn(N, 3, device=device)
        light_dir = torch.nn.functional.normalize(light_dir, dim=-1)

        base_color = torch.rand(N, 3, device=device)
        roughness = torch.rand(N, 1, device=device)
        metallic = torch.zeros(N, 1, device=device)

        result = GGX_BRDF.evaluate(
            normal, view_dir, light_dir,
            base_color, roughness, metallic,
        )

        assert result.shape == (N, 3)

    def test_evaluate_differentiable(self, device):
        """BRDF should be differentiable w.r.t. roughness."""
        from messiah_optimizer.pipeline.brdf import GGX_BRDF

        normal = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        view_dir = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        light_dir = torch.tensor([[0.0, 0.5, 0.866]], device=device)

        base_color = torch.tensor([[0.5, 0.5, 0.5]], device=device)
        roughness = torch.tensor([[0.5]], device=device, requires_grad=True)
        metallic = torch.tensor([[0.0]], device=device)

        result = GGX_BRDF.evaluate(
            normal, view_dir, light_dir,
            base_color, roughness, metallic,
        )

        result.sum().backward()
        assert roughness.grad is not None

    def test_energy_conservation(self, device):
        """BRDF * NdotL integrated over hemisphere should not exceed 1."""
        from messiah_optimizer.pipeline.brdf import GGX_BRDF

        normal = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        view_dir = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        base_color = torch.tensor([[1.0, 1.0, 1.0]], device=device)
        roughness = torch.tensor([[0.5]], device=device)
        metallic = torch.tensor([[0.0]], device=device)

        # Monte Carlo integration over hemisphere
        N_samples = 10000
        torch.manual_seed(42)

        # Random directions in upper hemisphere
        theta = torch.acos(torch.rand(N_samples, device=device))  # [0, pi/2]
        phi = 2 * math.pi * torch.rand(N_samples, device=device)

        light_dirs = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ], dim=-1)

        normals = normal.expand(N_samples, -1)
        views = view_dir.expand(N_samples, -1)
        colors = base_color.expand(N_samples, -1)
        roughs = roughness.expand(N_samples, -1)
        metals = metallic.expand(N_samples, -1)

        brdf_val = GGX_BRDF.evaluate(normals, views, light_dirs, colors, roughs, metals)

        ndotl = torch.clamp(light_dirs[:, 2:3], 0, 1)
        integrand = brdf_val * ndotl

        # Monte Carlo: integral ≈ (2π) * mean(integrand)
        integral = 2 * math.pi * integrand.mean(dim=0)

        # Should be <= 1 per channel (with tolerance for numerical issues)
        assert (integral < 1.1).all(), \
            f"Energy not conserved: integral = {integral}"
