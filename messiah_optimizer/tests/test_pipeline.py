"""
Tests for the differentiable rendering pipeline.
"""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def simple_triangle(device):
    """A single triangle mesh for basic tests."""
    vertices = torch.tensor([
        [-0.5, -0.5, 0.0],
        [ 0.5, -0.5, 0.0],
        [ 0.0,  0.5, 0.0],
    ], dtype=torch.float32, device=device)

    triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)

    normals = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)

    uvs = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 1.0],
    ], dtype=torch.float32, device=device)

    return {
        'vertices': vertices,
        'triangles': triangles,
        'normals': normals,
        'uvs': uvs,
        'vertex_count': 3,
        'triangle_count': 1,
    }


@pytest.fixture
def simple_cube(device):
    """A unit cube."""
    vertices = torch.tensor([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    ], dtype=torch.float32, device=device)

    triangles = torch.tensor([
        [0,1,2], [0,2,3], [4,6,5], [4,7,6],
        [0,4,5], [0,5,1], [2,6,7], [2,7,3],
        [0,3,7], [0,7,4], [1,5,6], [1,6,2],
    ], dtype=torch.int32, device=device)

    return {'vertices': vertices, 'triangles': triangles}


class TestCamera:
    def test_camera_creation(self, device):
        from messiah_optimizer.pipeline.camera import Camera

        cam = Camera(
            eye=[0, 0, 3],
            target=[0, 0, 0],
            up=[0, 1, 0],
            fov=60.0,
            aspect=1.0,
            device=device,
        )

        assert cam.mvp is not None
        assert cam.mvp.shape == (4, 4)

    def test_create_orbit_cameras(self, device):
        from messiah_optimizer.pipeline.camera import create_orbit_cameras

        cameras = create_orbit_cameras(
            num_views=8, distance=3.0, target=[0, 0, 0],
            fov=60.0, aspect=1.0, device=device,
        )

        assert len(cameras) == 8
        for cam in cameras:
            assert cam.mvp.shape == (4, 4)

    def test_transform_pos(self, device):
        from messiah_optimizer.pipeline.camera import Camera, transform_pos

        cam = Camera(eye=[0, 0, 3], target=[0, 0, 0], device=device)
        pos = torch.tensor([[0, 0, 0]], dtype=torch.float32, device=device)

        clip_pos = transform_pos(cam.mvp, pos)
        assert clip_pos.shape == (1, 4)


class TestTonemapping:
    def test_linear_to_srgb_roundtrip(self, device):
        from messiah_optimizer.pipeline.tonemapping import linear_to_srgb, srgb_to_linear

        linear = torch.rand(1, 64, 64, 3, device=device)
        srgb = linear_to_srgb(linear)
        back = srgb_to_linear(srgb)

        assert torch.allclose(linear, back, atol=1e-5)

    def test_aces_tonemap_range(self, device):
        from messiah_optimizer.pipeline.tonemapping import aces_tonemap

        hdr = torch.rand(1, 64, 64, 3, device=device) * 10.0
        ldr = aces_tonemap(hdr)

        assert ldr.min() >= 0.0
        assert ldr.max() <= 1.05  # Allow slight overshoot

    def test_reinhard_tonemap(self, device):
        from messiah_optimizer.pipeline.tonemapping import reinhard_tonemap

        hdr = torch.ones(1, 64, 64, 3, device=device)
        ldr = reinhard_tonemap(hdr)

        # Reinhard(1.0) = 1.0 / 2.0 = 0.5
        assert torch.allclose(ldr, torch.full_like(ldr, 0.5), atol=0.01)


class TestPipelineIntegration:
    """Integration tests requiring nvdiffrast CUDA."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_pipeline_render(self, simple_triangle):
        try:
            import nvdiffrast.torch as dr
        except ImportError:
            pytest.skip("nvdiffrast not installed")

        from messiah_optimizer.pipeline.messiah_pipeline import MessiahDiffPipeline
        from messiah_optimizer.pipeline.camera import Camera

        pipeline = MessiahDiffPipeline(resolution=(256, 256))

        cam = Camera(eye=[0, 0, 2], target=[0, 0, 0], device='cuda')

        vtx_attr = {
            'normal': simple_triangle['normals'],
            'uv': simple_triangle['uvs'],
            'pos_world': simple_triangle['vertices'],
        }

        base_color = torch.ones(1, 16, 16, 3, device='cuda') * 0.5
        base_color.requires_grad_(True)

        result = pipeline.render(
            vertices=simple_triangle['vertices'],
            triangles=simple_triangle['triangles'],
            vtx_attr=vtx_attr,
            mvp=cam.mvp,
            base_color_tex=base_color,
        )

        assert 'color' in result
        assert result['color'].shape[0] == 1
        assert result['color'].shape[3] == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_pipeline_gradient_flow(self, simple_triangle):
        try:
            import nvdiffrast.torch as dr
        except ImportError:
            pytest.skip("nvdiffrast not installed")

        from messiah_optimizer.pipeline.messiah_pipeline import MessiahDiffPipeline
        from messiah_optimizer.pipeline.camera import Camera

        pipeline = MessiahDiffPipeline(resolution=(128, 128))
        cam = Camera(eye=[0, 0, 2], target=[0, 0, 0], device='cuda')

        vtx_attr = {
            'normal': simple_triangle['normals'],
            'uv': simple_triangle['uvs'],
            'pos_world': simple_triangle['vertices'],
        }

        tex = torch.ones(1, 8, 8, 3, device='cuda') * 0.5
        tex.requires_grad_(True)

        result = pipeline.render(
            vertices=simple_triangle['vertices'],
            triangles=simple_triangle['triangles'],
            vtx_attr=vtx_attr,
            mvp=cam.mvp,
            base_color_tex=tex,
        )

        loss = result['color'].mean()
        loss.backward()

        assert tex.grad is not None
        assert tex.grad.abs().sum() > 0, "Gradients should flow through the pipeline"
