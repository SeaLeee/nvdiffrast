"""Test GPU rendering pipeline with nvdiffrast"""
import torch
import sys
sys.path.insert(0, '.')

from pipeline.messiah_pipeline import MessiahDiffPipeline
from pipeline.camera import Camera
from pipeline.procedural import create_uv_sphere, create_solid_color_textures

device = 'cuda'
print('Creating pipeline...')
pipe = MessiahDiffPipeline(resolution=[512, 512], device=device)
print('Pipeline created')

print('Creating mesh...')
mesh = create_uv_sphere(radius=1.0, rings=48, sectors=96, device=device)
print('Mesh: verts=%s, tris=%s' % (mesh['vertices'].shape, mesh['triangles'].shape))

print('Creating textures...')
textures = create_solid_color_textures([0.7, 0.3, 0.2], roughness_val=0.4, metallic_val=0.0, resolution=256, device=device)
print('Textures: base_color=%s' % str(textures['base_color'].shape))

print('Creating camera...')
camera = Camera(position=[0, 0, 3], target=[0, 0, 0], fov=45, device=device)

print('Rendering...')
vtx_attr = {'normal': mesh['normals'], 'uv': mesh['uvs']}
result = pipe.render_from_camera(camera, mesh['vertices'], mesh['triangles'], vtx_attr, textures)
if isinstance(result, tuple):
    print('Result is tuple of %d elements' % len(result))
    for i, r in enumerate(result):
        if hasattr(r, 'shape'):
            print('  [%d] shape=%s dtype=%s' % (i, r.shape, r.dtype))
        else:
            print('  [%d] type=%s' % (i, type(r)))
else:
    print('Render result shape: %s' % str(result.shape))

# Save rendered image
import numpy as np
from PIL import Image
if isinstance(result, tuple):
    img = result[0]
else:
    img = result
img_np = img.squeeze(0).detach().cpu().clamp(0, 1).numpy()
img_np = (img_np * 255).astype(np.uint8)
if img_np.shape[-1] == 4:
    img_np = img_np[:, :, :3]
Image.fromarray(img_np).save('gpu_render_test.png')
print('Saved gpu_render_test.png')
print('GPU render OK!')
