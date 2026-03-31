"""Quick integration test for scene loading + rendering."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from pipeline.procedural import create_uv_sphere, create_solid_color_textures
from pipeline.software_renderer import render_preview

# Load JSON scene
with open('samples/demo_scene.json', 'r') as f:
    scene_data = json.load(f)

# Create mesh
mesh = create_uv_sphere(radius=1.0, rings=32, sectors=64, device='cpu')

# Extract material
mat = scene_data['materials'][0]
color = mat.get('base_color', [0.8, 0.8, 0.82])
rough = mat.get('roughness', 0.5)
metal = mat.get('metallic', 0.0)
textures = create_solid_color_textures(color, rough, metal, resolution=64, device='cpu')

# Camera
cam = {'position': [2.82, 1.03, 2.82], 'target': [0, 0, 0], 'fov': 60.0, 'azimuth': 0, 'elevation': 20}
light_dir = scene_data['lights'][0].get('direction', None)

# Render
t0 = time.time()
image = render_preview(
    mesh['vertices'], mesh['triangles'], mesh['normals'],
    mesh['uvs'], textures, cam, resolution=(384, 384), light_dir=light_dir)
dt = time.time() - t0

print(f"Scene: {scene_data['scene_name']}")
print(f"Material: {mat['name']} / {mat['shading_model']}")
print(f"BaseColor: {color}, Rough: {rough}, Metal: {metal}")
print(f"Rendered {image.shape} in {dt:.2f}s")
fg = (image.sum(axis=-1) > 50).sum()
print(f"Foreground pixels: {fg}/{384*384}")

# Save test image
from PIL import Image
img = Image.fromarray(image)
img.save('samples/test_render.png')
print("Saved: samples/test_render.png")
print("SUCCESS")
