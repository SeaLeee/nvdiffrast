from .camera import Camera, create_orbit_cameras
from .tonemapping import aces_tonemap, reinhard_tonemap

# Lazy imports for modules that require nvdiffrast/CUDA
def __getattr__(name):
    if name == 'MessiahDiffPipeline':
        from .messiah_pipeline import MessiahDiffPipeline
        return MessiahDiffPipeline
    if name == 'GGX_BRDF':
        from .brdf import GGX_BRDF
        return GGX_BRDF
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")
