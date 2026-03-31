from setuptools import setup, find_packages

setup(
    name="messiah-optimizer",
    version="0.1.0",
    description="NvDiffRast-based Shader/Texture Optimizer for Messiah Engine",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "nvdiffrast>=0.3.1",
        "PyQt6>=6.5.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "torchvision>=0.15.0",
        "pygltflib>=1.16.0",
        "imageio>=2.31.0",
        "matplotlib>=3.7.0",
        "PyOpenGL>=3.1.6",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "messiah-optimizer=editor.main:main",
        ],
    },
)
