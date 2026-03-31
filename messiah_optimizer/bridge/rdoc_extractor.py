"""
RenderDoc texture extraction helper.

Provides two extraction modes:
  1. Framebuffer-based: Uses renderdoccmd thumb (already in RenderDocCapture)
  2. Replay-based: Uses renderdoccmd + Python scripting for individual texture extraction

Since renderdoccmd's Python replay scripting has limited support,
this module uses an alternative approach:
  - Convert .rdc to image sequences via renderdoccmd convert
  - Parse RenderDoc's XML export for texture references
  - Copy/rename extracted files to a structured output directory

For full texture-level extraction, users should open the .rdc in qrenderdoc GUI
and use the texture viewer to export individual textures.
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExtractedTexture:
    """A texture extracted from a RenderDoc capture."""
    resource_id: int = 0
    name: str = ''
    width: int = 0
    height: int = 0
    format: str = ''
    image_path: str = ''
    mip_levels: int = 1


@dataclass
class ExtractionResult:
    """Result of texture extraction from a .rdc capture."""
    success: bool = False
    rdc_path: str = ''
    output_dir: str = ''
    framebuffer_path: str = ''
    textures: List[ExtractedTexture] = field(default_factory=list)
    error: str = ''
    extraction_time: float = 0.0


class RenderDocExtractor:
    """
    Extracts textures and framebuffers from RenderDoc .rdc capture files.

    Uses renderdoccmd CLI for extraction. For full texture extraction,
    generates a Python replay script that renderdoccmd can execute.
    """

    def __init__(self, renderdoc_path: str = r'C:\Program Files\RenderDoc'):
        self.renderdoc_path = renderdoc_path
        self.cmd_path = os.path.join(renderdoc_path, 'renderdoccmd.exe')

    def is_available(self) -> bool:
        return os.path.exists(self.cmd_path)

    def extract_framebuffer(self, rdc_path: str, output_dir: str) -> Optional[str]:
        """Extract framebuffer thumbnail from .rdc file."""
        if not os.path.exists(rdc_path):
            return None

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, 'framebuffer.png')

        cmd = f'"{self.cmd_path}" thumb -i "{rdc_path}" -o "{out_path}" -format png'
        try:
            subprocess.run(cmd, shell=True, capture_output=True, timeout=60)
            if os.path.exists(out_path):
                return out_path
        except Exception as e:
            logger.error(f"Framebuffer extraction failed: {e}")
        return None

    def extract_all(self, rdc_path: str, output_dir: str) -> ExtractionResult:
        """
        Extract framebuffer and available texture data from a .rdc capture.

        This performs:
        1. Framebuffer extraction via renderdoccmd thumb
        2. Generate and run a replay script for texture enumeration
        """
        start = time.time()
        result = ExtractionResult(rdc_path=rdc_path, output_dir=output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Framebuffer
        fb = self.extract_framebuffer(rdc_path, output_dir)
        if fb:
            result.framebuffer_path = fb

        # Step 2: Try Python replay script for texture extraction
        textures = self._replay_extract_textures(rdc_path, output_dir)
        if textures:
            result.textures = textures

        result.success = bool(result.framebuffer_path or result.textures)
        result.extraction_time = time.time() - start
        logger.info(f"Extraction from {rdc_path}: framebuffer={'yes' if fb else 'no'}, "
                     f"textures={len(result.textures)}, time={result.extraction_time:.1f}s")
        return result

    def _replay_extract_textures(
        self,
        rdc_path: str,
        output_dir: str,
    ) -> List[ExtractedTexture]:
        """
        Run a RenderDoc Python replay script to extract texture resources.
        Creates a temporary .py script and executes it via renderdoccmd.
        """
        script_path = os.path.join(output_dir, '_extract_textures.py')
        result_path = os.path.join(output_dir, '_extraction_result.json')
        textures_dir = os.path.join(output_dir, 'textures')
        os.makedirs(textures_dir, exist_ok=True)

        # Generate the replay script
        # This script uses RenderDoc's Python API inside the replay context
        script_content = self._generate_replay_script(
            rdc_path, textures_dir, result_path
        )

        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)

        # Execute: renderdoccmd python <script> [capture]
        # Note: renderdoccmd Python execution requires specific RenderDoc version support
        cmd = f'"{self.cmd_path}" replay "{rdc_path}" --python "{script_path}"'
        logger.info(f"Running replay script: {cmd}")

        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, timeout=120
            )

            if os.path.exists(result_path):
                with open(result_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                textures = []
                for tex_info in data.get('textures', []):
                    et = ExtractedTexture(
                        resource_id=tex_info.get('id', 0),
                        name=tex_info.get('name', ''),
                        width=tex_info.get('width', 0),
                        height=tex_info.get('height', 0),
                        format=tex_info.get('format', ''),
                        image_path=tex_info.get('image_path', ''),
                    )
                    textures.append(et)
                return textures
            else:
                # Replay script didn't produce output — may not be supported
                stderr = proc.stderr.decode(errors='replace')
                if stderr:
                    logger.debug(f"Replay script stderr: {stderr[:500]}")
                logger.info("RenderDoc replay scripting not available for texture extraction. "
                            "Use qrenderdoc GUI for manual texture export.")
                return []

        except subprocess.TimeoutExpired:
            logger.warning("Replay script timed out")
            return []
        except Exception as e:
            logger.warning(f"Replay script execution failed: {e}")
            return []

    def _generate_replay_script(
        self,
        rdc_path: str,
        textures_dir: str,
        result_path: str,
    ) -> str:
        """Generate a Python script for RenderDoc replay texture extraction."""
        # Escape backslashes for the embedded Python string
        tex_dir_escaped = textures_dir.replace('\\', '\\\\')
        result_escaped = result_path.replace('\\', '\\\\')

        return f'''# Auto-generated RenderDoc texture extraction script
# Executed inside renderdoccmd replay context
import json
import os

try:
    import renderdoc as rd
except ImportError:
    # Not running inside RenderDoc context
    with open(r"{result_escaped}", "w") as f:
        json.dump({{"error": "Not in RenderDoc context", "textures": []}}, f)
    raise SystemExit(0)

def extract_textures(controller):
    textures_dir = r"{tex_dir_escaped}"
    os.makedirs(textures_dir, exist_ok=True)

    results = []
    textures = controller.GetTextures()

    for i, tex in enumerate(textures):
        name = tex.name if hasattr(tex, "name") else f"texture_{{i}}"
        width = tex.width if hasattr(tex, "width") else 0
        height = tex.height if hasattr(tex, "height") else 0
        fmt = str(tex.format.strname) if hasattr(tex, "format") else ""

        out_path = os.path.join(textures_dir, f"tex_{{i}}_{{width}}x{{height}}.png")

        # Try to save the texture
        try:
            save = rd.TextureSave()
            save.resourceId = tex.resourceId
            save.destType = rd.FileType.PNG
            save.mip = 0
            save.slice.sliceIndex = 0
            controller.SaveTexture(save, out_path)
            if os.path.exists(out_path):
                results.append({{
                    "id": int(tex.resourceId),
                    "name": name,
                    "width": width,
                    "height": height,
                    "format": fmt,
                    "image_path": out_path,
                }})
        except Exception as e:
            pass  # Some textures may not be saveable

    with open(r"{result_escaped}", "w") as f:
        json.dump({{"textures": results}}, f, indent=2)

# Run extraction
if "controller" in dir():
    extract_textures(controller)
'''

    def list_captures(self, capture_dir: str) -> List[dict]:
        """List all .rdc capture files in a directory."""
        captures = []
        if not os.path.isdir(capture_dir):
            return captures

        for fname in os.listdir(capture_dir):
            if fname.endswith('.rdc'):
                fpath = os.path.join(capture_dir, fname)
                captures.append({
                    'filename': fname,
                    'path': fpath,
                    'size': os.path.getsize(fpath),
                    'modified': os.path.getmtime(fpath),
                })
        captures.sort(key=lambda c: c['modified'], reverse=True)
        return captures
