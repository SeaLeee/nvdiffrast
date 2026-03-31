"""
RenderDoc capture integration for Messiah engine.

Uses renderdoc.dll (ctypes) and renderdoccmd.exe for:
  - Programmatic frame capture
  - Visual A/B comparison (before/after optimization)
  - Automated capture-and-compare workflows
"""

import ctypes
import ctypes.wintypes
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ============ RenderDoc Constants ============


class CaptureOption(IntEnum):
    AllowVSync = 0
    AllowFullscreen = 1
    APIValidation = 2
    CaptureCallstacks = 3
    CaptureCallstacksOnlyActions = 4
    DelayForDebugger = 5
    VerifyBufferAccess = 6
    HookIntoChildren = 7
    RefAllResources = 8
    CaptureAllCmdLists = 10
    DebugOutputMute = 11
    AllowUnsupportedVendorExtensions = 12
    SoftMemoryLimit = 13


class OverlayBits(IntEnum):
    Enabled = 0x1
    FrameRate = 0x2
    FrameNumber = 0x4
    CaptureList = 0x8
    Default = 0x1 | 0x2 | 0x4 | 0x8
    All = 0xFFFFFFFF
    Disabled = 0


@dataclass
class CaptureResult:
    """Result of a frame capture operation."""
    success: bool
    filepath: str = ''
    timestamp: float = 0.0
    error: str = ''


class RenderDocCapture:
    """
    Interface to RenderDoc for programmatic frame capture.
    """

    # Default installation path
    DEFAULT_INSTALL = r'C:\Program Files\RenderDoc'

    def __init__(self, renderdoc_path: str = None):
        """
        Args:
            renderdoc_path: Path to RenderDoc installation directory.
                           If None, uses default path.
        """
        self.renderdoc_path = renderdoc_path or self.DEFAULT_INSTALL
        self.dll_path = os.path.join(self.renderdoc_path, 'renderdoc.dll')
        self.cmd_path = os.path.join(self.renderdoc_path, 'renderdoccmd.exe')

        self._api = None
        self._loaded = False
        self._capture_dir = ''

    # ============ Discovery ============

    @staticmethod
    def find_renderdoc() -> Optional[str]:
        """Find RenderDoc installation path."""
        candidates = [
            r'C:\Program Files\RenderDoc',
            r'C:\Program Files (x86)\RenderDoc',
            os.path.expandvars(r'%LOCALAPPDATA%\RenderDoc'),
        ]
        for path in candidates:
            if os.path.exists(os.path.join(path, 'renderdoc.dll')):
                return path
        return None

    def is_available(self) -> bool:
        """Check if RenderDoc is available."""
        return os.path.exists(self.dll_path) and os.path.exists(self.cmd_path)

    # ============ In-Process API (ctypes) ============

    def load_api(self) -> bool:
        """
        Load the RenderDoc in-process API via ctypes.
        NOTE: This must be called BEFORE the target application creates any
        graphics API device (D3D, OpenGL, Vulkan). Typically used when
        the optimizer itself hosts the rendering, or when injected early.
        """
        if self._loaded:
            return True

        if not os.path.exists(self.dll_path):
            logger.error(f"renderdoc.dll not found at {self.dll_path}")
            return False

        try:
            dll = ctypes.cdll.LoadLibrary(self.dll_path)

            # RENDERDOC_GetAPI(eRENDERDOC_API_1_6_0, &api)
            RENDERDOC_GetAPI = dll.RENDERDOC_GetAPI
            RENDERDOC_GetAPI.restype = ctypes.c_int
            RENDERDOC_GetAPI.argtypes = [ctypes.c_int, ctypes.c_void_p]

            api_ptr = ctypes.c_void_p()
            # eRENDERDOC_API_Version_1_6_0 = 10600
            ret = RENDERDOC_GetAPI(10600, ctypes.byref(api_ptr))
            if ret != 1:
                logger.error(f"RENDERDOC_GetAPI returned {ret}")
                return False

            self._api = api_ptr
            self._loaded = True
            logger.info("RenderDoc API loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load RenderDoc API: {e}")
            return False

    # ============ CLI-based Capture ============

    def capture_with_cmd(
        self,
        exe_path: str,
        working_dir: str = None,
        args: str = '',
        capture_dir: str = None,
        wait_seconds: float = 10.0,
        num_frames: int = 1,
    ) -> CaptureResult:
        """
        Launch an application under RenderDoc and capture frames.
        Uses renderdoccmd.exe for out-of-process capture.

        Args:
            exe_path: Path to the target executable
            working_dir: Working directory for the executable
            args: Command-line arguments for the executable
            capture_dir: Directory to save captures. If None, uses temp dir.
            wait_seconds: Seconds to wait before capture
            num_frames: Number of frames to capture
        """
        if not os.path.exists(self.cmd_path):
            return CaptureResult(success=False, error='renderdoccmd.exe not found')

        if not os.path.exists(exe_path):
            return CaptureResult(success=False, error=f'Executable not found: {exe_path}')

        if capture_dir is None:
            capture_dir = os.path.join(os.path.dirname(exe_path), 'rdoc_captures')
        os.makedirs(capture_dir, exist_ok=True)
        self._capture_dir = capture_dir

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        capture_file = os.path.join(capture_dir, f'capture_{timestamp}')

        # Build renderdoccmd command
        cmd_parts = [
            f'"{self.cmd_path}"',
            'capture',
            f'--wait-for-exit',
            f'--capture-file "{capture_file}"',
            f'-w "{working_dir or os.path.dirname(exe_path)}"',
        ]

        if args:
            cmd_parts.append(f'-- "{exe_path}" {args}')
        else:
            cmd_parts.append(f'-- "{exe_path}"')

        cmd = ' '.join(cmd_parts)
        logger.info(f"Launching RenderDoc capture: {cmd}")

        try:
            proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Wait for capture to complete (or timeout)
            timeout = max(wait_seconds + 30, 120)
            stdout, stderr = proc.communicate(timeout=timeout)

            # Look for the .rdc file
            rdc_files = [f for f in os.listdir(capture_dir)
                         if f.startswith(f'capture_{timestamp}') and f.endswith('.rdc')]

            if rdc_files:
                rdc_path = os.path.join(capture_dir, rdc_files[0])
                return CaptureResult(
                    success=True,
                    filepath=rdc_path,
                    timestamp=time.time(),
                )
            else:
                return CaptureResult(
                    success=False,
                    error=f'No .rdc file generated. stdout={stdout.decode(errors="replace")}, '
                          f'stderr={stderr.decode(errors="replace")}',
                )
        except subprocess.TimeoutExpired:
            proc.kill()
            return CaptureResult(success=False, error='Capture timed out')
        except Exception as e:
            return CaptureResult(success=False, error=str(e))

    # ============ Replay / Extraction ============

    def extract_framebuffer(
        self,
        rdc_path: str,
        output_path: str = None,
    ) -> Optional[str]:
        """
        Extract the final framebuffer from a .rdc capture as an image.
        Uses renderdoccmd's replay capabilities.

        Args:
            rdc_path: Path to the .rdc capture file
            output_path: Output image path. If None, generates automatically.

        Returns:
            Path to the extracted image, or None on failure.
        """
        if not os.path.exists(rdc_path):
            logger.error(f".rdc file not found: {rdc_path}")
            return None

        if output_path is None:
            base = os.path.splitext(rdc_path)[0]
            output_path = f'{base}_framebuffer.png'

        # Use renderdoccmd to extract thumbnail/framebuffer
        cmd = f'"{self.cmd_path}" thumb -i "{rdc_path}" -o "{output_path}" -format png'
        logger.info(f"Extracting framebuffer: {cmd}")

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, timeout=60
            )
            if os.path.exists(output_path):
                logger.info(f"Framebuffer extracted to: {output_path}")
                return output_path
            else:
                logger.warning(f"Failed to extract framebuffer. "
                               f"stderr={result.stderr.decode(errors='replace')}")
                return None
        except Exception as e:
            logger.error(f"Error extracting framebuffer: {e}")
            return None

    def open_in_renderdoc(self, rdc_path: str) -> bool:
        """Open a capture file in RenderDoc GUI for manual inspection."""
        if not os.path.exists(rdc_path):
            return False

        qrenderdoc = os.path.join(self.renderdoc_path, 'qrenderdoc.exe')
        if not os.path.exists(qrenderdoc):
            logger.error("qrenderdoc.exe not found")
            return False

        try:
            subprocess.Popen([qrenderdoc, rdc_path])
            return True
        except Exception as e:
            logger.error(f"Failed to open RenderDoc: {e}")
            return False

    # ============ A/B Comparison ============

    def compare_captures(
        self,
        before_image: str,
        after_image: str,
        output_path: str = None,
    ) -> Optional[dict]:
        """
        Compare two framebuffer images (before/after optimization).
        Returns pixel difference metrics.

        Args:
            before_image: Path to the 'before' framebuffer image
            after_image: Path to the 'after' framebuffer image
            output_path: Optional path to save the diff image

        Returns:
            Dictionary with comparison metrics, or None on failure.
        """
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            logger.error("Pillow and numpy required for image comparison")
            return None

        if not os.path.exists(before_image) or not os.path.exists(after_image):
            logger.error("One or both images not found")
            return None

        try:
            img_before = np.array(Image.open(before_image).convert('RGB'), dtype=np.float32)
            img_after = np.array(Image.open(after_image).convert('RGB'), dtype=np.float32)

            # Resize if different dimensions
            if img_before.shape != img_after.shape:
                h = min(img_before.shape[0], img_after.shape[0])
                w = min(img_before.shape[1], img_after.shape[1])
                img_before = img_before[:h, :w]
                img_after = img_after[:h, :w]

            # Compute metrics
            diff = np.abs(img_before - img_after)
            mse = np.mean(diff ** 2)
            psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-10)) if mse > 0 else float('inf')
            max_diff = float(np.max(diff))
            mean_diff = float(np.mean(diff))

            # Per-channel stats
            channel_mse = [float(np.mean(diff[:, :, c] ** 2)) for c in range(3)]

            result = {
                'mse': float(mse),
                'psnr': float(psnr),
                'max_pixel_diff': max_diff,
                'mean_pixel_diff': mean_diff,
                'channel_mse': {'R': channel_mse[0], 'G': channel_mse[1], 'B': channel_mse[2]},
                'resolution': list(img_before.shape[:2]),
                'identical': bool(mse < 1e-6),
            }

            # Save diff image if requested
            if output_path:
                diff_scaled = np.clip(diff * 10, 0, 255).astype(np.uint8)  # Amplify diff
                Image.fromarray(diff_scaled).save(output_path)
                result['diff_image'] = output_path

            return result
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return None


class CaptureWorkflow:
    """
    High-level workflow for capture-based A/B comparison.
    Integrates resource resolution + frame capture.
    """

    def __init__(self, renderdoc: RenderDocCapture, capture_dir: str):
        self.renderdoc = renderdoc
        self.capture_dir = capture_dir
        os.makedirs(capture_dir, exist_ok=True)

        self.before_capture: Optional[str] = None
        self.after_capture: Optional[str] = None

    def capture_before(self, rdc_path: str) -> bool:
        """
        Register the 'before' frame capture (either an .rdc or extracted image).
        Can be a pre-existing .rdc file or image.
        """
        if not os.path.exists(rdc_path):
            return False

        if rdc_path.endswith('.rdc'):
            # Extract framebuffer
            img_path = self.renderdoc.extract_framebuffer(
                rdc_path,
                os.path.join(self.capture_dir, 'before_framebuffer.png'),
            )
            self.before_capture = img_path
        else:
            # Assume it's already an image
            self.before_capture = rdc_path
        return self.before_capture is not None

    def capture_after(self, rdc_path: str) -> bool:
        """Register the 'after' frame capture."""
        if not os.path.exists(rdc_path):
            return False

        if rdc_path.endswith('.rdc'):
            img_path = self.renderdoc.extract_framebuffer(
                rdc_path,
                os.path.join(self.capture_dir, 'after_framebuffer.png'),
            )
            self.after_capture = img_path
        else:
            self.after_capture = rdc_path
        return self.after_capture is not None

    def compare(self) -> Optional[dict]:
        """Compare before/after captures."""
        if not self.before_capture or not self.after_capture:
            logger.error("Need both before and after captures")
            return None

        diff_path = os.path.join(self.capture_dir, 'diff_image.png')
        return self.renderdoc.compare_captures(
            self.before_capture, self.after_capture, diff_path
        )
