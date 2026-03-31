"""
Local Bridge Server — runs in the optimizer process.

Architecture:
  - Uses ResourceResolver to parse iworld → ilevel → repository XML
    for targeted resource acquisition (no full directory scanning)
  - Integrates RenderDoc capture for visual A/B comparison
  - Writes optimized files back to engine/repository directories

Resource flow:
  1. User selects an .iworld file (a specific world)
  2. Bridge resolves which resources that world uses (via GUID chain)
  3. Optimizer works on only those resources
  4. Optimized results written back; RenderDoc captures for visual diff
"""

import json
import os
import shutil
import logging
from typing import Optional, List

from .resource_resolver import ResourceResolver, WorldInfo, ResourceInfo
from .renderdoc_capture import RenderDocCapture, CaptureWorkflow
from .unified_pipeline import UnifiedPipeline, ResourceSource, ComparisonResult
from .rdoc_extractor import RenderDocExtractor

logger = logging.getLogger(__name__)


class LocalBridgeServer:
    """
    Bridge between optimizer and Messiah engine project.

    Uses targeted resource resolution (iworld/ilevel/repository XML)
    instead of scanning the entire Content/Shaders tree.
    Combines repository-based and RenderDoc-based approaches via UnifiedPipeline.
    """

    def __init__(self, engine_root: str,
                 worlds_dir: str = None,
                 repository_dir: str = None):
        """
        Args:
            engine_root: Engine root directory, e.g. D:\\NewTrunk\\Engine\\src\\Engine
            worlds_dir: Path to Worlds. Default: I:\\trunk_bjs\\common\\resource\\Package\\Worlds
            repository_dir: Path to Repository. Default: I:\\trunk_bjs\\common\\resource\\Package\\Repository
        """
        self.engine_root = os.path.normpath(engine_root)
        self.worlds_dir = worlds_dir or r'I:\trunk_bjs\common\resource\Package\Worlds'
        self.repository_dir = repository_dir or r'I:\trunk_bjs\common\resource\Package\Repository'

        self._running = False
        self._resolver = ResourceResolver(self.worlds_dir, self.repository_dir)
        self._renderdoc = RenderDocCapture()
        self._extractor = RenderDocExtractor(self._renderdoc.renderdoc_path)
        self._capture_workflow: Optional[CaptureWorkflow] = None
        self._current_world: Optional[WorldInfo] = None

        # Unified pipeline (lazy init after start)
        self._pipeline: Optional[UnifiedPipeline] = None
        # Snapshots for A/B comparison
        self._snapshot_before: Optional[UnifiedPipeline.Snapshot] = None
        self._snapshot_after: Optional[UnifiedPipeline.Snapshot] = None

        self._validate_paths()

    def _validate_paths(self):
        """Check that key paths exist."""
        if not os.path.isdir(self.engine_root):
            logger.warning(f"Engine root not found: {self.engine_root}")
        if not os.path.isdir(self.worlds_dir):
            logger.warning(f"Worlds directory not found: {self.worlds_dir}")
        if not os.path.isdir(self.repository_dir):
            logger.warning(f"Repository directory not found: {self.repository_dir}")

    @property
    def content_dir(self) -> str:
        return os.path.join(self.engine_root, 'Content')

    @property
    def shaders_dir(self) -> str:
        return os.path.join(self.engine_root, 'Shaders')

    @property
    def local_data_dir(self) -> str:
        return os.path.join(self.engine_root, 'LocalData')

    def start(self):
        self._running = True
        # Initialize unified pipeline
        output_dir = os.path.join(self.engine_root, 'LocalData', 'optimizer_output')
        self._pipeline = UnifiedPipeline(self._resolver, self._renderdoc, output_dir)
        logger.info(f"Local bridge started — engine: {self.engine_root}")
        logger.info(f"  Worlds: {self.worlds_dir}")
        logger.info(f"  Repository: {self.repository_dir}")

    def stop(self):
        self._running = False
        self._current_world = None
        logger.info("Local bridge stopped")

    def is_connected(self) -> bool:
        return self._running and os.path.isdir(self.engine_root)

    # ============ World Selection & Resource Resolution ============

    def list_worlds(self) -> List[dict]:
        """List all available .iworld files."""
        return self._resolver.list_worlds()

    def select_world(self, iworld_path: str) -> dict:
        """
        Select a world and resolve its resources.
        This is the primary entry point — replaces the old export_scene().

        Args:
            iworld_path: Full path to the .iworld file

        Returns:
            World stats and resource summary
        """
        logger.info(f"Selecting world: {iworld_path}")

        # Parse world → collect GUIDs
        world = self._resolver.parse_world(iworld_path)

        # Resolve GUIDs → actual resource info
        world = self._resolver.resolve_world(world)
        self._current_world = world

        stats = self._resolver.get_world_stats(world)
        logger.info(f"World resolved: {stats}")
        return stats

    def get_current_world(self) -> Optional[WorldInfo]:
        return self._current_world

    def get_textures(self) -> List[ResourceInfo]:
        """Get all textures from the currently selected world."""
        if not self._current_world:
            return []
        return self._resolver.get_textures(self._current_world)

    def get_meshes(self) -> List[ResourceInfo]:
        if not self._current_world:
            return []
        return self._resolver.get_meshes(self._current_world)

    def get_materials(self) -> List[ResourceInfo]:
        if not self._current_world:
            return []
        return self._resolver.get_materials(self._current_world)

    def get_effects(self) -> List[ResourceInfo]:
        if not self._current_world:
            return []
        return self._resolver.get_effects(self._current_world)

    def get_resource_data_path(self, guid: str) -> Optional[str]:
        """Get the actual data file path for a resource GUID."""
        info = self._resolver.get_resource(guid)
        if info:
            return self._resolver.get_resource_data_path(info)
        return None

    def export_scene(self, output_dir: str) -> dict:
        """
        Export the currently selected world's resources to a directory.
        Replaces old scanning approach with targeted resolution.
        """
        if not self._current_world:
            return {'error': 'No world selected. Call select_world() first.'}

        os.makedirs(output_dir, exist_ok=True)
        world = self._current_world

        result = {
            'engine_root': self.engine_root,
            'world_name': world.name,
            'stats': self._resolver.get_world_stats(world),
            'textures': [
                {'guid': r.guid, 'name': r.name, 'type': r.type,
                 'package': r.package, 'source_path': r.source_path}
                for r in self._resolver.get_textures(world)
            ],
            'materials': [
                {'guid': r.guid, 'name': r.name, 'type': r.type,
                 'package': r.package}
                for r in self._resolver.get_materials(world)
            ],
            'meshes': [
                {'guid': r.guid, 'name': r.name, 'type': r.type,
                 'package': r.package}
                for r in self._resolver.get_meshes(world)
            ],
        }

        manifest_path = os.path.join(output_dir, 'scene_manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Scene exported to {output_dir}: "
                     f"{len(result['textures'])} textures, "
                     f"{len(result['materials'])} materials, "
                     f"{len(result['meshes'])} meshes")
        return result

    # ============ RenderDoc Capture ============

    def is_renderdoc_available(self) -> bool:
        return self._renderdoc.is_available()

    def capture_before(self, rdc_path: str) -> bool:
        """Register a 'before optimization' capture."""
        capture_dir = os.path.join(self.engine_root, 'LocalData', 'rdoc_comparison')
        if not self._capture_workflow:
            self._capture_workflow = CaptureWorkflow(self._renderdoc, capture_dir)
        return self._capture_workflow.capture_before(rdc_path)

    def capture_after(self, rdc_path: str) -> bool:
        """Register an 'after optimization' capture."""
        if not self._capture_workflow:
            capture_dir = os.path.join(self.engine_root, 'LocalData', 'rdoc_comparison')
            self._capture_workflow = CaptureWorkflow(self._renderdoc, capture_dir)
        return self._capture_workflow.capture_after(rdc_path)

    def compare_captures(self) -> Optional[dict]:
        """Compare before/after RenderDoc captures."""
        if not self._capture_workflow:
            return None
        return self._capture_workflow.compare()

    def open_capture_in_renderdoc(self, rdc_path: str) -> bool:
        """Open a .rdc file in RenderDoc GUI."""
        return self._renderdoc.open_in_renderdoc(rdc_path)

    # ============ Import / Write Results ============

    def import_texture(self, source: str, target_asset: str) -> dict:
        """
        Copy an optimized texture into the engine project.

        Args:
            source: Path to the optimized texture file
            target_asset: Relative path inside engine root (e.g. Content/Textures/diffuse.png)

        Returns:
            dict with status and target path
        """
        dst = os.path.join(self.engine_root, target_asset)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        # Backup original
        if os.path.exists(dst):
            backup = dst + '.bak'
            if not os.path.exists(backup):
                shutil.copy2(dst, backup)
                logger.info(f"Backed up: {target_asset} -> {target_asset}.bak")

        shutil.copy2(source, dst)
        logger.info(f"Imported texture: {source} -> {dst}")

        return {'status': 'ok', 'target': dst}

    def import_shader(self, source: str, target_asset: str) -> dict:
        """Copy an optimized shader into the engine project."""
        dst = os.path.join(self.engine_root, target_asset)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if os.path.exists(dst):
            backup = dst + '.bak'
            if not os.path.exists(backup):
                shutil.copy2(dst, backup)

        shutil.copy2(source, dst)
        logger.info(f"Imported shader: {source} -> {dst}")
        return {'status': 'ok', 'target': dst}

    def batch_import(self, results_dir: str, manifest: dict) -> dict:
        """
        Import all optimizer results from a directory.

        Args:
            results_dir: Directory with optimized files
            manifest: dict mapping output filenames to target asset paths

        Returns:
            Summary of imported files
        """
        imported = []
        errors = []

        for filename, target_asset in manifest.items():
            src = os.path.join(results_dir, filename)
            if not os.path.exists(src):
                errors.append(f"Missing: {filename}")
                continue
            try:
                self.import_texture(src, target_asset)
                imported.append(target_asset)
            except Exception as e:
                errors.append(f"{filename}: {e}")

        return {
            'status': 'ok' if not errors else 'partial',
            'imported': imported,
            'errors': errors,
        }

    # ============ Read specific files ============

    def read_material_file(self, rel_path: str) -> Optional[dict]:
        """Read and parse a material file."""
        full_path = os.path.join(self.engine_root, rel_path)
        if not os.path.exists(full_path):
            return None
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not JSON, return raw text
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                return {'raw_content': f.read(), 'path': rel_path}

    def read_texture_file(self, rel_path: str) -> Optional[str]:
        """Return absolute path to a texture file if it exists."""
        full_path = os.path.join(self.engine_root, rel_path)
        if os.path.exists(full_path):
            return full_path
        return None

    # ============ Engine Interaction Helpers ============

    def write_command_file(self, command: str, params: dict = None):
        """
        Write a command file that the engine can detect (if file watching is set up).
        This is a passive signal mechanism - works if engine monitors this directory.
        """
        cmd_dir = os.path.join(self.engine_root, 'LocalData', 'optimizer_commands')
        os.makedirs(cmd_dir, exist_ok=True)
        cmd = {
            'command': command,
            'params': params or {},
        }
        cmd_path = os.path.join(cmd_dir, 'pending_command.json')
        with open(cmd_path, 'w', encoding='utf-8') as f:
            json.dump(cmd, f, ensure_ascii=False)

    def get_engine_info(self) -> dict:
        """Get basic engine project information."""
        info = {
            'engine_root': self.engine_root,
            'worlds_dir': self.worlds_dir,
            'repository_dir': self.repository_dir,
            'exists': os.path.isdir(self.engine_root),
            'renderdoc_available': self._renderdoc.is_available(),
        }

        # World count
        try:
            worlds = self.list_worlds()
            info['world_count'] = len(worlds)
        except Exception:
            info['world_count'] = 0

        # Current world info
        if self._current_world:
            info['current_world'] = self._resolver.get_world_stats(self._current_world)

        return info

    # ============ Unified Pipeline (Dual Approach) ============

    @property
    def pipeline(self) -> Optional[UnifiedPipeline]:
        return self._pipeline

    def get_resource_inventory(self, rdc_path: str = None) -> dict:
        """
        Build unified resource inventory for the current world.
        Combines repository metadata with optional RenderDoc visual data.
        """
        if not self._pipeline or not self._current_world:
            return {'error': 'No pipeline or world selected'}
        return self._pipeline.build_resource_inventory(self._current_world, rdc_path)

    def take_snapshot_before(self, rdc_path: str = None) -> dict:
        """
        Take a 'before optimization' snapshot using available sources.
        Repository data comes from the currently selected world.
        Visual data comes from the provided .rdc capture path.
        """
        if not self._pipeline:
            return {'error': 'Pipeline not initialized. Call start() first.'}

        self._snapshot_before = self._pipeline.take_snapshot(
            world=self._current_world,
            rdc_path=rdc_path,
            label='before',
        )
        return {
            'status': 'ok',
            'label': 'before',
            'source': self._snapshot_before.source.value,
            'resources': len(self._snapshot_before.resources),
            'total_size': self._snapshot_before.total_data_size,
            'framebuffer': self._snapshot_before.framebuffer_image or '',
        }

    def take_snapshot_after(self, rdc_path: str = None) -> dict:
        """
        Take an 'after optimization' snapshot.
        """
        if not self._pipeline:
            return {'error': 'Pipeline not initialized'}

        self._snapshot_after = self._pipeline.take_snapshot(
            world=self._current_world,
            rdc_path=rdc_path,
            label='after',
        )
        return {
            'status': 'ok',
            'label': 'after',
            'source': self._snapshot_after.source.value,
            'resources': len(self._snapshot_after.resources),
            'total_size': self._snapshot_after.total_data_size,
            'framebuffer': self._snapshot_after.framebuffer_image or '',
        }

    def run_comparison(self) -> Optional[dict]:
        """
        Compare before/after snapshots using all available data.
        Returns combined comparison metrics.
        """
        if not self._pipeline:
            return {'error': 'Pipeline not initialized'}
        if not self._snapshot_before or not self._snapshot_after:
            return {'error': 'Need both before and after snapshots'}

        result = self._pipeline.compare_snapshots(
            self._snapshot_before, self._snapshot_after
        )
        return result.to_dict()

    def run_full_ab_comparison(
        self,
        rdc_before: str = None,
        rdc_after: str = None,
    ) -> Optional[dict]:
        """
        Run a complete A/B comparison workflow.

        Uses the current world for repository data + optional .rdc captures.
        Supports:
          - Repository only (no rdc paths)
          - RenderDoc only (no world selected, both rdc paths given)
          - Combined (world + rdc paths)
        """
        if not self._pipeline:
            return {'error': 'Pipeline not initialized'}

        result = self._pipeline.run_ab_comparison(
            world_before=self._current_world,
            world_after=self._current_world,
            rdc_before=rdc_before,
            rdc_after=rdc_after,
        )
        return result.to_dict()

    def extract_from_rdc(self, rdc_path: str) -> dict:
        """
        Extract framebuffer and textures from a RenderDoc capture file.
        """
        output_dir = os.path.join(
            self.engine_root, 'LocalData', 'rdoc_extractions',
            os.path.splitext(os.path.basename(rdc_path))[0]
        )
        result = self._extractor.extract_all(rdc_path, output_dir)
        return {
            'success': result.success,
            'framebuffer': result.framebuffer_path,
            'textures': len(result.textures),
            'output_dir': result.output_dir,
            'time': result.extraction_time,
        }

    def list_rdc_captures(self, capture_dir: str = None) -> List[dict]:
        """List available .rdc capture files."""
        if capture_dir is None:
            capture_dir = os.path.join(self.engine_root, 'LocalData', 'rdoc_comparison')
        return self._extractor.list_captures(capture_dir)
