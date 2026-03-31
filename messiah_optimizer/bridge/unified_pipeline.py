"""
Unified resource pipeline combining two approaches:

  1. Repository-based: iworld → ilevel → repository XML chain → resource files on disk
     - Provides metadata (GUID, type, name, package), data file paths, file sizes
     - Good for precise resource targeting, batch processing, size analysis

  2. RenderDoc-based: Frame capture (.rdc) → framebuffer extraction → visual comparison
     - Provides actual rendered output as ground truth
     - Good for visual quality validation (PSNR/MSE), A/B comparison

Combined mode uses repository for resource identification and RenderDoc for
visual quality verification, giving both data-level and pixel-level comparison.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .resource_resolver import ResourceInfo, ResourceResolver, WorldInfo
from .renderdoc_capture import RenderDocCapture, CaptureWorkflow

logger = logging.getLogger(__name__)


class ResourceSource(Enum):
    """How a resource was obtained."""
    REPOSITORY = "repository"
    RENDERDOC = "renderdoc"
    COMBINED = "combined"


@dataclass
class UnifiedResource:
    """A resource entry that can come from either or both sources."""
    guid: str = ''
    name: str = ''
    type: str = ''
    source: ResourceSource = ResourceSource.REPOSITORY

    # Repository data
    repo_info: Optional[ResourceInfo] = None
    data_path: Optional[str] = None
    data_size: int = 0
    data_md5: str = ''

    # RenderDoc data (framebuffer / extracted image associated with this resource)
    rdc_image_path: Optional[str] = None

    # Cross-reference
    matched: bool = False  # Found in both sources


@dataclass
class ComparisonResult:
    """Result of comparing before/after states using one or both approaches."""
    method: str = ''  # "visual", "resource", "combined"
    timestamp: float = 0.0

    # Visual comparison (RenderDoc framebuffers)
    visual_metrics: Optional[dict] = None  # mse, psnr, max_diff, etc.
    before_image: str = ''
    after_image: str = ''
    diff_image: str = ''

    # Resource-level comparison
    resources_before: int = 0
    resources_after: int = 0
    total_size_before: int = 0
    total_size_after: int = 0
    size_delta: int = 0
    changed_resources: List[dict] = field(default_factory=list)
    added_resources: List[str] = field(default_factory=list)
    removed_resources: List[str] = field(default_factory=list)

    # Combined quality score
    quality_score: float = 1.0  # 0..1, higher is better (1 = no quality loss)
    size_reduction_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            'method': self.method,
            'timestamp': self.timestamp,
            'visual': self.visual_metrics,
            'before_image': self.before_image,
            'after_image': self.after_image,
            'diff_image': self.diff_image,
            'resources_before': self.resources_before,
            'resources_after': self.resources_after,
            'total_size_before': self.total_size_before,
            'total_size_after': self.total_size_after,
            'size_delta': self.size_delta,
            'size_reduction_pct': self.size_reduction_pct,
            'changed_count': len(self.changed_resources),
            'added_count': len(self.added_resources),
            'removed_count': len(self.removed_resources),
            'quality_score': self.quality_score,
        }


class UnifiedPipeline:
    """
    Combines repository-based and RenderDoc-based resource workflows.

    Usage:
        pipeline = UnifiedPipeline(resolver, renderdoc, output_dir)

        # Approach 1: Repository only
        resources = pipeline.load_from_repository(world)

        # Approach 2: RenderDoc only
        visual = pipeline.capture_visual(exe_path)

        # Combined: load resources + capture visual, then compare
        snapshot_before = pipeline.take_snapshot(world, rdc_before)
        # ... run optimization ...
        snapshot_after = pipeline.take_snapshot(world, rdc_after)
        result = pipeline.compare_snapshots(snapshot_before, snapshot_after)
    """

    def __init__(
        self,
        resolver: ResourceResolver,
        renderdoc: RenderDocCapture,
        output_dir: str,
    ):
        self.resolver = resolver
        self.renderdoc = renderdoc
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # Approach 1: Repository-based resource loading
    # ================================================================

    def load_from_repository(self, world: WorldInfo) -> List[UnifiedResource]:
        """
        Load resources from the repository for a resolved world.
        Catalogs data file paths, sizes, and MD5 hashes.
        """
        resources = []
        for guid, info in world.resources.items():
            ur = UnifiedResource(
                guid=guid,
                name=info.name,
                type=info.type,
                source=ResourceSource.REPOSITORY,
                repo_info=info,
                data_md5=info.md5 or '',
            )

            # Locate actual data file on disk
            data_path = self.resolver.get_resource_data_path(info)
            if data_path and os.path.exists(data_path):
                ur.data_path = data_path
                ur.data_size = os.path.getsize(data_path)

            resources.append(ur)

        logger.info(f"Repository: loaded {len(resources)} resources from world '{world.name}', "
                     f"{sum(1 for r in resources if r.data_path)} have data files")
        return resources

    def load_textures_from_repository(self, world: WorldInfo) -> List[UnifiedResource]:
        """Load only texture resources from repository."""
        textures = self.resolver.get_textures(world)
        resources = []
        for info in textures:
            ur = UnifiedResource(
                guid=info.guid,
                name=info.name,
                type=info.type,
                source=ResourceSource.REPOSITORY,
                repo_info=info,
                data_md5=info.md5 or '',
            )
            data_path = self.resolver.get_resource_data_path(info)
            if data_path and os.path.exists(data_path):
                ur.data_path = data_path
                ur.data_size = os.path.getsize(data_path)
            resources.append(ur)
        return resources

    # ================================================================
    # Approach 2: RenderDoc-based visual capture
    # ================================================================

    def capture_framebuffer(
        self,
        rdc_path: str,
        label: str = '',
    ) -> Optional[str]:
        """
        Extract framebuffer image from an existing .rdc capture.

        Args:
            rdc_path: Path to .rdc file (from an external capture)
            label: Optional label for the output image filename

        Returns:
            Path to extracted image, or None on failure.
        """
        if not os.path.exists(rdc_path):
            logger.error(f".rdc file not found: {rdc_path}")
            return None

        suffix = f'_{label}' if label else ''
        out_path = os.path.join(
            self.output_dir,
            f'framebuffer{suffix}_{int(time.time())}.png'
        )
        return self.renderdoc.extract_framebuffer(rdc_path, out_path)

    def compare_visual(
        self,
        before_image: str,
        after_image: str,
    ) -> ComparisonResult:
        """
        Compare two framebuffer images (before/after) using pixel metrics.
        Pure RenderDoc-based visual quality assessment.
        """
        result = ComparisonResult(
            method='visual',
            timestamp=time.time(),
            before_image=before_image,
            after_image=after_image,
        )

        metrics = self.renderdoc.compare_captures(before_image, after_image)
        if metrics:
            result.visual_metrics = metrics
            result.diff_image = metrics.get('diff_image', '')

            # Quality score from PSNR: >40dB is excellent, 30-40 good, <30 noticeable
            psnr = metrics.get('psnr', float('inf'))
            if psnr == float('inf'):
                result.quality_score = 1.0
            elif psnr >= 50:
                result.quality_score = 1.0
            elif psnr >= 40:
                result.quality_score = 0.95 + 0.05 * (psnr - 40) / 10
            elif psnr >= 30:
                result.quality_score = 0.80 + 0.15 * (psnr - 30) / 10
            else:
                result.quality_score = max(0.0, 0.80 * psnr / 30)

        return result

    # ================================================================
    # Snapshots: capture state from either/both sources
    # ================================================================

    @dataclass
    class Snapshot:
        """A point-in-time snapshot of resources and/or visual state."""
        label: str = ''
        timestamp: float = 0.0
        resources: List['UnifiedResource'] = field(default_factory=list)
        framebuffer_image: Optional[str] = None
        world_name: str = ''
        source: ResourceSource = ResourceSource.COMBINED
        total_data_size: int = 0

    def take_snapshot(
        self,
        world: Optional[WorldInfo] = None,
        rdc_path: Optional[str] = None,
        label: str = '',
    ) -> 'UnifiedPipeline.Snapshot':
        """
        Take a snapshot of the current state using available sources.

        Args:
            world: Resolved WorldInfo (repository approach, optional)
            rdc_path: Path to .rdc capture file (RenderDoc approach, optional)
            label: Human-readable label for this snapshot (e.g. 'before', 'after')

        Returns:
            Snapshot containing resources and/or visual data
        """
        snap = UnifiedPipeline.Snapshot(
            label=label,
            timestamp=time.time(),
        )

        if world and rdc_path:
            snap.source = ResourceSource.COMBINED
        elif world:
            snap.source = ResourceSource.REPOSITORY
        elif rdc_path:
            snap.source = ResourceSource.RENDERDOC
        else:
            logger.warning("take_snapshot called with no world and no rdc_path")
            return snap

        # Repository resources
        if world:
            snap.resources = self.load_from_repository(world)
            snap.world_name = world.name
            snap.total_data_size = sum(r.data_size for r in snap.resources)

        # RenderDoc framebuffer
        if rdc_path:
            img = self.capture_framebuffer(rdc_path, label=label)
            if img:
                snap.framebuffer_image = img

        return snap

    # ================================================================
    # Comparison: before/after using one or both approaches
    # ================================================================

    def compare_snapshots(
        self,
        before: 'UnifiedPipeline.Snapshot',
        after: 'UnifiedPipeline.Snapshot',
    ) -> ComparisonResult:
        """
        Compare two snapshots to evaluate optimization impact.
        Automatically uses whatever data is available in both snapshots.
        """
        has_visual = bool(before.framebuffer_image and after.framebuffer_image)
        has_resources = bool(before.resources and after.resources)

        if has_visual and has_resources:
            method = 'combined'
        elif has_visual:
            method = 'visual'
        elif has_resources:
            method = 'resource'
        else:
            return ComparisonResult(method='none', timestamp=time.time())

        result = ComparisonResult(method=method, timestamp=time.time())

        # --- Visual comparison ---
        if has_visual:
            diff_path = os.path.join(
                self.output_dir,
                f'diff_{before.label}_{after.label}_{int(time.time())}.png'
            )
            metrics = self.renderdoc.compare_captures(
                before.framebuffer_image,
                after.framebuffer_image,
                output_path=diff_path,
            )
            if metrics:
                result.visual_metrics = metrics
                result.before_image = before.framebuffer_image
                result.after_image = after.framebuffer_image
                result.diff_image = metrics.get('diff_image', diff_path)

        # --- Resource-level comparison ---
        if has_resources:
            self._compare_resources(before, after, result)

        # --- Combined quality score ---
        result.quality_score = self._compute_quality_score(result)

        # Persist report
        self._save_report(result, before.label, after.label)

        return result

    def _compare_resources(
        self,
        before: 'UnifiedPipeline.Snapshot',
        after: 'UnifiedPipeline.Snapshot',
        result: ComparisonResult,
    ):
        """Compute resource-level differences between two snapshots."""
        before_map: Dict[str, UnifiedResource] = {r.guid: r for r in before.resources}
        after_map: Dict[str, UnifiedResource] = {r.guid: r for r in after.resources}

        result.resources_before = len(before_map)
        result.resources_after = len(after_map)
        result.total_size_before = before.total_data_size
        result.total_size_after = after.total_data_size
        result.size_delta = after.total_data_size - before.total_data_size

        if before.total_data_size > 0:
            result.size_reduction_pct = (
                (before.total_data_size - after.total_data_size) /
                before.total_data_size * 100
            )

        # Find changes
        all_guids = set(before_map.keys()) | set(after_map.keys())
        for guid in all_guids:
            b = before_map.get(guid)
            a = after_map.get(guid)

            if b and not a:
                result.removed_resources.append(guid)
            elif a and not b:
                result.added_resources.append(guid)
            elif b and a:
                # Check if data changed
                changed = False
                change = {'guid': guid, 'name': b.name, 'type': b.type}
                if b.data_size != a.data_size:
                    change['size_before'] = b.data_size
                    change['size_after'] = a.data_size
                    change['size_delta'] = a.data_size - b.data_size
                    changed = True
                if b.data_md5 and a.data_md5 and b.data_md5 != a.data_md5:
                    change['md5_changed'] = True
                    changed = True
                if changed:
                    result.changed_resources.append(change)

    def _compute_quality_score(self, result: ComparisonResult) -> float:
        """
        Compute a combined quality score (0..1) from visual and resource metrics.
        1.0 = no quality loss, 0.0 = severe degradation.
        """
        scores = []

        # Visual quality score from PSNR
        if result.visual_metrics:
            psnr = result.visual_metrics.get('psnr', float('inf'))
            if psnr == float('inf') or psnr >= 50:
                scores.append(1.0)
            elif psnr >= 40:
                scores.append(0.95 + 0.05 * (psnr - 40) / 10)
            elif psnr >= 30:
                scores.append(0.80 + 0.15 * (psnr - 30) / 10)
            else:
                scores.append(max(0.0, 0.80 * psnr / 30))

        # Resource integrity score (penalize unexpected changes)
        if result.resources_before > 0:
            removed_ratio = len(result.removed_resources) / result.resources_before
            resource_score = 1.0 - removed_ratio
            scores.append(max(0.0, resource_score))

        if not scores:
            return 1.0
        return sum(scores) / len(scores)

    def _save_report(self, result: ComparisonResult, label_before: str, label_after: str):
        """Save comparison report as JSON."""
        report_path = os.path.join(
            self.output_dir,
            f'comparison_{label_before}_vs_{label_after}_{int(result.timestamp)}.json'
        )
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Comparison report saved: {report_path}")
        except Exception as e:
            logger.warning(f"Failed to save report: {e}")

    # ================================================================
    # Convenience: full A/B comparison workflow
    # ================================================================

    def run_ab_comparison(
        self,
        world_before: Optional[WorldInfo],
        world_after: Optional[WorldInfo],
        rdc_before: Optional[str] = None,
        rdc_after: Optional[str] = None,
    ) -> ComparisonResult:
        """
        Run a complete A/B comparison workflow.

        Supports all combinations:
          - Both worlds + both RDC captures → full combined comparison
          - Only worlds → resource-level comparison only
          - Only RDC captures → visual comparison only

        Args:
            world_before: World state before optimization (or None)
            world_after: World state after optimization (or None)
            rdc_before: .rdc capture before optimization (or None)
            rdc_after: .rdc capture after optimization (or None)

        Returns:
            ComparisonResult with all available metrics
        """
        logger.info("Running A/B comparison...")

        snap_before = self.take_snapshot(
            world=world_before, rdc_path=rdc_before, label='before')
        snap_after = self.take_snapshot(
            world=world_after, rdc_path=rdc_after, label='after')

        result = self.compare_snapshots(snap_before, snap_after)

        logger.info(f"A/B comparison complete: method={result.method}, "
                     f"quality={result.quality_score:.3f}, "
                     f"size_delta={result.size_delta:+d} bytes")
        return result

    # ================================================================
    # Resource inventory: unified view across sources
    # ================================================================

    def build_resource_inventory(
        self,
        world: WorldInfo,
        rdc_path: Optional[str] = None,
    ) -> dict:
        """
        Build a unified resource inventory combining repository metadata
        with optional RenderDoc visual data.

        Returns a summary dict suitable for UI display.
        """
        resources = self.load_from_repository(world)

        # Aggregate stats
        by_type: Dict[str, dict] = {}
        total_size = 0
        files_found = 0

        for r in resources:
            t = r.type or 'Unknown'
            if t not in by_type:
                by_type[t] = {'count': 0, 'total_size': 0, 'with_data': 0}
            by_type[t]['count'] += 1
            by_type[t]['total_size'] += r.data_size
            if r.data_path:
                by_type[t]['with_data'] += 1
                files_found += 1
            total_size += r.data_size

        inventory = {
            'world_name': world.name,
            'total_resources': len(resources),
            'files_found': files_found,
            'total_data_size': total_size,
            'total_data_size_mb': round(total_size / (1024 * 1024), 2),
            'by_type': by_type,
            'source': ResourceSource.COMBINED.value if rdc_path else ResourceSource.REPOSITORY.value,
        }

        # Add RenderDoc visual if available
        if rdc_path and self.renderdoc.is_available():
            fb_image = self.capture_framebuffer(rdc_path, label='inventory')
            if fb_image:
                inventory['framebuffer_image'] = fb_image
                inventory['renderdoc_available'] = True
            else:
                inventory['renderdoc_available'] = False
        else:
            inventory['renderdoc_available'] = self.renderdoc.is_available()

        return inventory
