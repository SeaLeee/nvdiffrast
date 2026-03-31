"""
Messiah Resource Resolver — parse iworld/ilevel files and resolve resource GUIDs
via the Repository index, without scanning the entire asset tree.

Architecture:
  1. User selects an .iworld file
  2. We parse it → extract Level names
  3. For each Level, parse the corresponding .ilevel file → collect resource GUIDs
  4. Build/cache the repository index (GUID → Type/Name/Package/SourcePath)
  5. Resolve GUIDs → get only the resources this world actually uses

This is the targeted alternative to scanning Content/Shaders directories.
"""

import hashlib
import os
import json
import logging
import pickle
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class ResourceInfo:
    """A single resource entry from a repository."""
    guid: str
    type: str           # Texture, Mesh, Material, Effect, etc.
    name: str           # Human-readable name
    package: str        # Package path within repo (e.g. Env/Cubes/hex_world_W12K)
    res_class: str      # Class (Texture2D, StaticMesh, etc.)
    source_path: str    # Original source file path
    repo_name: str      # Which .local repo this belongs to
    flags: int = 0
    md5: str = ''
    deps: List[str] = field(default_factory=list)  # Dependency GUIDs (ordered)


@dataclass
class WorldInfo:
    """Parsed world structure."""
    name: str
    iworld_path: str
    levels: List[str] = field(default_factory=list)
    resource_guids: Set[str] = field(default_factory=set)
    resources: Dict[str, ResourceInfo] = field(default_factory=dict)
    # All resources including dependencies (mesh, material, texture via chain)
    all_resources: Dict[str, ResourceInfo] = field(default_factory=dict)


class ResourceResolver:
    """
    Parses Messiah iworld/ilevel XML files and resolves resource GUIDs
    through the Repository index.
    """

    def __init__(self, worlds_dir: str, repository_dir: str, cache_dir: str = None):
        """
        Args:
            worlds_dir: Path to Worlds directory (contains .iworld/.ilevel files)
                        e.g. I:\\trunk_bjs\\common\\resource\\Package\\Worlds
            repository_dir: Path to Repository directory (contains *.local/ repos)
                        e.g. I:\\trunk_bjs\\common\\resource\\Package\\Repository
            cache_dir: Directory for caching the repo index. Defaults to
                       <repository_dir>/../.resolver_cache
        """
        self.worlds_dir = os.path.normpath(worlds_dir)
        self.repository_dir = os.path.normpath(repository_dir)
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(self.repository_dir), '.resolver_cache')

        # Cache: GUID → ResourceInfo
        self._repo_index: Dict[str, ResourceInfo] = {}
        self._index_built = False

    # ============ World Listing ============

    def list_worlds(self) -> List[dict]:
        """List all available .iworld files."""
        worlds = []
        if not os.path.isdir(self.worlds_dir):
            logger.warning(f"Worlds directory not found: {self.worlds_dir}")
            return worlds

        for fname in os.listdir(self.worlds_dir):
            if fname.endswith('.iworld'):
                worlds.append({
                    'name': fname.replace('.iworld', ''),
                    'filename': fname,
                    'path': os.path.join(self.worlds_dir, fname),
                })
        worlds.sort(key=lambda w: w['name'])
        return worlds

    # ============ World Parsing ============

    def parse_world(self, iworld_path: str) -> WorldInfo:
        """
        Parse an .iworld file and all its .ilevel files.
        Collects all resource GUIDs used by this world.
        """
        world_name = os.path.splitext(os.path.basename(iworld_path))[0]
        world = WorldInfo(name=world_name, iworld_path=iworld_path)

        # Parse iworld XML to get level list
        try:
            tree = ET.parse(iworld_path)
            root = tree.getroot()
            for level_elem in root.iter('Level'):
                level_name = level_elem.get('Name', '')
                if level_name:
                    world.levels.append(level_name)
        except ET.ParseError as e:
            logger.error(f"Failed to parse iworld {iworld_path}: {e}")
            return world

        logger.info(f"World '{world_name}' has {len(world.levels)} levels")

        # Parse each corresponding .ilevel file
        worlds_dir = os.path.dirname(iworld_path)
        for level_name in world.levels:
            ilevel_fname = f"{world_name}@{level_name}.ilevel"
            ilevel_path = os.path.join(worlds_dir, ilevel_fname)
            if os.path.exists(ilevel_path):
                guids = self._parse_ilevel(ilevel_path)
                world.resource_guids.update(guids)
            else:
                logger.debug(f"ilevel not found: {ilevel_fname}")

        logger.info(f"World '{world_name}': {len(world.resource_guids)} unique resource GUIDs")
        return world

    def _parse_ilevel(self, ilevel_path: str) -> Set[str]:
        """Parse an .ilevel file and extract all resource GUIDs."""
        guids = set()
        try:
            tree = ET.parse(ilevel_path)
            root = tree.getroot()
            # Scan ALL elements for GUID-like text content
            # (GUIDs appear in many tags: Resource, DensityMap, ShortResource,
            #  Card0Resource, Proxy0Resource, OverrideMaterial, Lightmap, etc.)
            for elem in root.iter():
                text = (elem.text or '').strip()
                if self._is_valid_guid(text):
                    guids.add(text)
        except ET.ParseError as e:
            logger.warning(f"Failed to parse ilevel {ilevel_path}: {e}")
        except Exception as e:
            logger.warning(f"Error reading {ilevel_path}: {e}")
        return guids

    @staticmethod
    def _is_valid_guid(text: str) -> bool:
        """Check if text looks like a valid GUID."""
        if not text or len(text) != 36:
            return False
        # Must not be all zeros
        if text == '00000000-0000-0000-0000-000000000000':
            return False
        parts = text.split('-')
        if len(parts) != 5:
            return False
        try:
            for part in parts:
                int(part, 16)
            return True
        except ValueError:
            return False

    # ============ Repository Index (with disk cache) ============

    def _cache_path(self) -> str:
        """Return path to the repo index cache file."""
        key = hashlib.md5(self.repository_dir.encode()).hexdigest()[:12]
        return os.path.join(self.cache_dir, f'repo_index_{key}.pkl')

    def _load_cache(self) -> bool:
        """Try to load the repo index from disk cache (pickle for speed)."""
        cp = self._cache_path()
        if not os.path.exists(cp):
            return False
        try:
            with open(cp, 'rb') as f:
                self._repo_index = pickle.load(f)
            self._index_built = True
            logger.info(f"Loaded repo index from cache: {len(self._repo_index)} entries")
            return True
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return False

    def _save_cache(self):
        """Save the repo index to disk cache (pickle for speed)."""
        os.makedirs(self.cache_dir, exist_ok=True)
        cp = self._cache_path()
        try:
            with open(cp, 'wb') as f:
                pickle.dump(self._repo_index, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved repo index cache: {len(self._repo_index)} entries -> {cp}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def build_repo_index(self, repo_names: List[str] = None, force: bool = False):
        """
        Build the resource index from repository files.
        Uses disk cache for subsequent runs (first build may take minutes).

        Args:
            repo_names: Optional list of specific .local repos to index.
                       If None, indexes all repos.
            force: If True, ignore cache and rebuild from scratch.
        """
        if not force and not self._index_built and self._load_cache():
            return  # Loaded from cache

        if not os.path.isdir(self.repository_dir):
            logger.error(f"Repository directory not found: {self.repository_dir}")
            return

        if repo_names is None:
            # Index all .local directories
            repo_names = [d for d in os.listdir(self.repository_dir)
                          if d.endswith('.local') and
                          os.path.isdir(os.path.join(self.repository_dir, d))]

        total = 0
        for i, repo_name in enumerate(repo_names):
            repo_path = os.path.join(self.repository_dir, repo_name)
            repo_file = os.path.join(repo_path, 'resource.repository')
            if not os.path.exists(repo_file):
                continue
            count = self._parse_repository_file(repo_file, repo_name)
            total += count
            if (i + 1) % 50 == 0:
                logger.info(f"  Indexed {i+1}/{len(repo_names)} repos ({total} resources)...")

        self._index_built = True
        self._save_cache()
        logger.info(f"Repository index built: {total} resources from {len(repo_names)} repos")

    def build_index_for_guids(self, guids: Set[str]):
        """
        Build repository index, load from cache if available.
        If no cache, does a full build (which then caches for next time).
        """
        # Try cache first
        if not self._index_built and self._load_cache():
            pass  # Loaded from cache

        remaining = guids - set(self._repo_index.keys())
        if not remaining:
            return  # All already indexed

        if self._index_built:
            # Already have a full index but some GUIDs not found — that's OK
            # (resources may have been deleted or belong to a different branch)
            logger.info(f"{len(remaining)} GUIDs not found in index (may not exist)")
            return

        if not os.path.isdir(self.repository_dir):
            return

        # No index at all — do full build (then cache for next time)
        logger.info(f"No index available, building full index for {len(guids)} GUIDs...")
        self.build_repo_index(force=True)

    def _parse_repository_file(self, repo_file: str, repo_name: str) -> int:
        """Parse a resource.repository XML and add entries to index."""
        count = 0
        try:
            tree = ET.parse(repo_file)
            root = tree.getroot()
            for item in root.iter('Item'):
                guid_elem = item.find('GUID')
                if guid_elem is None:
                    continue
                guid = (guid_elem.text or '').strip()
                if not guid:
                    continue

                info = ResourceInfo(
                    guid=guid,
                    type=self._get_text(item, 'Type'),
                    name=self._get_text(item, 'Name'),
                    package=self._get_text(item, 'Package'),
                    res_class=self._get_text(item, 'Class'),
                    source_path='',
                    repo_name=repo_name,
                    flags=int(self._get_text(item, 'Flags') or '0'),
                    deps=[(d.text or '').strip() for d in item.findall('Deps') if d.text and d.text.strip()],
                )
                # Get annotation details
                annotation = item.find('Annotation')
                if annotation is not None:
                    info.source_path = self._get_text(annotation, 'SourcePath')
                    info.md5 = self._get_text(annotation, 'MD5')

                self._repo_index[guid] = info
                count += 1
        except ET.ParseError as e:
            logger.warning(f"Failed to parse {repo_file}: {e}")
        except Exception as e:
            logger.warning(f"Error reading {repo_file}: {e}")
        return count

    @staticmethod
    def _get_text(elem, tag: str) -> str:
        """Get text content of a child element, or empty string."""
        child = elem.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return ''

    # ============ Resolution ============

    def resolve_world(self, world: WorldInfo) -> WorldInfo:
        """
        Resolve all resource GUIDs in a world to actual ResourceInfo objects.
        Builds index on-demand if needed.
        """
        if not self._index_built:
            # Build targeted index for just the GUIDs we need
            self.build_index_for_guids(world.resource_guids)

        resolved = 0
        for guid in world.resource_guids:
            if guid in self._repo_index:
                world.resources[guid] = self._repo_index[guid]
                resolved += 1

        logger.info(f"Resolved {resolved}/{len(world.resource_guids)} resources for '{world.name}'")

        # Traverse dependency chain to discover meshes, materials, textures etc.
        self.resolve_dependencies(world)

        return world

    def resolve_dependencies(self, world: WorldInfo, max_depth: int = 4):
        """
        Recursively follow dependency chains to discover sub-resources.

        The ilevel only directly references top-level objects (LodModel, Data, Prefab).
        The actual Mesh, Material, Texture resources are nested:
          LodModel → Model (via Deps) → Mesh + Material + Skin (via Deps) → Texture (via Deps)

        Populates world.all_resources with the complete flattened set.
        """
        # Start with directly resolved resources
        world.all_resources = dict(world.resources)
        visited = set(world.all_resources.keys())
        frontier = set()

        # Collect all deps from resolved resources
        for info in world.resources.values():
            for dep_guid in info.deps:
                if dep_guid and dep_guid not in visited:
                    frontier.add(dep_guid)

        depth = 0
        while frontier and depth < max_depth:
            depth += 1
            new_frontier = set()
            resolved_this_level = 0

            for guid in frontier:
                if guid in visited:
                    continue
                visited.add(guid)

                info = self._repo_index.get(guid)
                if info:
                    world.all_resources[guid] = info
                    resolved_this_level += 1
                    # Follow this resource's deps too
                    for dep_guid in info.deps:
                        if dep_guid and dep_guid not in visited:
                            new_frontier.add(dep_guid)

            logger.info(f"  Dep chain depth {depth}: resolved {resolved_this_level} "
                         f"from {len(frontier)} candidates ({len(new_frontier)} new deps)")
            frontier = new_frontier

        total_deps = len(world.all_resources) - len(world.resources)
        logger.info(f"Dependency resolution: {total_deps} additional resources "
                     f"discovered ({len(world.all_resources)} total)")

    def get_resource(self, guid: str) -> Optional[ResourceInfo]:
        """Look up a single resource by GUID."""
        return self._repo_index.get(guid)

    # ============ Resource Filtering ============

    def get_resources_by_type(self, world: WorldInfo, res_type: str) -> List[ResourceInfo]:
        """Get all resources of a given type from a resolved world (including deps)."""
        return [r for r in world.all_resources.values()
                if r.type.lower() == res_type.lower()]

    def get_textures(self, world: WorldInfo) -> List[ResourceInfo]:
        return self.get_resources_by_type(world, 'Texture')

    def get_meshes(self, world: WorldInfo) -> List[ResourceInfo]:
        return [r for r in world.all_resources.values()
                if r.type.lower() in ('mesh', 'staticmesh', 'skeletalmesh')]

    def get_materials(self, world: WorldInfo) -> List[ResourceInfo]:
        return self.get_resources_by_type(world, 'Material')

    def get_effects(self, world: WorldInfo) -> List[ResourceInfo]:
        return self.get_resources_by_type(world, 'Effect')

    # ============ Resource Stats ============

    def get_world_stats(self, world: WorldInfo) -> dict:
        """Get statistics about a resolved world's resources."""
        # Direct references from ilevel
        direct_type_counts = {}
        for r in world.resources.values():
            t = r.type or 'Unknown'
            direct_type_counts[t] = direct_type_counts.get(t, 0) + 1

        # All resources including deps
        all_type_counts = {}
        for r in world.all_resources.values():
            t = r.type or 'Unknown'
            all_type_counts[t] = all_type_counts.get(t, 0) + 1

        return {
            'world_name': world.name,
            'total_levels': len(world.levels),
            'total_guids': len(world.resource_guids),
            'resolved': len(world.resources),
            'unresolved': len(world.resource_guids) - len(world.resources),
            'by_type': direct_type_counts,
            'all_resources': len(world.all_resources),
            'all_by_type': all_type_counts,
        }

    # ============ Resource Data Path ============

    def get_resource_data_path(self, info: ResourceInfo) -> Optional[str]:
        """
        Get the path to the actual resource data file in the repository.

        Repository layout (hex-sharded):
            <repo_dir>/<repo_name>/<Type>/<GUID[0:2]>/{<GUID>}/resource.xml
            <repo_dir>/<repo_name>/<Type>/<GUID[0:2]>/{<GUID>}/resource.data
            <repo_dir>/<repo_name>/<Type>/<GUID[0:2]>/{<GUID>}/source.tga (textures)
            <repo_dir>/<repo_name>/<Type>/<GUID[0:2]>/{<GUID>}/texture.xml (textures)
        """
        repo_path = os.path.join(self.repository_dir, info.repo_name)

        # Primary structure: repo/<Type>/<GUID[0:2]>/{<GUID>}/*
        if info.type and len(info.guid) >= 2:
            hex_prefix = info.guid[:2]
            guid_dir = os.path.join(repo_path, info.type, hex_prefix,
                                    '{' + info.guid + '}')
            if os.path.isdir(guid_dir):
                # Priority: resource.data > resource.xml > texture.xml > source.*
                for fname in ('resource.data', 'resource.xml', 'texture.xml'):
                    fpath = os.path.join(guid_dir, fname)
                    if os.path.exists(fpath):
                        return fpath
                # Fallback: any data file
                for fname in os.listdir(guid_dir):
                    fpath = os.path.join(guid_dir, fname)
                    if os.path.isfile(fpath):
                        return fpath

        # Fallback: try Class name as directory (e.g. Texture2D)
        if info.res_class and info.res_class != info.type and len(info.guid) >= 2:
            hex_prefix = info.guid[:2]
            guid_dir = os.path.join(repo_path, info.res_class, hex_prefix,
                                    '{' + info.guid + '}')
            if os.path.isdir(guid_dir):
                for fname in ('resource.data', 'resource.xml', 'texture.xml'):
                    fpath = os.path.join(guid_dir, fname)
                    if os.path.exists(fpath):
                        return fpath

        return None

    def get_resource_dir(self, info: ResourceInfo) -> Optional[str]:
        """
        Get the directory containing all files for a resource.
        Returns: <repo>/<Type>/<GUID[0:2]>/{<GUID>}/ or None.
        """
        repo_path = os.path.join(self.repository_dir, info.repo_name)
        if info.type and len(info.guid) >= 2:
            guid_dir = os.path.join(repo_path, info.type, info.guid[:2],
                                    '{' + info.guid + '}')
            if os.path.isdir(guid_dir):
                return guid_dir
        if info.res_class and info.res_class != info.type and len(info.guid) >= 2:
            guid_dir = os.path.join(repo_path, info.res_class, info.guid[:2],
                                    '{' + info.guid + '}')
            if os.path.isdir(guid_dir):
                return guid_dir
        return None

    def get_resource_files(self, info: ResourceInfo) -> List[str]:
        """
        Get all files associated with a resource GUID in the repository.
        Returns list of file paths (may include resource.data, resource.xml, source.tga etc.)
        """
        res_dir = self.get_resource_dir(info)
        if not res_dir:
            return []
        return [os.path.join(res_dir, f) for f in os.listdir(res_dir)
                if os.path.isfile(os.path.join(res_dir, f))]

    def get_resource_total_size(self, info: ResourceInfo) -> int:
        """Get total size in bytes of all files for a resource."""
        return sum(os.path.getsize(f) for f in self.get_resource_files(info))

    def load_resource_data(self, info: ResourceInfo) -> Optional[bytes]:
        """
        Load the raw binary data of a resource from repository.
        Returns the contents of resource.data file, or None if not found.
        """
        data_path = self.get_resource_data_path(info)
        if data_path and os.path.exists(data_path):
            try:
                with open(data_path, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to read resource data {data_path}: {e}")
        return None
