"""
Material I/O - Load and save material parameter files.
Supports JSON and Messiah XML formats.
"""

import os
import json
from typing import Optional


def load_material(path: str) -> dict:
    """
    Load material parameters from file.

    Args:
        path: .json or .xml file path

    Returns:
        dict with material parameters
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.json':
        return _load_json(path)
    elif ext == '.xml':
        return _load_xml(path)
    else:
        raise ValueError(f"Unsupported material format: {ext}")


def save_material(data: dict, path: str, fmt: str = None):
    """
    Save material parameters to file.

    Args:
        data: Material parameter dict
        path: Output path
        fmt:  Force format ('json', 'xml', or None for auto-detect)
    """
    if fmt is None:
        fmt = os.path.splitext(path)[1].lstrip('.').lower()

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    if fmt == 'json':
        _save_json(data, path)
    elif fmt == 'xml':
        _save_messiah_xml(data, path)
    else:
        _save_json(data, path)


def _load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_json(data: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_xml(path: str) -> dict:
    """Parse Messiah material XML."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()

    result = {}

    # Shading model
    sm = root.find('ShadingModel')
    if sm is not None:
        result['shading_model'] = sm.text

    # Parameters
    params = root.find('Parameters')
    if params is not None:
        for child in params:
            try:
                result[child.tag.lower()] = float(child.text)
            except (ValueError, TypeError):
                result[child.tag.lower()] = child.text

    # Textures
    textures = root.find('Textures')
    if textures is not None:
        result['textures'] = {}
        for child in textures:
            result['textures'][child.tag.lower()] = child.text

    return result


def _save_messiah_xml(data: dict, path: str):
    """Save as Messiah-compatible material XML."""
    lines = ['<?xml version="1.0" encoding="utf-8"?>', '<Material>']

    # Shading model
    shading_model = data.get('shading_model', 'DefaultLit')
    lines.append(f'    <ShadingModel>{shading_model}</ShadingModel>')

    # Parameters
    param_keys = ['roughness', 'metallic', 'emission_strength',
                  'specular', 'opacity']
    has_params = any(k in data for k in param_keys)

    if has_params:
        lines.append('    <Parameters>')
        for key in param_keys:
            if key in data:
                val = data[key]
                tag = key.title().replace('_', '')
                if isinstance(val, float):
                    lines.append(f'        <{tag}>{val:.6f}</{tag}>')
                else:
                    lines.append(f'        <{tag}>{val}</{tag}>')
        lines.append('    </Parameters>')

    # Textures
    if 'textures' in data:
        lines.append('    <Textures>')
        for name, tex_path in data['textures'].items():
            tag = name.title().replace('_', '')
            lines.append(f'        <{tag}>{tex_path}</{tag}>')
        lines.append('    </Textures>')

    lines.append('</Material>')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
