"""Catalog of standard structural parts that join motor modules together.

Parts are identified by the mesh file name the Onshape exports use for them
(``data/parts.json``), so any robot built from these parts gets the same masses.
"""
from dataclasses import dataclass
import json

from arthrobot.paths import DATA_DIR


@dataclass(frozen=True)
class Part:
    mesh_file: str
    name: str
    material: str
    mass_kg: float
    estimated: bool = False


def load_parts() -> dict[str, Part]:
    """Standard parts keyed by mesh file name."""
    records = json.loads((DATA_DIR / 'parts.json').read_text())['parts']
    return {mesh: Part(mesh_file=mesh, name=record['name'], material=record['material'],
                       mass_kg=record['mass_kg'], estimated=record.get('estimated', False))
            for mesh, record in records.items()}
