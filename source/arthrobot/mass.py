"""Mass properties of parts and rigid bodies.

A part's recorded mass is spread over its CAD volume at uniform density, so its
center of mass and inertia follow the part's shape. These are geometry-based
estimates, not measured or material-accurate values.
"""
from typing import NamedTuple, Sequence

import numpy as np
import trimesh


class MassProperties(NamedTuple):
    """Mass, center of mass and inertia tensor about the center of mass, in one frame."""
    mass: float
    center: np.ndarray
    inertia: np.ndarray


def combine(parts: Sequence[MassProperties]) -> MassProperties:
    """Combine parts expressed in a common frame (parallel-axis theorem)."""
    mass = sum(part.mass for part in parts)
    if mass <= 0:
        raise ValueError('A rigid body needs a positive total mass')
    center = sum(part.mass * part.center for part in parts) / mass
    inertia = np.zeros((3, 3))
    for part in parts:
        offset = part.center - center
        inertia += part.inertia + part.mass * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
    return MassProperties(mass, center, (inertia + inertia.T) / 2)


def transform(properties: MassProperties, matrix: np.ndarray) -> MassProperties:
    """Express mass properties in another frame; ``matrix`` maps old coordinates to new."""
    rotation, translation = matrix[:3, :3], matrix[:3, 3]
    return MassProperties(properties.mass, rotation @ properties.center + translation,
                          rotation @ properties.inertia @ rotation.T)


def uniform_density(mesh: trimesh.Trimesh, mass: float, placement: np.ndarray) -> MassProperties:
    """Mass properties of a solid mesh of the given mass, placed by a 4x4 transform."""
    rotation, translation = placement[:3, :3], placement[:3, 3]
    inertia = rotation @ (mesh.moment_inertia * mass / mesh.volume) @ rotation.T
    return MassProperties(mass, rotation @ mesh.center_mass + translation, inertia)


def distribute_by_volume(placed_meshes: Sequence[tuple[trimesh.Trimesh, np.ndarray]],
                         total_mass: float) -> list[MassProperties]:
    """Split ``total_mass`` over several placed meshes in proportion to their volumes."""
    total_volume = sum(mesh.volume for mesh, _ in placed_meshes)
    return [uniform_density(mesh, total_mass * mesh.volume / total_volume, placement)
            for mesh, placement in placed_meshes]


def is_physical_inertia(inertia: np.ndarray, tolerance: float = 1e-10) -> bool:
    """Positive-definite with principal moments satisfying the triangle inequality."""
    moments = np.linalg.eigvalsh(inertia)
    return bool(moments[0] > 0 and moments[2] <= moments[0] + moments[1] + tolerance)
