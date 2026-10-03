"""Self-collision joint limits for the get-up task, measured offline with convex hulls.

PhysX simulates every collision mesh of the get-up asset as its convex hull, so two
hulls intersecting under forward kinematics is the contact PhysX would see. Each
rotary joint is swept alone from the standing pose (all other joints held), in
both directions, until a hull on its moving side touches a hull on the fixed side.
Bodies connected by a joint are skipped, as in the simulator. The limit is the
last clear angle minus a safety margin.

Single-joint sweeps give independent ranges only; combined poses can still
self-collide, which simulated self-collision handles. The wheels stay continuous.

Run ``scripts/humanoid/sweep_joint_limits.py`` to measure new limits.
"""
import json
from pathlib import Path
from typing import NamedTuple
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from arthrobot.urdf import descendants, forward_kinematics, joint_child, joint_parent, mesh_scale, origin_matrix
from arthrobot_assets.humanoid import WHEEL_JOINTS

JOINT_LIMITS_FILE = Path(__file__).resolve().parent / 'data/joint_limits.json'


class PlacedHull(NamedTuple):
    """Convex hull vertices in the world frame, with their axis-aligned bounds."""
    points: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def load_link_hulls(robot: ET.Element) -> dict[str, list[np.ndarray]]:
    """Per link: the convex hull vertices of each collision mesh, in the link frame."""
    vertex_cache, hulls = {}, {}
    for link in robot.findall('link'):
        pieces = []
        for collision in link.findall('collision'):
            mesh = collision.find('geometry/mesh')
            filename = mesh.get('filename')
            if filename not in vertex_cache:
                vertex_cache[filename] = np.asarray(trimesh.load_mesh(filename).vertices)
            vertices = vertex_cache[filename] * mesh_scale(mesh)
            vertices = trimesh.transform_points(vertices, origin_matrix(collision.find('origin')))
            pieces.append(vertices[ConvexHull(vertices).vertices])
        hulls[link.get('name')] = pieces
    return hulls


def place_hulls(pieces: list[np.ndarray], frame: np.ndarray) -> list[PlacedHull]:
    placed = []
    for vertices in pieces:
        points = trimesh.transform_points(vertices, frame)
        placed.append(PlacedHull(points, points.min(0), points.max(0)))
    return placed


def hulls_intersect(a: PlacedHull, b: PlacedHull, clearance: float) -> bool:
    """True if two convex point sets come within ``clearance`` metres per axis (an LP feasibility test)."""
    if np.any(a.lower - clearance > b.upper) or np.any(b.lower - clearance > a.upper):
        return False
    # Find convex weights w_a, w_b with |sum(w_a * a) - sum(w_b * b)| <= clearance on every axis.
    count_a, count_b = len(a.points), len(b.points)
    difference = np.hstack((a.points.T, -b.points.T))
    inequality = np.vstack((difference, -difference))
    weight_sums = np.zeros((2, count_a + count_b))
    weight_sums[0, :count_a] = 1.
    weight_sums[1, count_a:] = 1.
    result = linprog(np.zeros(count_a + count_b), A_ub=inequality, b_ub=np.full(6, clearance),
                     A_eq=weight_sums, b_eq=[1., 1.], bounds=(0, None), method='highs')
    return result.status == 0


def jointed_pairs(robot: ET.Element) -> set[frozenset]:
    """Body pairs connected by a joint; the simulator never collides these."""
    return {frozenset((joint_parent(joint), joint_child(joint))) for joint in robot.findall('joint')}


def first_self_collision(world_hulls: dict[str, list[PlacedHull]], jointed: set[frozenset],
                         clearance: float = 0., moving: set[str] | None = None) -> tuple[str, str] | None:
    """The first colliding body pair, or None.

    With ``moving`` given, only pairs of one moving and one fixed body are tested.
    """
    names = list(world_hulls)
    for i, first in enumerate(names):
        if moving is not None and first not in moving:
            continue
        for second in (names if moving is not None else names[i + 1:]):
            if (moving is not None and second in moving) or frozenset((first, second)) in jointed:
                continue
            if any(hulls_intersect(a, b, clearance) for a in world_hulls[first] for b in world_hulls[second]):
                return first, second
    return None


def sweep_joint(robot: ET.Element, hulls: dict[str, list[np.ndarray]], nominal: dict[str, float], joint_name: str,
                step: float, span: float, clearance: float) -> dict:
    """Free travel of one joint from ``nominal`` in both directions (radians)."""
    jointed = jointed_pairs(robot)
    moving = descendants(robot, joint_child(robot.find(f"joint[@name='{joint_name}']")))

    def collision_at(positions):
        frames = forward_kinematics(robot, positions)
        world = {name: place_hulls(pieces, frames[name]) for name, pieces in hulls.items()}
        return first_self_collision(world, jointed, clearance, moving)

    start = nominal.get(joint_name, 0.)
    contact_at_nominal = collision_at(nominal)
    result = dict(nominal=start, contact_at_nominal=list(contact_at_nominal) if contact_at_nominal else None)
    for sign, side in ((1, 'upper'), (-1, 'lower')):
        clear, contact = 0., None
        for step_index in range(1, int(span / step) + 1):
            pair = collision_at({**nominal, joint_name: start + sign * step_index * step})
            if pair and pair != contact_at_nominal:
                contact = pair
                break
            clear = step_index * step
        result[f'{side}_clear_offset'] = sign * clear
        result[f'{side}_contact'] = list(contact) if contact else None
    return result


def measure_joint_limits(robot: ET.Element, nominal: dict[str, float], step_deg: float = 2., span_deg: float = 170.,
                         clearance_mm: float = 3., margin_deg: float = 5., joints: list[str] | None = None,
                         log=print) -> dict:
    """Sweep every rotary non-wheel joint (or only ``joints``) and return the limits report."""
    hulls = load_link_hulls(robot)
    step, span, margin = np.radians(step_deg), np.radians(span_deg), np.radians(margin_deg)
    limits = {}
    for joint in robot.findall('joint'):
        name = joint.get('name')
        if name in WHEEL_JOINTS or joint.get('type') not in ('continuous', 'revolute'):
            continue
        if joints is not None and name not in joints:
            continue
        sweep = sweep_joint(robot, hulls, nominal, name, step, span, clearance_mm / 1000.)
        # The nominal pose always stays inside the range, even if the margin exceeds the free travel.
        sweep['lower'] = sweep['nominal'] + min(0., sweep['lower_clear_offset'] + margin)
        sweep['upper'] = sweep['nominal'] + max(0., sweep['upper_clear_offset'] - margin)
        limits[name] = sweep
        log(f"{name:12s} [{np.degrees(sweep['lower']):7.1f}, {np.degrees(sweep['upper']):7.1f}] deg"
            f"  lower hit {sweep['lower_contact']}  upper hit {sweep['upper_contact']}")
    return dict(method='single-joint convex-hull sweep from nominal standing pose',
                step_deg=step_deg, span_deg=span_deg, clearance_mm=clearance_mm, margin_deg=margin_deg,
                wheels='continuous',
                note='Independent ranges only; combined poses rely on simulated self-collision. '
                     'Not a substitute for real cable/bracket limits.',
                limits=limits)


def load_joint_limits(joint_order: tuple[str, ...], path: Path = JOINT_LIMITS_FILE) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper limits in ``joint_order``; joints without a measured limit get [-pi, pi]."""
    limits = json.loads(path.read_text())['limits']
    lower = np.array([limits[name]['lower'] if name in limits else -np.pi for name in joint_order])
    upper = np.array([limits[name]['upper'] if name in limits else np.pi for name in joint_order])
    return lower, upper
