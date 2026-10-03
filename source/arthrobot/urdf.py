"""URDF helpers: homogeneous transforms, forward kinematics and file I/O.

Transforms are 4x4 NumPy arrays. URDF ``rpy`` angles are fixed-axis
roll-pitch-yaw, which is SciPy's extrinsic ``'xyz'`` Euler convention.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from arthrobot.mass import MassProperties

ROTARY_JOINT_TYPES = ('revolute', 'continuous')


def origin_matrix(origin: ET.Element | None) -> np.ndarray:
    """Transform described by an ``<origin xyz=... rpy=...>`` element (identity if missing)."""
    transform = np.eye(4)
    if origin is not None:
        transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        roll_pitch_yaw = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
        transform[:3, :3] = Rotation.from_euler('xyz', roll_pitch_yaw).as_matrix()
    return transform


def set_origin(element: ET.Element, transform: np.ndarray) -> None:
    """Write ``transform`` into the element's ``<origin>``, creating it if needed."""
    origin = element.find('origin')
    if origin is None:
        origin = ET.SubElement(element, 'origin')
    origin.set('xyz', ' '.join(f'{value:.15g}' for value in transform[:3, 3]))
    roll_pitch_yaw = Rotation.from_matrix(transform[:3, :3]).as_euler('xyz')
    origin.set('rpy', ' '.join(f'{value:.15g}' for value in roll_pitch_yaw))


def joint_parent(joint: ET.Element) -> str:
    return joint.find('parent').get('link')


def joint_child(joint: ET.Element) -> str:
    return joint.find('child').get('link')


def joint_axis(joint: ET.Element) -> np.ndarray:
    """Unit joint axis in the joint frame (URDF default: +X)."""
    element = joint.find('axis')
    axis = np.array([1., 0., 0.]) if element is None else np.fromstring(element.get('xyz', '1 0 0'), sep=' ')
    return axis / np.linalg.norm(axis)


def root_link(robot: ET.Element) -> str:
    """The single link that is never a joint child."""
    children = {joint_child(joint) for joint in robot.findall('joint')}
    roots = {link.get('name') for link in robot.findall('link')} - children
    if len(roots) != 1:
        raise ValueError(f'Expected one root link, found {sorted(roots)}')
    return roots.pop()


def resolve_mimic_positions(robot: ET.Element, joint_positions: dict[str, float]) -> dict[str, float]:
    """Add positions for ``<mimic>`` joints: multiplier * reference + offset."""
    positions = dict(joint_positions)
    for joint in robot.findall('joint'):
        mimic = joint.find('mimic')
        if mimic is not None:
            reference = positions.get(mimic.get('joint'), 0.)
            positions[joint.get('name')] = (reference * float(mimic.get('multiplier', '1'))
                                            + float(mimic.get('offset', '0')))
    return positions


def forward_kinematics(robot: ET.Element, joint_positions: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    """World transform of every link, with the root link at the identity.

    Rotary joints rotate about their axis, prismatic joints translate along it;
    mimic joints follow their reference joint. Joints not listed stay at zero.
    Raises if the joint graph is not a single tree.
    """
    positions = resolve_mimic_positions(robot, joint_positions or {})
    frames = {root_link(robot): np.eye(4)}
    remaining = list(robot.findall('joint'))
    while remaining:
        progressed = False
        for joint in remaining[:]:
            parent, child = joint_parent(joint), joint_child(joint)
            if parent not in frames:
                continue
            if child in frames:
                raise ValueError(f'Link {child!r} has more than one parent')
            motion = np.eye(4)
            position = positions.get(joint.get('name'), 0.)
            if joint.get('type') in ROTARY_JOINT_TYPES:
                motion[:3, :3] = Rotation.from_rotvec(joint_axis(joint) * position).as_matrix()
            elif joint.get('type') == 'prismatic':
                motion[:3, 3] = joint_axis(joint) * position
            frames[child] = frames[parent] @ origin_matrix(joint.find('origin')) @ motion
            remaining.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError('Joint graph is disconnected or cyclic')
    if len(frames) != len(robot.findall('link')):
        raise ValueError('Some links are not connected to the root')
    return frames


def descendants(robot: ET.Element, link: str) -> set[str]:
    """``link`` and every link below it in the kinematic tree."""
    children: dict[str, list[str]] = {}
    for joint in robot.findall('joint'):
        children.setdefault(joint_parent(joint), []).append(joint_child(joint))
    found, stack = set(), [link]
    while stack:
        name = stack.pop()
        found.add(name)
        stack.extend(children.get(name, []))
    return found


def read_inertial(link: ET.Element) -> MassProperties:
    """Mass properties of a link, with the inertia rotated into the link frame."""
    inertial = link.find('inertial')
    frame = origin_matrix(inertial.find('origin'))
    values = {key: float(value) for key, value in inertial.find('inertia').attrib.items()}
    tensor = np.array([[values['ixx'], values['ixy'], values['ixz']],
                       [values['ixy'], values['iyy'], values['iyz']],
                       [values['ixz'], values['iyz'], values['izz']]])
    rotation = frame[:3, :3]
    return MassProperties(float(inertial.find('mass').get('value')), frame[:3, 3], rotation @ tensor @ rotation.T)


def write_inertial(link: ET.Element, properties: MassProperties) -> None:
    """Replace the link's ``<inertial>`` with ``properties`` (inertia in the link frame)."""
    previous = link.find('inertial')
    if previous is not None:
        link.remove(previous)
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'mass', value=str(properties.mass))
    ET.SubElement(inertial, 'origin', xyz=' '.join(map(str, properties.center)), rpy='0 0 0')
    tensor = properties.inertia
    ET.SubElement(inertial, 'inertia', ixx=str(tensor[0, 0]), ixy=str(tensor[0, 1]), ixz=str(tensor[0, 2]),
                  iyy=str(tensor[1, 1]), iyz=str(tensor[1, 2]), izz=str(tensor[2, 2]))


def mesh_scale(mesh_element: ET.Element) -> np.ndarray:
    return np.fromstring(mesh_element.get('scale', '1 1 1'), sep=' ')


def resolve_package_path(filename: str, package_root: Path) -> Path:
    """Map an exporter path such as ``package://assembly_2/meshes/x.stl`` into ``package_root``."""
    return package_root / filename.removeprefix('package://')


def write_urdf(robot: ET.Element, path: Path) -> Path:
    """Indent and write a URDF tree with an XML declaration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot)
    ET.ElementTree(robot).write(path, encoding='utf-8', xml_declaration=True)
    return path
