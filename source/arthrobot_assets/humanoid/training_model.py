"""Training variant of the humanoid: 21 bodies, 20 motor joints, grippers parked open.

Starting from :mod:`.build`:

- each gripper's jaws and pinion are merged into its tool body (mass, inertia and
  geometry preserved), removing the stiff rack-and-pinion constraints;
- the torso frame is moved to the middle of the hip and shoulder joints and turned
  so that X points forward, Y left and Z up;
- a start pose is computed in which the center of mass sits above the wheel axle
  and the wheels touch the floor (12 mm clearance before settling);
- conservative arm collision boxes are recorded for the standing task's
  arm-clearance reward.

Usage: ``python -m arthrobot_assets.humanoid.training_model`` writes ``build/humanoid_training/``.
"""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from arthrobot import paths
from arthrobot.mass import combine, transform
from arthrobot.urdf import (forward_kinematics, joint_axis, joint_child, mesh_scale, origin_matrix, read_inertial,
                            set_origin, write_inertial, write_urdf)
from arthrobot_assets.humanoid import WHEEL_BODIES, WHEEL_JOINTS
from arthrobot_assets.humanoid.build import SOURCE_SHA256, build

WHEEL_FLOOR_CLEARANCE_M = .012


def output_dir() -> Path:
    return paths.build_dir('humanoid_training')


def make_training_model(output: Path | None = None) -> tuple[Path, dict]:
    """Write ``robot.urdf`` and ``model.json``; return the URDF path and the model summary."""
    output = output or output_dir()
    robot, source_report = build(paths.build_dir('humanoid') / 'collision_meshes')
    cad_frames = forward_kinematics(robot)
    cad_visual_poses = {visual.get('name'): cad_frames[link.get('name')] @ origin_matrix(visual.find('origin'))
                        for link in robot.findall('link') for visual in link.findall('visual')}

    for gripper in source_report['grippers'].values():
        _merge_into_parent(robot, gripper['tool'], [*gripper['jaws'], gripper['pinion']])
    new_from_cad = _move_base_frame(robot, cad_frames)

    frames = forward_kinematics(robot)
    max_geometry_error = max(
        float(np.max(np.abs(frames[link.get('name')] @ origin_matrix(visual.find('origin'))
                            - new_from_cad @ cad_visual_poses[visual.get('name')])))
        for link in robot.findall('link') for visual in link.findall('visual'))
    assert max_geometry_error < 1e-10

    body_masses = {link.get('name'): transform(read_inertial(link), frames[link.get('name')])
                   for link in robot.findall('link')}
    total = combine(list(body_masses.values()))
    axle = np.mean([frames[name][:3, 3] for name in WHEEL_BODIES], axis=0)
    # Pitch the robot so that its center of mass is above the axle; level the two wheels.
    pitch = -np.arctan2(total.center[0] - axle[0], total.center[2] - axle[2])
    wheel_offset = frames[WHEEL_BODIES[0]][:3, 3] - frames[WHEEL_BODIES[1]][:3, 3]
    roll = -np.arctan2(wheel_offset[2], wheel_offset[1])
    start_rotation = Rotation.from_euler('xyz', [roll, pitch, 0.])
    start_height = _start_height(robot, frames, start_rotation.as_matrix())
    for name in WHEEL_JOINTS:
        joint = robot.find(f"joint[@name='{name}']")
        axis = frames[joint_child(joint)][:3, :3] @ joint_axis(joint)
        assert abs(axis[1]) > .99, 'Wheel axles must point sideways'

    assert len(robot.findall('link')) == 21 and len(robot.findall('joint')) == 20
    np.testing.assert_allclose(total.mass, source_report['total_mass_kg'], atol=1e-10)
    urdf_path = write_urdf(robot, output / 'robot.urdf')
    model = dict(source_sha256=SOURCE_SHA256, urdf_sha256=hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
                 total_mass_kg=total.mass, rigid_bodies=21, rotary_joints=20,
                 body_masses={name: properties.mass for name, properties in body_masses.items()},
                 center_of_mass_local_m=total.center.tolist(), base_frame_in_cad=np.linalg.inv(new_from_cad).tolist(),
                 initial_height_m=start_height, initial_rpy=[float(roll), float(pitch), 0.],
                 initial_quaternion_wxyz=start_rotation.as_quat()[[3, 0, 1, 2]].tolist(),
                 axle_local_m=axle.tolist(), arm_collision_bounds_m=_arm_collision_bounds(robot),
                 max_geometry_error=max_geometry_error, motor=source_report['motor'],
                 grippers='parked open and merged into the tool bodies (geometry, mass and inertia preserved)',
                 geometry_visuals=len(robot.findall('.//visual')), collision_meshes=len(robot.findall('.//collision')))
    (output / 'model.json').write_text(json.dumps(model, indent=2) + '\n')
    return urdf_path, model


def load_training_model(output: Path | None = None) -> tuple[ET.Element, dict]:
    """Parse the written training URDF and its summary (build them first if missing)."""
    output = output or output_dir()
    if not (output / 'robot.urdf').exists() or not (output / 'model.json').exists():
        make_training_model(output)
    return ET.parse(output / 'robot.urdf').getroot(), json.loads((output / 'model.json').read_text())


def _merge_into_parent(robot: ET.Element, parent_name: str, child_names: list[str]) -> None:
    """Fuse child bodies into their parent at the zero joint position."""
    parent = robot.find(f"link[@name='{parent_name}']")
    merged = [read_inertial(parent)]
    for name in child_names:
        child = robot.find(f"link[@name='{name}']")
        joint = next(joint for joint in robot.findall('joint') if joint_child(joint) == name)
        child_in_parent = origin_matrix(joint.find('origin'))
        merged.append(transform(read_inertial(child), child_in_parent))
        for kind in ('visual', 'collision'):
            for element in child.findall(kind):
                element = copy.deepcopy(element)
                set_origin(element, child_in_parent @ origin_matrix(element.find('origin')))
                parent.append(element)
        robot.remove(child)
        robot.remove(joint)
    write_inertial(parent, combine(merged))


def _move_base_frame(robot: ET.Element, cad_frames: dict) -> np.ndarray:
    """Re-express the torso: origin between the hip and shoulder joints, X forward (CAD -Y), Z up."""
    origin = np.mean([cad_frames[f'{side}_{part}_body'][:3, 3]
                      for side in ('left', 'right') for part in ('hip_1', 'arm_1')], axis=0)
    cad_from_new = np.eye(4)
    cad_from_new[:3, :3] = Rotation.from_euler('z', -np.pi / 2).as_matrix()
    cad_from_new[:3, 3] = origin
    new_from_cad = np.linalg.inv(cad_from_new)
    torso = robot.find("link[@name='torso']")
    for element in [*torso.findall('visual'), *torso.findall('collision')]:
        set_origin(element, new_from_cad @ origin_matrix(element.find('origin')))
    write_inertial(torso, transform(read_inertial(torso), new_from_cad))
    for joint in robot.findall('joint'):
        if joint.find('parent').get('link') == 'torso':
            set_origin(joint, new_from_cad @ origin_matrix(joint.find('origin')))
    return new_from_cad


def _start_height(robot: ET.Element, frames: dict, rotation: np.ndarray) -> float:
    """Torso height at which the wheels are lowest and clear the floor by WHEEL_FLOOR_CLEARANCE_M."""
    vertex_cache: dict[str, np.ndarray] = {}
    wheel_bottom = lowest_point = float('inf')
    for link in robot.findall('link'):
        for visual in link.findall('visual'):
            path = visual.find('geometry/mesh').get('filename')
            if path not in vertex_cache:
                vertex_cache[path] = trimesh.load_mesh(path).vertices
            placement = frames[link.get('name')] @ origin_matrix(visual.find('origin'))
            heights = (rotation @ trimesh.transform_points(vertex_cache[path], placement).T)[2]
            lowest_point = min(lowest_point, heights.min())
            if link.get('name') in WHEEL_BODIES:
                wheel_bottom = min(wheel_bottom, heights.min())
    assert lowest_point > wheel_bottom - .001, 'A hand or other body starts below the wheels'
    return float(-wheel_bottom + WHEEL_FLOOR_CLEARANCE_M)


def _arm_collision_bounds(robot: ET.Element) -> dict[str, list[list[float]]]:
    """Axis-aligned bounds of each arm body's colliders, in its own frame."""
    bounds = {}
    for link in robot.findall('link'):
        if '_arm_' not in link.get('name'):
            continue
        points = []
        for collision in link.findall('collision'):
            mesh = collision.find('geometry/mesh')
            vertices = trimesh.load_mesh(mesh.get('filename')).vertices * mesh_scale(mesh)
            points.append(trimesh.transform_points(vertices, origin_matrix(collision.find('origin'))))
        cloud = np.concatenate(points)
        bounds[link.get('name')] = [cloud.min(0).tolist(), cloud.max(0).tolist()]
    return bounds


if __name__ == '__main__':
    path, summary = make_training_model()
    print(json.dumps({key: summary[key] for key in ('total_mass_kg', 'initial_height_m', 'initial_rpy')}, indent=2))
