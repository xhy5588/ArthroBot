"""The arm's model for NVIDIA cuRobo: a planning URDF plus collision spheres.

:func:`build` writes, under ``build/arm_planning/``:

- ``arm_planning.urdf``: ``build/arm/arm.urdf`` with
  - a ``mount`` base link: the training mount frame (base bottom on the joint 1 axis,
    x forward, z up), so planning targets and obstacles use the reach task's frame;
  - arm joints as revolute within +-180 deg (the training limits; the real cable limits
    are unknown) and a speed cap of MAX_JOINT_SPEED;
  - the gripper fixed open;
  - a ``tcp`` link: the gripper frame of :mod:`.kinematics` at the TCP (z = approach,
    y = jaw closing axis).
- ``arm_curobo.yml``: cuRobo's robot config, with collision spheres fitted to every link's
  meshes and the self-collision ignore matrix.

Both hold absolute mesh paths, which is why they are generated rather than committed.
cuRobo (v2, https://github.com/NVlabs/curobo) is needed only for :func:`build_config`.
"""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from arthrobot import paths
from arthrobot.motors import load_motor
from arthrobot.urdf import origin_matrix, root_link
from arthrobot_assets.arm import ARM_JOINTS, JAW_JOINTS, TOOL_BODY
from arthrobot_tasks.arm_planning.kinematics import ArmModel, arm_urdf
from arthrobot_tasks.arm_reach.mount import BASE_ANCHOR_IN_BASE, MOUNT_ROTATION, TCP_IN_TOOL

MAX_JOINT_SPEED = 2.0  # rad/s; the MG5010 does 7.75, but a printed arm should move gently.
CLIP_BUFFER = 0.02  # m, cuRobo MorphIt clip_plane_buffer default.
SPHERE_BUFFER = 0.005  # m added to every sphere radius, for obstacle and self-collision checks.
TIP_LENGTH = 0.008  # m of each jaw tip enclosed by extra spheres.
TIP_CLUSTERS = 3
JAW_TIP_PAST_TCP_M = 0.0469  # How far the jaws reach past the TCP along the approach axis.


def output_dir() -> Path:
    return paths.build_dir('arm_planning')


def planning_urdf_path() -> Path:
    return output_dir() / 'arm_planning.urdf'


def config_path() -> Path:
    """cuRobo robot config; run ``scripts/arm/build_curobo_robot.py`` first."""
    return output_dir() / 'arm_curobo.yml'


def _origin(xyz, rotation) -> ET.Element:
    element = ET.Element('origin')
    element.set('xyz', ' '.join(f'{x:.9g}' for x in xyz))
    element.set('rpy', ' '.join(f'{x:.9g}' for x in Rotation.from_matrix(rotation).as_euler('xyz')))
    return element


def _fixed_joint(name, parent, child, xyz, rotation) -> ET.Element:
    joint = ET.Element('joint', name=name, type='fixed')
    joint.append(_origin(xyz, rotation))
    ET.SubElement(joint, 'parent', link=parent)
    ET.SubElement(joint, 'child', link=child)
    return joint


def write_planning_urdf() -> Path:
    tree = ET.parse(arm_urdf())
    robot = tree.getroot()
    base = root_link(robot)
    effort = load_motor('mg5010').rated_torque_nm
    for joint in robot.findall('joint'):
        if joint.get('name') in ARM_JOINTS:
            joint.set('type', 'revolute')
            limit = joint.find('limit')
            if limit is None:
                limit = ET.SubElement(joint, 'limit')
            limit.attrib.update(lower=f'{-math.pi:.9g}', upper=f'{math.pi:.9g}', effort=f'{effort:g}',
                                velocity=f'{MAX_JOINT_SPEED:g}')
        else:
            # Gripper: fixed in its open state (all gripper joints at zero).
            joint.set('type', 'fixed')
            for tag in ('axis', 'limit', 'mimic', 'dynamics'):
                for child in joint.findall(tag):
                    joint.remove(child)
    # Mount frame: p_mount = R (p_base - anchor), so the base sits at -R anchor with rotation R.
    rotation = np.asarray(MOUNT_ROTATION, dtype=float)
    robot.insert(0, ET.Element('link', name='mount'))
    robot.append(_fixed_joint('mount_to_base', 'mount', base, -rotation @ np.asarray(BASE_ANCHOR_IN_BASE), rotation))
    robot.append(ET.Element('link', name='tcp'))
    robot.append(_fixed_joint('tool_to_tcp', TOOL_BODY, 'tcp', TCP_IN_TOOL, ArmModel().gripper_in_tool.numpy()))
    ET.indent(tree)
    path = planning_urdf_path()
    tree.write(path, xml_declaration=True, encoding='utf-8')
    return path


def tip_spheres(jaw_joint: str) -> list[dict]:
    """Spheres (jaw link frame) around the jaw's last TIP_LENGTH along the approach axis, in TIP_CLUSTERS groups."""
    import trimesh
    robot = ET.parse(arm_urdf()).getroot()
    joint = next(j for j in robot.findall('joint') if j.get('name') == jaw_joint)
    link = next(l for l in robot.findall('link') if l.get('name') == joint.find('child').get('link'))
    vertices = []
    for visual in link.findall('visual'):
        mesh = trimesh.load_mesh(visual.find('geometry/mesh').get('filename'), process=False)
        mesh.apply_transform(origin_matrix(visual.find('origin')))
        vertices.append(mesh.vertices)
    vertices = np.concatenate(vertices)  # Jaw link frame.
    tool_from_jaw = origin_matrix(joint.find('origin'))
    approach = tool_from_jaw[:3, :3].T @ ArmModel().gripper_in_tool.numpy()[:, 2]  # In the jaw frame.
    along = vertices @ approach
    tip = vertices[along > along.max() - TIP_LENGTH]
    # Split the tip along its longest extent and enclose each part.
    spread = tip - tip.mean(0)
    direction = np.linalg.svd(spread, full_matrices=False)[2][0]
    order = np.argsort(spread @ direction)
    spheres = []
    for part in np.array_split(tip[order], TIP_CLUSTERS):
        center = (part.max(0) + part.min(0)) / 2
        spheres.append({'center': [float(x) for x in center],
                        'radius': float(np.linalg.norm(part - center, axis=1).max())})
    return spheres


def build_config(sphere_density: float = 1.0, collision_samples: int = 5000, seed: int = 42) -> Path:
    """Fit collision spheres (cuRobo's RobotBuilder) and write the cuRobo robot config."""
    import torch
    import yaml
    from curobo.robot_builder import RobotBuilder
    np.random.seed(seed)
    torch.manual_seed(seed)
    urdf = planning_urdf_path()
    base = root_link(ET.parse(arm_urdf()).getroot())
    builder = RobotBuilder(urdf_path=str(urdf), asset_path='', tool_frames=['tcp'])
    # Base spheres must not reach below the base's bottom face (mount z = 0, which is base-frame x = anchor x).
    # The fitter keeps spheres a further CLIP_BUFFER beyond the plane, so the plane sits that much lower; at
    # the anchor itself it left 4 spheres covering 7% of the 5 cm tall base.
    builder.fit_collision_spheres(sphere_density=sphere_density, compute_metrics=True,
                                  clip_links={base: ('x', BASE_ANCHOR_IN_BASE[0] - CLIP_BUFFER)})
    print(f'Fitted {builder.num_spheres} spheres on {len(builder.collision_link_names)} links')
    for link, metric in builder.link_metrics.items():
        print(f'  {link:18s} spheres {metric.num_spheres:3d}  coverage {metric.coverage * 100:5.1f}%  '
              f'protrusion {metric.protrusion_dist_mean * 1000:5.1f} mm')
    builder.compute_collision_matrix(prune_collisions=True, num_samples=collision_samples)
    print(f'Self-collision ignore matrix: {len(builder.collision_matrix)} entries')
    path = config_path()
    builder.save(builder.build(), str(path))
    data = yaml.safe_load(path.read_text())
    data = data.get('robot_cfg', data)
    # Spheres cover 82-98% of each link; at one pose the simulator saw the forearm touch the gripper
    # (true mesh gap ~1.7 mm) while the bare spheres did not overlap. collision_sphere_buffer inflates every
    # sphere for both obstacle and self checks; self_collision_buffer would add to it, so it stays zero.
    data['kinematics']['collision_sphere_buffer'] = SPHERE_BUFFER
    data['kinematics']['self_collision_buffer'] = {link: 0. for link in data['kinematics']['self_collision_buffer']}
    # The jaws reach 46.9 mm past the TCP; the fitted spheres stop about 6 mm short (MorphIt penalizes
    # spheres sticking out of the thin tips), and in simulation the tips then touched the table at grasps
    # the planner thought were clear. Add spheres that enclose each jaw's last TIP_LENGTH.
    for jaw in JAW_JOINTS:
        data['kinematics']['collision_spheres'][jaw.replace('gripper_', '')] += tip_spheres(jaw)
    # Most cuRobo loaders (planner, collision checker) expect the config under a 'robot_cfg' key.
    path.write_text(yaml.safe_dump({'robot_cfg': data}, sort_keys=False))
    return path
