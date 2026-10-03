"""Upright standing pose of the humanoid, solved offline from the training model.

The leg joints are solved so that the whole-robot center of mass is above the
wheel axle, both wheels are level and both wheel axles are horizontal; the
shoulders are raised slightly so the long tools clear the legs. The result also
reports the static support forces and joint torques of that pose. It is a static
estimate (vertical wheel reactions, no contacts); simulation validates it.

The standing policy stores the pose it was trained with in its checkpoint
settings; other tasks read the pose from there.
"""
import json

import numpy as np
from scipy.optimize import least_squares
import trimesh

from arthrobot.urdf import (descendants, forward_kinematics, joint_axis, joint_child, mesh_scale, origin_matrix,
                            read_inertial)
from arthrobot_assets.humanoid import WHEEL_BODIES, WHEEL_JOINTS
from arthrobot_assets.humanoid.training_model import load_training_model, output_dir

# Shoulders raised so the PD-held tools still clear the legs after sagging under their weight.
SHOULDER_POSE = {'left_arm_1': -.25, 'right_arm_1': .20}
LEG_JOINT_BOUND_RAD = .6
WHEEL_FLOOR_CLEARANCE_M = .003
GRAVITY = 9.81


def solve_pose() -> dict:
    robot, _ = load_training_model()
    link_masses = {link.get('name'): read_inertial(link) for link in robot.findall('link')}
    joints = {joint.get('name'): joint for joint in robot.findall('joint')}
    legs = [name for name in joints if 'hip' in name or 'knee' in name]
    total_mass = sum(properties.mass for properties in link_masses.values())

    def evaluate(leg_angles):
        frames = forward_kinematics(robot, dict(SHOULDER_POSE, **dict(zip(legs, leg_angles))))
        centers = {name: frame[:3, :3] @ link_masses[name].center + frame[:3, 3] for name, frame in frames.items()}
        center_of_mass = sum(link_masses[name].mass * center for name, center in centers.items()) / total_mass
        axles = np.array([frames[name][:3, 3] for name in WHEEL_BODIES])
        return frames, centers, center_of_mass, axles

    def residual(leg_angles):
        frames, _, center_of_mass, axles = evaluate(leg_angles)
        wheel_axes = [frames[joint_child(joints[name])][:3, :3] @ joint_axis(joints[name]) for name in WHEEL_JOINTS]
        return np.r_[(center_of_mass[0] - axles[:, 0].mean()) * 100,     # COM above the axle
                     (axles[0, [0, 2]] - axles[1, [0, 2]]) * 100,        # wheels side by side
                     np.array(wheel_axes)[:, [0, 2]].ravel() * 5,        # axles horizontal
                     leg_angles * .02]                                   # stay near the CAD pose

    fit = least_squares(residual, np.zeros(len(legs)), bounds=(-LEG_JOINT_BOUND_RAD, LEG_JOINT_BOUND_RAD),
                        max_nfev=1000)
    frames, centers, center_of_mass, axles = evaluate(fit.x)
    collision_points = _collision_points(robot, frames)
    height = float(-min(collision_points[name][:, 2].min() for name in WHEEL_BODIES) + WHEEL_FLOOR_CLEARANCE_M)
    # Vertical wheel reactions from the lateral COM position.
    left_support = total_mass * GRAVITY * (center_of_mass[1] - axles[1, 1]) / (axles[0, 1] - axles[1, 1])
    support = np.array([left_support, total_mass * GRAVITY - left_support])
    static_torques = {}
    for name, joint in joints.items():
        child = joint_child(joint)
        moving = descendants(robot, child)
        pivot = frames[child][:3, 3]
        axis = frames[child][:3, :3] @ joint_axis(joint)
        moment = sum((np.cross(centers[link] - pivot, [0., 0., -GRAVITY * link_masses[link].mass]) for link in moving),
                     np.zeros(3))
        for index, wheel in enumerate(WHEEL_BODIES):
            if wheel in moving:
                moment += np.cross(axles[index] - pivot, [0., 0., support[index]])
        static_torques[name] = float(-axis @ moment)
    pose = dict(joint_positions=dict(SHOULDER_POSE, **dict(zip(legs, fit.x.tolist()))), initial_height_m=height,
                initial_rpy=[0., 0., 0.], initial_quaternion_wxyz=[1., 0., 0., 0.],
                center_of_mass_local_m=center_of_mass.tolist(), torso_upright=True,
                com_forward_offset_from_axle_m=float(center_of_mass[0] - axles[:, 0].mean()),
                wheel_center_height_difference_m=float(axles[0, 2] - axles[1, 2]),
                arm_clearance_m=float(min(points[:, 2].min() for name, points in collision_points.items()
                                          if '_arm_' in name) + height),
                support_force_n=support.tolist(), static_joint_torque_nm=static_torques,
                maximum_static_torque_nm=max(map(abs, static_torques.values())),
                note='Static estimate with vertical wheel reactions, exact URDF COMs and no self-contact forces. '
                     'Dynamic simulation must validate the pose.')
    assert fit.success and abs(pose['com_forward_offset_from_axle_m']) < .002
    assert abs(pose['wheel_center_height_difference_m']) < .002
    assert min(support) > 0 and pose['arm_clearance_m'] > .03
    assert pose['maximum_static_torque_nm'] < 13.
    return pose


def _collision_points(robot, frames) -> dict[str, np.ndarray]:
    points, vertex_cache = {}, {}
    for link in robot.findall('link'):
        clouds = []
        for collision in link.findall('collision'):
            mesh = collision.find('geometry/mesh')
            path = mesh.get('filename')
            if path not in vertex_cache:
                vertex_cache[path] = trimesh.load_mesh(path).vertices
            placement = frames[link.get('name')] @ origin_matrix(collision.find('origin'))
            clouds.append(trimesh.transform_points(vertex_cache[path] * mesh_scale(mesh), placement))
        points[link.get('name')] = np.concatenate(clouds)
    return points


if __name__ == '__main__':
    result = solve_pose()
    (output_dir() / 'nominal_pose.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
