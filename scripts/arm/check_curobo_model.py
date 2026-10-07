"""Check the arm's cuRobo model against the simulator-validated torch model; writes a JSON report.

    python scripts/arm/check_curobo_model.py

1. Forward kinematics: cuRobo's 'tcp' pose must match arthrobot_tasks.arm_planning.kinematics
   (TCP position and gripper axes, in the mount frame) at random joint angles.
2. Jaw tips: the spheres must reach past the end of the jaws (46.9 mm beyond the TCP along the
   approach axis).
3. Self-collision against PhysX: poses where the simulated arm touched itself must be flagged;
   IK grasp poses the simulator ran without contact should be free.
"""
import json
import math

import numpy as np
import torch

from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

from arthrobot_assets.arm import ARM_JOINTS
from arthrobot_tasks.arm_planning.controllers import IKController
from arthrobot_tasks.arm_planning.curobo_robot import JAW_TIP_PAST_TCP_M, config_path, output_dir
from arthrobot_tasks.arm_planning.kinematics import ArmModel
from arthrobot_tasks.arm_planning.targets import sample_grasps

DEVICE = 'cuda:0'
# Joint angles (deg) where the simulated rigid arm touched itself.
PHYSX_CONTACT_DEG = {
    'wrist folded into forearm (grasp IK, J5 -64)': (-34, 12, -91, 71, -64, -37),
    'jaws on base (grasp 12 cm from base)': (-44, 126, -91, -95, -12, -144),
    'jaws on shoulder (grasp 12 cm from base)': (-39, 123, -91, -73, -31, 16),
    'wrist folded into forearm (position IK, J5 -147)': (-11, 4, 59, -44, -147, 0),
}


def quaternion_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).view(*q.shape[:-1], 3, 3)


def contact_free_grasp_poses(model, sequences=32, targets=6):
    """IK goals of grasp sequences (straight down and tilted) that the simulator ran without contact."""
    goals = []
    for mode in ('down', 'tilted'):
        position, rotation, _ = sample_grasps(model, sequences, targets, 7, DEVICE, mode)
        no = torch.zeros(sequences, dtype=torch.bool, device=DEVICE)
        ik = IKController(model, sequences, DEVICE, no, no)
        for k in range(targets):
            ik.new_target(position[:, k], rotation[:, k])
            goals.append(ik.q_goal.clone())
            ik.q_desired = ik.q_goal.clone()
    return torch.cat(goals)


def main():
    config = str(config_path())
    model = ArmModel(device=DEVICE)
    kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(config))
    assert tuple(kinematics.joint_names) == ARM_JOINTS, kinematics.joint_names
    report = {}

    q = torch.tensor(np.random.default_rng(0).uniform(-math.pi, math.pi, (500, 6)), device=DEVICE)
    state = kinematics.compute_kinematics(JointState.from_position(q.float(), joint_names=kinematics.joint_names))
    pose = state.tool_poses.get_link_pose('tcp')
    position = pose.position.view(-1, 3).double()
    rotation = quaternion_to_matrix(pose.quaternion.view(-1, 4).double())
    expected_position, expected_rotation = model.pose(q)
    report['fk_max_position_error_mm'] = float((position - expected_position).norm(dim=-1).max() * 1e3)
    report['fk_max_axis_error'] = float((rotation - expected_rotation).abs().max())

    # Jaw tips: how far the spheres reach past the TCP along the approach axis (gripper open, q = 0).
    zero = torch.zeros(1, 6, device=DEVICE)
    spheres = kinematics.compute_kinematics(JointState.from_position(
        zero, joint_names=kinematics.joint_names)).robot_spheres.view(-1, 4)
    tcp_position, tcp_rotation = model.pose(zero)
    approach = tcp_rotation[0, :, 2].float()
    reach = ((spheres[:, :3] - tcp_position[0].float()) @ approach + spheres[:, 3]).max()
    report['spheres_reach_past_tcp_mm'] = round(float(reach) * 1e3, 1)
    report['jaw_tip_past_tcp_mm'] = JAW_TIP_PAST_TCP_M * 1e3

    checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=config, scene_model=None, collision_activation_distance=0.0))

    def self_collides(q_rad):
        q_rad = q_rad.float().view(-1, 1, 6)
        _, self_distance = checker.get_scene_self_collision_distance_from_joints(q_rad)
        return (self_distance.view(len(q_rad), -1).amax(-1) > 0).cpu()

    report['zero_pose_self_collision'] = bool(self_collides(zero)[0])
    known = torch.tensor(np.radians(list(PHYSX_CONTACT_DEG.values())), device=DEVICE)
    report['physx_contacts_flagged'] = {name: bool(f) for name, f in zip(PHYSX_CONTACT_DEG, self_collides(known))}
    free = contact_free_grasp_poses(model)
    report['physx_contact_free_grasps'] = len(free)
    report['physx_contact_free_flagged_fraction'] = round(float(self_collides(free).float().mean()), 3)
    print(json.dumps(report, indent=2))
    (output_dir() / 'check_curobo_model.json').write_text(json.dumps(report, indent=2) + '\n')
    assert report['fk_max_position_error_mm'] < 0.01 and report['fk_max_axis_error'] < 1e-5, 'FK mismatch'
    assert report['spheres_reach_past_tcp_mm'] > report['jaw_tip_past_tcp_mm'], 'jaw tips not covered'
    assert not report['zero_pose_self_collision'] and all(report['physx_contacts_flagged'].values())
    assert report['physx_contact_free_flagged_fraction'] < 0.02
    print('CUROBO_MODEL_CHECK_PASSED')


if __name__ == '__main__':
    main()
