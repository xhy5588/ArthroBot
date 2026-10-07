"""Arm kinematics and IK (arthrobot_tasks.arm_planning): FK against the URDF tools, Jacobians, gravity
torque against the simulator, position and gripper-pose IK, and the planning test scene."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from arthrobot.urdf import forward_kinematics
from arthrobot_assets.arm import ARM_JOINTS, TOOL_BODY
from arthrobot_tasks.arm_planning.controllers import IK_REST_WEIGHTS
from arthrobot_tasks.arm_planning.kinematics import WRIST_LIMIT, ArmModel, arm_urdf, axis_rotation, rotation_log
from arthrobot_tasks.arm_planning.obstacles import cuboids, curobo_scene
from arthrobot_tasks.arm_reach.mount import BASE_ANCHOR_IN_BASE, MOUNT_ROTATION, TCP_IN_TOOL

REACH_BOX = ((0.15, 0.40), (-0.25, 0.25), (0.05, 0.40))


@pytest.fixture(scope='module')
def model():
    return ArmModel()


def box_targets(n, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.stack([torch.rand(n, generator=generator) * (hi - lo) + lo for lo, hi in REACH_BOX], -1).double()


def test_fk_matches_urdf_tools(model):
    robot = ET.parse(arm_urdf()).getroot()
    q = np.random.default_rng(0).uniform(-1.5, 1.5, (10, 6))
    expected = [MOUNT_ROTATION @ ((forward_kinematics(robot, dict(zip(ARM_JOINTS, qi)))[TOOL_BODY]
                                   @ np.r_[TCP_IN_TOOL, 1])[:3] - BASE_ANCHOR_IN_BASE) for qi in q]
    np.testing.assert_allclose(model.tcp(torch.tensor(q)).numpy(), expected, atol=1e-9)


def test_jacobians_match_finite_differences(model):
    q = torch.tensor(np.random.default_rng(1).uniform(-1.5, 1.5, (4, 6)))
    step = 1e-6 * torch.eye(6, dtype=torch.float64)
    linear = torch.stack([(model.tcp(q + step[k]) - model.tcp(q)) / 1e-6 for k in range(6)], -1)
    np.testing.assert_allclose(model.jacobian(q).numpy(), linear.numpy(), atol=1e-5)
    _, before = model.pose(q)
    angular = torch.stack([rotation_log(model.pose(q + step[k])[1] @ before.transpose(1, 2)) / 1e-6
                           for k in range(6)], -1)
    np.testing.assert_allclose(model.jacobian6(q)[:, 3:].numpy(), angular.numpy(), atol=1e-5)


def test_gravity_torque_matches_simulator(model):
    # scripts/arm/check_reach_env.py: static torque at the zero pose with joint 1 vertical.
    np.testing.assert_allclose(model.gravity_torque(torch.zeros(1, 6))[0].numpy(),
                               [0.0, -7.73, -2.97, 0.06, 0.0, 0.0], atol=0.01)


def test_position_ik_keeps_the_wrist_near_straight(model):
    targets = box_targets(500, 2)
    zero = torch.zeros(500, 6, dtype=torch.float64)
    q, error = model.ik(targets, zero, rest=zero, rest_weights=torch.tensor(IK_REST_WEIGHTS).double())
    assert float(error.max()) < 5e-4
    # Joint 5 beyond about 70 deg folds the gripper into the forearm.
    assert float(q[:, 4].abs().max()) < np.radians(60)


def test_rotation_log_recovers_axis_and_angle():
    axis = torch.tensor([0.3, -0.5, 0.8], dtype=torch.float64)
    axis /= axis.norm()
    angles = torch.tensor([0.0, 0.3, 2.0, 3.1, np.pi], dtype=torch.float64)
    vectors = rotation_log(axis_rotation(axis, angles)[:, :3, :3])
    np.testing.assert_allclose(vectors.norm(dim=-1).numpy(), angles.numpy(), atol=1e-9)
    for vector, angle in zip(vectors[1:], angles[1:]):
        assert abs(float(vector @ axis)) == pytest.approx(float(angle), abs=1e-6)


def test_gripper_frame_is_a_rotation(model):
    _, rotation = model.pose(torch.zeros(1, 6))
    np.testing.assert_allclose((rotation[0].T @ rotation[0]).numpy(), np.eye(3), atol=1e-9)
    assert float(torch.linalg.det(rotation[0])) == pytest.approx(1.0, abs=1e-9)


def test_pose_ik_round_trip(model):
    q = torch.tensor(np.random.default_rng(4).uniform(-1.2, 1.2, (200, 6)))
    q[:, 4] = q[:, 4].clamp(-0.9, 0.9)
    position, rotation = model.pose(q)
    solved, position_error, rotation_error, valid = model.ik_pose(position, rotation, torch.zeros(200, 6))
    assert float(valid.double().mean()) > 0.98
    assert float(position_error[valid].max()) < 5e-4
    assert float(rotation_error[valid].max()) < np.radians(0.5)
    assert float(solved[valid, 4].abs().max()) <= WRIST_LIMIT + 1e-9


def test_top_down_grasps_reachable_near_the_base(model):
    generator = torch.Generator().manual_seed(5)
    n = 300
    position = torch.stack([torch.rand(n, generator=generator) * 0.10 + 0.15,
                            torch.rand(n, generator=generator) * 0.2 - 0.1,
                            torch.rand(n, generator=generator) * 0.10], -1).double()
    down = torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=torch.float64).expand(n, 3, 3)
    _, _, _, valid = model.ik_pose(position, down, torch.zeros(n, 6))
    assert float(valid.double().mean()) > 0.95


def test_planning_scene_leaves_the_base_free_and_grows_with_the_margin():
    boxes = cuboids()
    for name, (center, dims) in boxes.items():
        low, high = np.subtract(center, np.divide(dims, 2)), np.add(center, np.divide(dims, 2))
        inside_base = all(low[i] < 0.045 and high[i] > -0.045 for i in (0, 1)) and low[2] < 0.05
        assert not inside_base, f'{name} overlaps the base footprint'
    grown = curobo_scene(0.01)['cuboid']
    for name, (_, dims) in boxes.items():
        np.testing.assert_allclose(grown[name]['dims'], np.add(dims, 0.02))
