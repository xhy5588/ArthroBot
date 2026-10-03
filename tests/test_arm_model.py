"""The arm URDF built from the CAD export: inventory, mass budget, gripper kinematics and training mount."""
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from arthrobot.gripper import DM4310_GRIPPER
from arthrobot.mass import is_physical_inertia
from arthrobot.urdf import forward_kinematics, origin_matrix
from arthrobot_assets.arm import ARM_BODIES, ARM_JOINTS, JAW_JOINTS, PINION_JOINT, TOOL_BODY
from arthrobot_assets.arm.build import build
from arthrobot_tasks.arm_reach.mount import (BASE_ANCHOR_IN_BASE, BASE_POSITION, MOUNT_QUAT_WXYZ, MOUNT_ROTATION,
                                             TCP_IN_TOOL)

STROKE = DM4310_GRIPPER.jaw_stroke_m


@pytest.fixture(scope='module')
def arm():
    return build('mg5010')


def to_world(point_in_base):
    return MOUNT_ROTATION @ np.asarray(point_in_base) + np.asarray(BASE_POSITION)


def test_inventory_and_mass_budget(arm):
    robot, report = arm
    assert len(robot.findall('link')) == 10 and len(robot.findall('joint')) == 9
    assert report['included_visuals'] == 535 and len(report['excluded_visuals']) == 82
    assert len({visual['source'] for visual in report['visual_manifest']}) == 535
    assert report['total_mass_kg'] == pytest.approx(3.825)
    allocations = {allocation['component']: allocation['mass_kg'] for allocation in report['allocations']}
    assert [allocations[f'{joint}_motor'] for joint in ARM_JOINTS] == pytest.approx([.46] * 6)
    assert allocations['gripper_DM4310_motor'] == pytest.approx(.300)
    assert allocations['gripper_mechanism_estimated'] == pytest.approx(.160)
    for body in report['bodies'].values():
        assert is_physical_inertia(np.array(body['inertia_kg_m2']))


def test_motor_choice_changes_mass_only():
    _, mg4010 = build('mg4010')
    assert mg4010['total_mass_kg'] == pytest.approx(3.825 - 6 * (.46 - .25))
    assert mg4010['included_visuals'] == 535


def test_arm_joints_are_continuous_motor_modules(arm):
    robot, _ = arm
    for index, name in enumerate(ARM_JOINTS):
        joint = robot.find(f"joint[@name='{name}']")
        assert joint.get('type') == 'continuous'
        assert joint.find('parent').get('link') == ARM_BODIES[index]
        assert joint.find('child').get('link') == ARM_BODIES[index + 1]
        assert 'upper' not in joint.find('limit').attrib
        assert float(joint.find('limit').get('effort')) == 13.


def test_one_driven_jaw_with_mimic_jaw_and_pinion(arm):
    robot, _ = arm
    driven, follower = (robot.find(f"joint[@name='{name}']") for name in JAW_JOINTS)
    pinion = robot.find(f"joint[@name='{PINION_JOINT}']")
    assert float(driven.find('limit').get('effort')) == pytest.approx(375.)
    assert float(follower.find('limit').get('effort')) == 0. and float(pinion.find('limit').get('effort')) == 0.
    assert follower.find('mimic').get('joint') == JAW_JOINTS[0]
    assert float(follower.find('mimic').get('multiplier')) == 1.
    assert pinion.find('mimic').get('joint') == JAW_JOINTS[0]
    assert float(pinion.find('mimic').get('multiplier')) == pytest.approx(125.)
    assert float(pinion.find('limit').get('upper')) == pytest.approx(6.25)
    for joint in (driven, follower):
        assert joint.get('type') == 'prismatic' and joint.find('parent').get('link') == TOOL_BODY
        assert float(joint.find('limit').get('upper')) == pytest.approx(STROKE)


def test_jaws_close_symmetrically_and_arm_stays_still(arm):
    robot, _ = arm
    open_pose = forward_kinematics(robot)
    closed = forward_kinematics(robot, {JAW_JOINTS[0]: STROKE})
    for jaw in ('jaw_0', 'jaw_1'):
        assert np.linalg.norm(closed[jaw][:3, 3] - open_pose[jaw][:3, 3]) == pytest.approx(STROKE)
    np.testing.assert_allclose(closed['jaw_0'][:3, 3] + closed['jaw_1'][:3, 3],
                               open_pose['jaw_0'][:3, 3] + open_pose['jaw_1'][:3, 3], atol=1e-10)
    for body in ARM_BODIES:
        np.testing.assert_allclose(closed[body], open_pose[body], atol=1e-12)
    # The pinion turns in place.
    np.testing.assert_allclose(closed['gripper_pinion'][:3, 3], open_pose['gripper_pinion'][:3, 3])
    assert not np.allclose(closed['gripper_pinion'][:3, :3], open_pose['gripper_pinion'][:3, :3])


def test_every_jaw_part_moves_with_its_jaw(arm):
    _, report = arm
    for index, names in enumerate(report['jaw_parts']):
        for visual in report['visual_manifest']:
            if visual['source'].rsplit(':', 1)[0] in names:
                assert visual['body'] == f'jaw_{index}'
    for visual in report['visual_manifest']:
        if '6_rail_170' in visual['source']:
            assert visual['body'] == TOOL_BODY


def test_mount_quaternion_and_anchor():
    w, x, y, z = MOUNT_QUAT_WXYZ
    np.testing.assert_allclose(Rotation.from_quat([x, y, z, w]).as_matrix(), MOUNT_ROTATION, atol=1e-12)
    np.testing.assert_allclose(to_world(BASE_ANCHOR_IN_BASE), 0., atol=1e-12)


def test_mount_puts_joint_1_vertical_through_the_anchor(arm):
    robot, _ = arm
    joint = robot.find(f"joint[@name='{ARM_JOINTS[0]}']")
    frame = origin_matrix(joint.find('origin'))
    axis = MOUNT_ROTATION @ frame[:3, :3] @ np.fromstring(joint.find('axis').get('xyz'), sep=' ')
    assert abs(axis[2]) > np.cos(np.radians(.1))
    np.testing.assert_allclose(to_world(frame[:3, 3])[:2], 0., atol=1e-4)


def test_mount_anchor_is_the_bottom_of_the_base(arm):
    robot, _ = arm
    lowest = np.inf
    for visual in robot.find(f"link[@name='{ARM_BODIES[0]}']").findall('visual'):
        mesh = trimesh.load_mesh(visual.find('geometry/mesh').get('filename'), process=False)
        vertices = trimesh.transform_points(mesh.vertices, origin_matrix(visual.find('origin')))
        lowest = min(lowest, (vertices @ MOUNT_ROTATION.T + BASE_POSITION)[:, 2].min())
    assert lowest == pytest.approx(0., abs=1e-4)


def test_zero_pose_rises_from_the_base_and_reaches_forward(arm):
    robot, report = arm
    frames = forward_kinematics(robot)
    np.testing.assert_allclose(report['grasp_center_in_tool_m'], TCP_IN_TOOL)
    shoulder_com = frames['shoulder_output'] @ np.r_[report['bodies']['shoulder_output']['center_of_mass_m'], 1.]
    assert to_world(shoulder_com[:3])[2] > .02
    tcp = to_world((frames[TOOL_BODY] @ np.r_[TCP_IN_TOOL, 1.])[:3])
    assert tcp[0] > .4
