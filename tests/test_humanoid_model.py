"""The humanoid URDF built from the CAD export, its training variant and the standing pose."""
import hashlib

import numpy as np
import pytest

from arthrobot.gripper import DM4310_GRIPPER
from arthrobot.mass import is_physical_inertia
from arthrobot.urdf import descendants, forward_kinematics, read_inertial
from arthrobot_assets.humanoid import MOTOR_JOINTS, SOURCE_URDF, WHEEL_BODIES
from arthrobot_assets.humanoid.build import SOURCE_SHA256, build
from arthrobot_assets.humanoid.nominal_pose import solve_pose
from arthrobot_assets.humanoid.training_model import load_training_model
from arthrobot_tasks.humanoid.standing.policy import load_standing_settings

STROKE, RADIUS, TORQUE = DM4310_GRIPPER.jaw_stroke_m, DM4310_GRIPPER.pinion_radius_m, DM4310_GRIPPER.rated_torque_nm


@pytest.fixture(scope='module')
def humanoid(tmp_path_factory):
    return build(tmp_path_factory.mktemp('collision_meshes'))


@pytest.fixture(scope='module')
def training_model():
    return load_training_model()


def test_source_and_complete_inventory(humanoid):
    _, report = humanoid
    assert hashlib.sha256(SOURCE_URDF.read_bytes()).hexdigest() == SOURCE_SHA256
    assert report['included_visuals'] == 1750 and len(report['excluded_visuals']) == 669
    assert report['included_visuals'] + len(report['excluded_visuals']) == 2419
    assert not set(report['part_ownership']) & set(report['excluded_visuals'])
    assert report['geometry_preservation_max_error'] < 1e-10


def test_four_branches_and_physical_inertias(humanoid):
    robot, report = humanoid
    assert len(report['bodies']) == 27 and len(report['joints']) == 26
    assert sum(joint['parent'] == 'torso' for joint in report['joints']) == 4
    assert len(forward_kinematics(robot)) == 27
    for body in report['bodies'].values():
        assert body['mass_kg'] > 0 and is_physical_inertia(np.array(body['inertia_kg_m2']))


def test_each_motor_moves_exactly_its_branch(humanoid):
    robot, report = humanoid
    zero = forward_kinematics(robot)
    for joint in report['joints']:
        if joint['type'] != 'continuous':
            continue
        moved = forward_kinematics(robot, {joint['name']: .2})
        changed = {name for name in zero if not np.allclose(zero[name], moved[name], atol=1e-7)}
        assert changed == descendants(robot, joint['child']), joint['name']
        # The motor's housing parts sit on the parent body and its output parts on the child body.
        housing_parts = [part for part in report['part_ownership'].values()
                         if part['allocation'] == joint['motor_housing']]
        assert {part['body'] for part in housing_parts} == {joint['parent'], joint['child']}


def test_grippers_close_symmetrically_with_one_driven_jaw(humanoid):
    robot, report = humanoid
    zero = forward_kinematics(robot)
    for gripper in report['grippers'].values():
        closed = forward_kinematics(robot, {gripper['jaw_joints'][0]: STROKE})
        movement = [closed[jaw][:3, 3] - zero[jaw][:3, 3] for jaw in gripper['jaws']]
        np.testing.assert_allclose(movement[0], -movement[1], atol=2e-6)
        assert np.linalg.norm(movement[0]) == pytest.approx(STROKE)
        np.testing.assert_allclose(zero[gripper['pinion']][:3, 3], closed[gripper['pinion']][:3, 3], atol=1e-10)
        efforts = [float(robot.find(f"joint[@name='{name}']/limit").get('effort'))
                   for name in (*gripper['jaw_joints'], gripper['pinion_joint'])]
        assert efforts == [pytest.approx(TORQUE / RADIUS), 0., 0.]


def test_motor_specs_and_explicit_mass_estimates(humanoid):
    robot, report = humanoid
    assert report['motor']['model'] == 'MG5010E-i36-V3' and report['motor']['count'] == 20
    motors = [allocation for allocation in report['allocations'] if allocation['component'].startswith('mg4010_fc')]
    assert len(motors) == 20 and sum(allocation['mass_kg'] for allocation in motors) == pytest.approx(9.2)
    assert sum(allocation['mass_kg'] for allocation in report['allocations']) == pytest.approx(report['total_mass_kg'])
    for joint in robot.findall('joint'):
        if joint.get('type') == 'continuous':
            assert float(joint.find('limit').get('effort')) == 13. and joint.find('limit').get('lower') is None
    estimated = {allocation['component'] for allocation in report['allocations'] if allocation['estimated']}
    assert {'centerboard_w_battery:0', 'part_1:0'} <= estimated


def test_collision_meshes_cover_every_body(humanoid):
    robot, report = humanoid
    assert len(robot.findall('.//collision')) == 150 and report['motor_exterior_colliders'] == 40
    for link in robot.findall('link'):
        assert link.find('collision') is not None, link.get('name')


def test_training_model_is_one_free_tree_with_parked_grippers(training_model):
    robot, model = training_model
    links = {link.get('name') for link in robot.findall('link')}
    joints = robot.findall('joint')
    assert len(links) == 21 and len(joints) == 20
    assert links - {joint.find('child').get('link') for joint in joints} == {'torso'}
    assert {joint.get('name') for joint in joints} == set(MOTOR_JOINTS)
    assert all(joint.get('type') == 'continuous' for joint in joints)
    assert len(robot.findall('.//visual')) == 1750 and len(robot.findall('.//collision')) == 150
    assert model['max_geometry_error'] < 1e-10
    masses = [read_inertial(link) for link in robot.findall('link')]
    assert all(properties.mass > 0 and is_physical_inertia(properties.inertia) for properties in masses)
    assert sum(properties.mass for properties in masses) == pytest.approx(13.274277815549151, abs=1e-8)


def test_standing_pose_balances_over_the_axle(training_model):
    robot, _ = training_model
    pose = solve_pose()
    # The included standing policy was trained with this pose (equal up to the solver tolerance).
    assert pose['joint_positions'] == pytest.approx(load_standing_settings()['nominal_pose']['joint_positions'],
                                                    abs=1e-6)
    frames = forward_kinematics(robot, pose['joint_positions'])
    links = {link.get('name'): read_inertial(link) for link in robot.findall('link')}
    total = sum(properties.mass for properties in links.values())
    center = sum(properties.mass * (frames[name][:3, :3] @ properties.center + frames[name][:3, 3])
                 for name, properties in links.items()) / total
    wheels = np.array([frames[name][:3, 3] for name in WHEEL_BODIES])
    # The torso is upright in this pose: the COM is above the axle and between the wheels, which are level.
    assert abs(center[0] - wheels[:, 0].mean()) < .002
    assert abs(wheels[0, 2] - wheels[1, 2]) < .002
    assert min(wheels[:, 1]) < center[1] < max(wheels[:, 1]) and center[2] > wheels[:, 2].max()
    assert min(pose['support_force_n']) > 0 and pose['maximum_static_torque_nm'] < 13.
