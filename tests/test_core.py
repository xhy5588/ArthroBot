"""Shared building blocks: transforms, forward kinematics, mass properties, motor and gripper specs."""
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from arthrobot.gripper import DM4310_GRIPPER
from arthrobot.mass import MassProperties, combine, is_physical_inertia, transform, uniform_density
from arthrobot.motors import load_motor
from arthrobot.urdf import forward_kinematics, origin_matrix, read_inertial, set_origin, write_inertial

TWO_LINK_URDF = """
<robot name="test">
  <link name="base"/><link name="arm"/><link name="slider"/><link name="follower"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="slide" type="prismatic">
    <parent link="arm"/><child link="slider"/><origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="2 0 0"/>
  </joint>
  <joint name="follow" type="revolute">
    <parent link="base"/><child link="follower"/><axis xyz="0 0 1"/><mimic joint="shoulder" multiplier="-2" offset="0"/>
  </joint>
</robot>"""


def test_origin_round_trip():
    element = ET.Element('joint')
    transform_in = np.eye(4)
    transform_in[:3, :3] = Rotation.from_euler('xyz', [.3, -.2, 1.1]).as_matrix()
    transform_in[:3, 3] = [.1, -.2, .3]
    set_origin(element, transform_in)
    np.testing.assert_allclose(origin_matrix(element.find('origin')), transform_in, atol=1e-14)


def test_forward_kinematics_rotary_prismatic_and_mimic():
    robot = ET.fromstring(TWO_LINK_URDF)
    frames = forward_kinematics(robot, {'shoulder': math.pi / 2, 'slide': .5})
    np.testing.assert_allclose(frames['base'], np.eye(4))
    np.testing.assert_allclose(frames['arm'][:3, 3], [1., 0., 0.])
    # The prismatic axis is normalized and turns with the arm: +X becomes +Y.
    np.testing.assert_allclose(frames['slider'][:3, 3], [1., .5, 0.], atol=1e-12)
    # The mimic joint follows the shoulder with multiplier -2.
    np.testing.assert_allclose(frames['follower'][:3, :3], Rotation.from_euler('z', -math.pi).as_matrix(), atol=1e-12)


def test_parallel_axis_theorem():
    intrinsic = np.diag([.1, .2, .25])
    total = combine([MassProperties(2., np.array([-1., 0., 0.]), intrinsic),
                     MassProperties(2., np.array([1., 0., 0.]), intrinsic)])
    assert total.mass == 4.
    np.testing.assert_allclose(total.center, 0.)
    np.testing.assert_allclose(total.inertia, np.diag([.2, 4.4, 4.5]))


def test_uniform_density_matches_solid_box_formula():
    box = trimesh.creation.box(extents=(1., 2., 3.))
    placement = np.eye(4)
    placement[:3, 3] = [5., 0., 0.]
    properties = uniform_density(box, 6., placement)
    np.testing.assert_allclose(properties.center, [5., 0., 0.], atol=1e-12)
    np.testing.assert_allclose(properties.inertia, np.diag([6.5, 5., 2.5]), atol=1e-12)
    assert is_physical_inertia(properties.inertia)


def test_transform_and_inertial_round_trip():
    rotation = np.eye(4)
    rotation[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    moved = transform(MassProperties(1., np.array([1., 0., 0.]), np.diag([1., 2., 3.])), rotation)
    np.testing.assert_allclose(moved.center, [0., 1., 0.], atol=1e-12)
    np.testing.assert_allclose(moved.inertia, np.diag([2., 1., 3.]), atol=1e-12)
    link = ET.Element('link')
    write_inertial(link, moved)
    restored = read_inertial(link)
    assert restored.mass == moved.mass
    np.testing.assert_allclose(restored.center, moved.center, atol=1e-15)
    np.testing.assert_allclose(restored.inertia, moved.inertia, atol=1e-15)


def test_non_physical_inertia_is_rejected():
    assert not is_physical_inertia(np.diag([1., 1., 3.]))   # violates the triangle inequality
    assert not is_physical_inertia(np.diag([0., 1., 1.]))


def test_motor_specifications():
    motor = load_motor('mg5010')
    assert motor.rated_torque_nm == 13.
    assert motor.max_speed_rad_s == pytest.approx(74 * math.pi / 30)
    assert motor.reflected_inertia_kg_m2 == pytest.approx(8.5e-5 * 36 ** 2)
    assert load_motor('mg4010').mass_kg < motor.mass_kg
    with pytest.raises(KeyError):
        load_motor('unknown')


def test_rack_and_pinion_gripper():
    gripper = DM4310_GRIPPER
    assert gripper.pinion_radius_m == pytest.approx(.008)
    assert gripper.jaw_force_limit_n == pytest.approx(375.)
    assert gripper.equal_load_force_per_jaw_n == pytest.approx(187.5)
    assert gripper.full_stroke_pinion_angle_rad == pytest.approx(6.25)
    # Energy check: pinion work over the full stroke equals the work of both jaws at the equal-load force.
    assert gripper.rated_torque_nm * gripper.full_stroke_pinion_angle_rad == pytest.approx(
        2 * gripper.equal_load_force_per_jaw_n * gripper.jaw_stroke_m)
