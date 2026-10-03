"""Interactive humanoid motor rig, and its actuator validation (``--validate``).

By default the torso is fixed in the air (``--clearance-m`` above the floor) so
every motor can be driven safely: sliders for the 18 limb joints, wheel speeds,
both grippers, and a one-motor-at-a-time demo. ``--free-base`` releases the
torso; no balance controller is provided (use the standing policy for that).

``--validate`` (headless, fixed torso) drives every rotary motor to +/-12 deg,
spins each wheel at +/-30 rpm for over one turn, closes each gripper to 0/25/50/0 mm
in zero gravity, then holds the zero pose under gravity for 6 s. Self-collision
stays on throughout. Writes ``build/humanoid/validation.json``.

    python scripts/humanoid/view.py [--demo] [--free-base]
    python scripts/humanoid/view.py --validate [--validate-joint left_knee]
"""
import argparse
import hashlib
import json
import math
import sys
import traceback

from arthrobot.gripper import DM4310_GRIPPER as GRIPPER, VALIDATED_PHYSICS_DT as PHYSICS_DT

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--headless', action='store_true')
parser.add_argument('--validate', action='store_true', help='Test every motor and gripper, then a gravity hold.')
parser.add_argument('--validate-joint', help='Validate a single rotary joint only (diagnostic).')
parser.add_argument('--demo', action='store_true', help='Move one motor at a time.')
parser.add_argument('--free-base', action='store_true', help='Release the torso (no balance controller).')
parser.add_argument('--steps', type=int, default=0)
parser.add_argument('--rebuild', action='store_true')
parser.add_argument('--clearance-m', type=float, default=.4, help='Height of the lowest part above the floor.')
args = parser.parse_args()
if args.validate_joint:
    args.validate = True
if args.validate:
    args.headless = True
    assert not args.free_base, 'Motor validation uses the fixed-torso rig'
if not args.headless:
    from arthrobot.sim.rtx_compat import enable
    enable()
from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=args.headless, width=1600, height=1000).app

import xml.etree.ElementTree as ET  # noqa: E402

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import GroundPlane  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402

from arthrobot.urdf import forward_kinematics  # noqa: E402
from arthrobot_assets.humanoid.usd import ensure_humanoid_usd, output_dir  # noqa: E402

ROBOT_PATH = '/World/Robot'
TARGET_RATE_RAD_S = math.radians(30)        # slider ramp in the GUI
TEST_RATE_RAD_S = math.radians(60)          # validation ramp
TEST_ANGLE_RAD = math.radians(12)
WHEEL_TEST_SPEED_RAD_S = math.pi            # 30 rpm
GRAVITY_HOLD_S, GRAVITY_HOLD_MAX_ERROR_RAD = 6., .35


def build_scene(usd_path):
    world = World(stage_units_in_meters=1., physics_dt=PHYSICS_DT, rendering_dt=1 / 60)
    physics = world.get_physics_context()
    physics.set_solver_type('PGS')
    physics.set_gravity(-9.81)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    robot_prim = add_reference_to_stage(str(usd_path), ROBOT_PATH)
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(robot_prim)
    low, high = np.array(bounds.ComputeAlignedRange().GetMin()), np.array(bounds.ComputeAlignedRange().GetMax())
    lift = float(-low[2] + args.clearance_m)
    xform = UsdGeom.Xformable(robot_prim)
    existing_ops = xform.GetOrderedXformOps()
    lift_op = xform.AddTranslateOp(opSuffix='mountLift')
    lift_op.Set(Gf.Vec3d(0, 0, lift))
    xform.SetXformOpOrder([lift_op, *existing_ops])
    world.scene.add(GroundPlane('/World/Ground', z_position=0., size=6.))
    UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(700.)
    center, size = (low + high) / 2 + [0, 0, lift], float(max(high - low))
    camera = UsdGeom.Camera.Define(stage, '/World/Camera')
    camera.CreateFocalLengthAttr(35.)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
    eye = center + size * np.array([.25, -1.8, .4])
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*center), Gf.Vec3d(0, 0, 1)).GetInverse())
    if not args.headless:
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().set_active_camera('/World/Camera')
    return world, stage, robot_prim, lift


def check_runtime_properties(robot, report):
    assert robot.num_dof == 26 and set(robot.body_names) == set(report['bodies'])
    index = {name: i for i, name in enumerate(robot.dof_names)}
    rotary_ids = [index[joint['name']] for joint in report['joints'] if joint['type'] == 'continuous']
    motor = report['motor']
    efforts = robot.get_max_efforts().copy()
    np.testing.assert_allclose(efforts[0, rotary_ids], motor['rated_torque_nm'], rtol=1e-5)
    np.testing.assert_allclose(robot.get_armatures()[0, rotary_ids], motor['reflected_inertia_kg_m2'], rtol=1e-5)
    np.testing.assert_allclose(robot.get_body_masses()[0], [report['bodies'][name]['mass_kg'] for name in robot.body_names],
                               rtol=1e-5)
    for gripper in report['grippers'].values():
        ids = [index[name] for name in [*gripper['jaw_joints'], gripper['pinion_joint']]]
        np.testing.assert_allclose(efforts[0, ids], [GRIPPER.jaw_force_limit_n, 0., 0.], atol=1e-5)
    return index, efforts


def couple_pinions(targets, report, index):
    for gripper in report['grippers'].values():
        targets[0, index[gripper['pinion_joint']]] = targets[0, index[gripper['jaw_joints'][0]]] / GRIPPER.pinion_radius_m


def main():
    usd_path, report = ensure_humanoid_usd(fixed_torso=not args.free_base, rebuild=args.rebuild)
    world, stage, robot_prim, lift = build_scene(usd_path)
    if args.validate:
        world.get_physics_context().set_gravity(0.)   # isolate motion; gravity returns for the final hold
    robot = world.scene.add(Articulation(ROBOT_PATH, name='humanoid'))
    world.reset()
    index, max_efforts = check_runtime_properties(robot, report)
    visual_count = sum(prim.IsA(UsdGeom.Mesh) and '/visuals/' in str(prim.GetPath())
                       for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()))
    assert visual_count == report['included_visuals'], visual_count
    result = dict(passed=False, fixed_torso=not args.free_base, body_count=len(robot.body_names),
                  dof_count=robot.num_dof, actuator_count=22, visual_count=visual_count,
                  collider_count=report['collision_mesh_count'], estimated_total_mass_kg=report['total_mass_kg'],
                  motor_tests=[], gripper_tests=[], max_coupling_error_m=0., gravity_hold=None)
    print('HUMANOID_READY: ' + json.dumps(result), flush=True)
    if args.validate:
        validate_motors(world, robot, report, index, result, lift)
        print('HUMANOID_VALIDATED: ' + json.dumps({key: result[key] for key in ('passed', 'max_coupling_error_m')}),
              flush=True)
        return
    run_interactive(world, robot, report, index, max_efforts)


def run_interactive(world, robot, report, index, max_efforts):
    rotary = [joint['name'] for joint in report['joints'] if joint['type'] == 'continuous']
    wheel_ids = [index[name] for name in rotary if name.endswith('_wheel')]
    panel = None
    if not args.headless:
        from arthrobot.sim.ui.humanoid_panel import HumanoidPanel
        panel = HumanoidPanel(report, demo=args.demo, fixed_torso=not args.free_base)
    targets = np.zeros((1, robot.num_dof), dtype=np.float32)
    desired, wheel_speeds = targets.copy(), targets.copy()
    control_dt = PHYSICS_DT if args.headless else 1 / 60
    gains = robot.get_gains()
    motors_enabled, wheels_in_velocity_mode = True, False
    demo_time, step = 0., 0
    while app.is_running() and (not args.steps or step < args.steps):
        if not world.is_playing():
            app.update()
            continue
        desired[:], wheel_speeds[:] = 0, 0
        demo = panel.demo.get_value_as_bool() if panel else args.demo
        if panel:
            for name, angle in panel.joint_targets_rad().items():
                desired[0, index[name]] = angle
            for name, speed in panel.wheel_speeds_rad_s().items():
                wheel_speeds[0, index[name]] = speed
            for side, gripper in report['grippers'].items():
                desired[0, [index[name] for name in gripper['jaw_joints']]] = panel.closing_m(side)
        if demo:
            # 6 s per motor: 20 rotary motors, then each gripper.
            slot, phase = int(demo_time // 6) % 22, demo_time % 6
            desired[:], wheel_speeds[:] = 0, 0
            if slot < 20:
                active = rotary[slot]
                desired[0, index[active]] = math.radians(8) * math.sin(2 * math.pi * phase / 6)
            else:
                side = ('left', 'right')[slot - 20]
                active = side + '_gripper'
                jaws = [index[name] for name in report['grippers'][side]['jaw_joints']]
                desired[0, jaws] = .025 * (1 - math.cos(2 * math.pi * phase / 6))
            if panel:
                panel.status.text = 'Testing: ' + active
            demo_time += control_dt
        else:
            demo_time = 0.
        # Manual wheel control is by speed: drop the wheels' position gains while it is active.
        velocity_mode = bool(panel) and not demo
        if velocity_mode != wheels_in_velocity_mode:
            stiffness, damping = (gain.copy() for gain in gains)
            if velocity_mode:
                stiffness[0, wheel_ids] = 0.
            robot.set_gains(kps=stiffness, kds=damping)
            wheels_in_velocity_mode = velocity_mode
        enable_requested = panel.motors_enabled.get_value_as_bool() if panel else True
        if enable_requested != motors_enabled:
            robot.set_max_efforts(max_efforts if enable_requested else np.zeros_like(max_efforts))
            if enable_requested:
                targets[:] = robot.get_joint_positions()
                if panel:
                    panel.hold_measured(robot.dof_names, targets[0])
                    desired[:], wheel_speeds[:] = targets, 0.
            motors_enabled = enable_requested
        max_change = np.full(robot.num_dof, TARGET_RATE_RAD_S * control_dt)
        for gripper in report['grippers'].values():
            max_change[[index[name] for name in gripper['jaw_joints']]] = GRIPPER.target_jaw_speed_m_s * control_dt
        targets += np.clip(desired - targets, -max_change, max_change)
        couple_pinions(targets, report, index)
        robot.set_joint_position_targets(targets)
        robot.set_joint_velocity_targets(wheel_speeds)
        world.step(render=not args.headless)
        positions, velocities = robot.get_joint_positions(), robot.get_joint_velocities()
        assert np.isfinite(positions).all() and np.isfinite(velocities).all()
        if panel and step % 6 == 0:
            panel.update(index, positions[0], velocities[0], motors_enabled, demo)
        step += 1


def validate_motors(world, robot, report, index, result, lift):
    """Zero-gravity motion tests, then a gravity hold. No teleporting, no raised torque limits."""
    directory = output_dir()
    urdf = ET.parse(directory / 'humanoid.urdf').getroot()
    body_index = {name: i for i, name in enumerate(robot.body_names)}
    result.update(test_conditions=('Rotary motors +/-12 deg; wheels also +/-30 rpm for over one turn each way; '
                                   'grippers 0/25/50/0 mm; zero gravity, then a 6 s gravity hold. '
                                   'Self-collision on throughout.'),
                  scope=args.validate_joint or 'all_22_actuators',
                  urdf_sha256=hashlib.sha256((directory / 'humanoid.urdf').read_bytes()).hexdigest(),
                  max_body_fk_error_m=0.)
    targets = np.zeros((1, robot.num_dof), dtype=np.float32)
    zero = np.zeros_like(targets)

    def drive_to(desired, seconds, jaw_rate_m_s=.015):
        max_change = np.full(robot.num_dof, TEST_RATE_RAD_S * PHYSICS_DT)
        for name, i in index.items():
            if name.endswith('_wheel'):
                max_change[i] = 2 * math.pi * PHYSICS_DT
        for gripper in report['grippers'].values():
            max_change[[index[name] for name in gripper['jaw_joints']]] = jaw_rate_m_s * PHYSICS_DT
        for _ in range(round(seconds / PHYSICS_DT)):
            targets[:] += np.clip(desired - targets, -max_change, max_change)
            couple_pinions(targets, report, index)
            robot.set_joint_position_targets(targets)
            world.step(render=False)
            positions = robot.get_joint_positions()
            assert np.isfinite(positions).all() and np.isfinite(robot.get_joint_velocities()).all()
            for gripper in report['grippers'].values():
                jaws = positions[0, [index[name] for name in gripper['jaw_joints']]]
                pinion = positions[0, index[gripper['pinion_joint']]]
                error = float(np.abs(jaws - pinion * GRIPPER.pinion_radius_m).max())
                result['max_coupling_error_m'] = max(result['max_coupling_error_m'], error)
        return robot.get_joint_positions()[0].copy()

    def check_body_poses(positions):
        """Simulated body positions must match URDF forward kinematics."""
        expected = forward_kinematics(urdf, dict(zip(robot.dof_names, positions)))
        poses = robot._physics_view.get_link_transforms()[0, :, :3]
        error = max(float(np.linalg.norm(poses[body_index[name]] - (frame[:3, 3] + [0, 0, lift])))
                    for name, frame in expected.items())
        result['max_body_fk_error_m'] = max(result['max_body_fk_error_m'], error)
        assert error < .002, ('Body hierarchy does not match forward kinematics', error)

    drive_to(zero, 1.)
    for spec in report['joints']:
        if spec['type'] != 'continuous' or (args.validate_joint and spec['name'] != args.validate_joint):
            continue
        name = spec['name']
        samples = []
        for angle in (TEST_ANGLE_RAD, -TEST_ANGLE_RAD, 0.):
            desired = zero.copy()
            desired[0, index[name]] = angle
            positions = drive_to(desired, 2.)
            error = abs(float(positions[index[name]]) - angle)
            samples.append(dict(target_rad=angle, actual_rad=float(positions[index[name]]), error_rad=error))
            check_body_poses(positions)
            assert error < .04, (name, samples)
        if name.endswith('_wheel'):
            samples += _spin_wheel(world, robot, index[name])
            check_body_poses(drive_to(zero, 2.))
        result['motor_tests'].append(dict(joint=name, source_joint=spec['source'], passed=True, samples=samples))
        print('MOTOR_TEST: ' + json.dumps(result['motor_tests'][-1]), flush=True)
    if args.validate_joint:
        assert len(result['motor_tests']) == 1, args.validate_joint
        result['passed'] = True
        (directory / f'validation_{args.validate_joint}.json').write_text(json.dumps(result, indent=2) + '\n')
        return
    for side, gripper in report['grippers'].items():
        samples = []
        jaw_ids = [index[name] for name in gripper['jaw_joints']]
        for closing in (0., .025, .05, 0.):
            desired = zero.copy()
            desired[0, jaw_ids] = closing
            positions = drive_to(desired, 5.)
            samples.append(dict(target_closing_m=closing, actual_jaws_m=positions[jaw_ids].tolist(),
                                pinion_rad=float(positions[index[gripper['pinion_joint']]])))
            assert np.max(np.abs(positions[jaw_ids] - closing)) < .002, (side, samples)
        result['gripper_tests'].append(dict(gripper=side, passed=True, samples=samples))
        print('GRIPPER_TEST: ' + json.dumps(result['gripper_tests'][-1]), flush=True)
    world.get_physics_context().set_gravity(-9.81)
    positions = drive_to(zero, GRAVITY_HOLD_S)
    rotary_ids = [index[joint['name']] for joint in report['joints'] if joint['type'] == 'continuous']
    result['gravity_hold'] = dict(duration_s=GRAVITY_HOLD_S, max_rotary_error_rad=float(np.abs(positions[rotary_ids]).max()),
                                  actual_angles_rad={robot.dof_names[i]: float(positions[i]) for i in rotary_ids})
    assert result['gravity_hold']['max_rotary_error_rad'] < GRAVITY_HOLD_MAX_ERROR_RAD, result['gravity_hold']
    assert result['max_coupling_error_m'] < .0001, result['max_coupling_error_m']
    assert len(result['motor_tests']) == 20 and len(result['gripper_tests']) == 2
    result['passed'] = True
    (directory / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')


def _spin_wheel(world, robot, wheel_id):
    """Spin one wheel by speed control in both directions; wheel angles wrap, so travel is accumulated."""
    stiffness, damping = robot.get_gains()
    speed_stiffness = stiffness.copy()
    speed_stiffness[0, wheel_id] = 0.
    robot.set_gains(kps=speed_stiffness, kds=damping)
    samples = []
    for speed in (WHEEL_TEST_SPEED_RAD_S, -WHEEL_TEST_SPEED_RAD_S, 0.):
        velocities = np.zeros((1, robot.num_dof), dtype=np.float32)
        velocities[0, wheel_id] = speed
        robot.set_joint_velocity_targets(velocities)
        previous = float(robot.get_joint_positions()[0, wheel_id])
        travel = 0.
        for _ in range(round((3. if speed else 1.) / PHYSICS_DT)):
            world.step(render=False)
            current = float(robot.get_joint_positions()[0, wheel_id])
            delta = current - previous
            travel += math.atan2(math.sin(delta), math.cos(delta))
            previous = current
        measured_speed = float(robot.get_joint_velocities()[0, wheel_id])
        sample = dict(target_velocity_rad_s=speed, actual_velocity_rad_s=measured_speed, traveled_rad=travel)
        assert abs(measured_speed - speed) < .02, sample
        if speed:
            assert math.copysign(1., speed) * travel > 2 * math.pi, sample
        samples.append(sample)
    robot.set_gains(kps=stiffness, kds=damping)
    return samples


try:
    main()
except BaseException:
    traceback.print_exc()
    print('HUMANOID_FAILED', flush=True)
    sys.exit(1)
finally:
    app.close()
