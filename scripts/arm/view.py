"""Interactive arm viewer, and the arm's physics validation (``--validate``).

The arm is mounted as in its CAD file (joint 1 horizontal), raised by
``--mount-lift-m`` above a ground plane. Physics: 960 Hz, PGS 128/128, gravity on.

GUI: joint sliders (with an optional one-joint-at-a-time demo) and gripper
controls (opening, shared torque limit, cameras). ``--validate`` runs 28 s
headless: the gripper closes to 25, 50 and 100 % of its stroke, reopens, then
the arm moves while holding the jaws half closed. It checks tracking, jaw-to-jaw
contact at full closure and the rack-pinion coupling, and writes
``build/arm/validation.json``.

    python scripts/arm/view.py [--gripper-demo] [--motor mg4010]
    python scripts/arm/view.py --validate
"""
import argparse
import json
import math
import sys
import traceback

from arthrobot.gripper import DM4310_GRIPPER as GRIPPER, VALIDATED_PHYSICS_DT

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--motor', choices=['mg5010', 'mg4010'], default='mg5010', help='Joint module motor.')
parser.add_argument('--gripper-mass-kg', type=float, help='Measured whole-gripper mass, including its motor.')
parser.add_argument('--mount-lift-m', type=float, default=.75, help='Raise the arm above its CAD placement.')
parser.add_argument('--headless', action='store_true')
parser.add_argument('--validate', action='store_true', help='Run the 28 s physics validation and exit.')
parser.add_argument('--gripper-demo', action='store_true', help='Cycle the gripper open and closed.')
parser.add_argument('--steps', type=int, default=0, help='Stop after this many control steps (0: run until closed).')
parser.add_argument('--rebuild', action='store_true', help='Re-import the USD even if the URDF is unchanged.')
args = parser.parse_args()

VALIDATION_SECONDS = 28.
CHECKPOINT_SECONDS = (4, 8, 12, 18, 24, 28)
JOINT_TARGET_RATE_RAD_S = math.radians(30)   # ramp for slider and demo targets
if args.validate:
    args.headless = True
    args.steps = round(VALIDATION_SECONDS / VALIDATED_PHYSICS_DT)
if not args.headless:
    from arthrobot.sim.rtx_compat import enable
    enable()
from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=args.headless, width=1440, height=900).app

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import GroundPlane  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402

from arthrobot.motors import load_motor  # noqa: E402
from arthrobot.sim.contacts import ContactMonitor  # noqa: E402
from arthrobot_assets.arm import ARM_JOINTS, BODIES, JAW_JOINTS, PINION_JOINT, TOOL_BODY  # noqa: E402
from arthrobot_assets.arm.usd import ensure_arm_usd, output_dir  # noqa: E402

ARM_PATH = '/World/Arm'


def validation_targets(time_s: float) -> tuple[float, np.ndarray | None]:
    """Jaw closing target (m) and arm joint targets (rad, or None for zero) at a validation time."""
    if time_s < 4:
        return 0., None
    if time_s < 8:
        return GRIPPER.jaw_stroke_m / 4, None
    if time_s < 12:
        return GRIPPER.jaw_stroke_m / 2, None
    if time_s < 18:
        return GRIPPER.jaw_stroke_m, None
    if time_s < 24:
        return 0., None
    return .024, np.deg2rad([8, -8, 8, -8, 8, -8])


def build_scene(usd_path, report):
    """World, lifted arm, ground, light and the two cameras; returns (world, ground height)."""
    world = World(stage_units_in_meters=1., physics_dt=VALIDATED_PHYSICS_DT, rendering_dt=1 / 60)
    world.get_physics_context().set_gravity(-9.81)
    world.get_physics_context().set_solver_type('PGS')
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
    arm = add_reference_to_stage(str(usd_path), ARM_PATH)
    bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    ground_z = float(bounds_cache.ComputeWorldBound(arm).ComputeAlignedRange().GetMin()[2] - .25)
    # Raise the whole arm; the ground stays where it was.
    mount = UsdGeom.Xformable(arm)
    existing_ops = mount.GetOrderedXformOps()
    lift = mount.AddTranslateOp(opSuffix='mountLift')
    lift.Set(Gf.Vec3d(0, 0, args.mount_lift_m))
    mount.SetXformOpOrder([lift, *existing_ops])

    bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    bounds = bounds_cache.ComputeWorldBound(arm).ComputeAlignedRange()
    low, high = np.array(bounds.GetMin()), np.array(bounds.GetMax())
    center, size = (low + high) / 2, float(max(high - low))
    world.scene.add(GroundPlane('/World/Ground', z_position=ground_z, size=max(2., size * 5)))
    UsdLux.DomeLight.Define(stage, '/World/Light').CreateIntensityAttr(600.)
    _add_camera(stage, '/World/Camera', center + size * np.array([1.5, .7, .75]), center, up=(0, 0, 1))
    # Gripper close-up, attached to the tool body: looks at the grasp center across the jaw axis.
    grasp_center = np.array(report['grasp_center_in_tool_m'])
    closing_axis = np.array(report['jaw_closing_axis_in_tool'])
    up = -grasp_center / np.linalg.norm(grasp_center)
    side = np.cross(closing_axis, up)
    side /= np.linalg.norm(side)
    _add_camera(stage, f'{ARM_PATH}/{TOOL_BODY}/GripperCamera', grasp_center + .38 * side + .08 * up, grasp_center, up)
    if not args.headless:
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().set_active_camera('/World/Camera')
    return world, stage, ground_z


def _add_camera(stage, path, eye, target, up):
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(28.)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.001, 100.))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
    camera.AddTransformOp().Set(view.GetInverse())


def check_runtime_properties(robot, report, motor):
    """PhysX must report the authored masses, caps and armature."""
    assert robot.num_dof == 9 and set(robot.body_names) == set(BODIES), (robot.dof_names, robot.body_names)
    arm_ids = [robot.dof_names.index(name) for name in ARM_JOINTS]
    jaw_ids = [robot.dof_names.index(name) for name in JAW_JOINTS]
    pinion_id = robot.dof_names.index(PINION_JOINT)
    np.testing.assert_allclose(robot.get_body_masses()[0], [report['bodies'][name]['mass_kg'] for name in robot.body_names],
                               rtol=1e-5)
    efforts = robot.get_max_efforts()[0]
    np.testing.assert_allclose(efforts[arm_ids], motor.rated_torque_nm, rtol=1e-5)
    np.testing.assert_allclose(efforts[jaw_ids], [GRIPPER.jaw_force_limit_n, 0.], rtol=1e-5)
    np.testing.assert_allclose(efforts[pinion_id], 0., atol=1e-8)
    velocity_limits = robot.get_joint_max_velocities()[0]
    np.testing.assert_allclose(velocity_limits[arm_ids], motor.max_speed_rad_s, rtol=1e-5)
    # Unlimited gripper speed is reported as a large finite value.
    assert np.all(velocity_limits[[*jaw_ids, pinion_id]] > 1e3), velocity_limits
    armatures = robot.get_armatures()[0]
    np.testing.assert_allclose(armatures[arm_ids], motor.reflected_inertia_kg_m2, rtol=1e-5)
    np.testing.assert_allclose(armatures[pinion_id], 0., atol=1e-8)
    return arm_ids, jaw_ids, pinion_id


def main():
    motor = load_motor(args.motor)
    usd_path, report = ensure_arm_usd(args.motor, args.gripper_mass_kg, args.rebuild)
    world, stage, ground_z = build_scene(usd_path, report)
    contacts = ContactMonitor(stage, ARM_PATH)
    arm_prim = stage.GetPrimAtPath(ARM_PATH)
    visual_count = sum(prim.IsA(UsdGeom.Mesh) and '/visuals/' in str(prim.GetPath())
                       for prim in Usd.PrimRange(arm_prim, Usd.TraverseInstanceProxies()))
    assert visual_count == report['included_visuals'], visual_count
    for prim in Usd.PrimRange(arm_prim):
        if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in ARM_JOINTS:
            joint = UsdPhysics.RevoluteJoint(prim)
            limits = [joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]
            assert not any(value is not None and np.isfinite(value) for value in limits), 'Arm joints are continuous'

    robot = world.scene.add(Articulation(ARM_PATH, name='arm'))
    world.reset()
    arm_ids, jaw_ids, pinion_id = check_runtime_properties(robot, report, motor)
    joint_panel = gripper_panel = None
    if not args.headless:
        from arthrobot.sim.ui.gripper_panel import GripperPanel
        from arthrobot.sim.ui.joint_panel import JointPanel
        joint_panel = JointPanel(list(ARM_JOINTS), mass_kg=report['total_mass_kg'],
                                 torque_cap_nm=motor.rated_torque_nm, motor_model=motor.model)
        gripper_panel = GripperPanel(demo=args.gripper_demo, mass_kg=report['gripper_mass_kg'],
                                     mass_is_estimate=report['gripper_mass_is_estimate'],
                                     camera_paths=[('Gripper view', f'{ARM_PATH}/{TOOL_BODY}/GripperCamera'),
                                                   ('Whole arm view', '/World/Camera')])
    result = dict(joint_names=robot.dof_names, body_names=robot.body_names, visual_meshes=visual_count,
                  mount_lift_m=args.mount_lift_m, ground_z_m=ground_z, mass_kg=report['total_mass_kg'],
                  motor=motor.summary(), collider_count=report['included_visuals'], gripper=report['gripper'],
                  gripper_mass_is_estimate=report['gripper_mass_is_estimate'], checkpoints=[],
                  max_arm_error_rad=0., max_jaw_error_m=0., max_coupling_error_m=0., steps=0)
    print('ARM_READY: ' + json.dumps({key: result[key] for key in ('joint_names', 'mass_kg', 'collider_count')}),
          flush=True)

    targets = np.zeros_like(robot.get_joint_positions())
    control_dt = VALIDATED_PHYSICS_DT if args.headless else 1 / 60
    torque_limit = GRIPPER.rated_torque_nm
    checkpoint_steps = [round(seconds / VALIDATED_PHYSICS_DT) for seconds in CHECKPOINT_SECONDS]
    step = demo_step = 0
    while app.is_running() and (not args.steps or step < args.steps):
        if not world.is_playing():
            app.update()
            continue
        time_s = step * control_dt
        desired = np.zeros_like(targets)
        if joint_panel:
            desired[0, arm_ids] = joint_panel.targets()[0]
            if joint_panel.demo.get_value_as_bool():
                demo_joint = arm_ids[int(time_s // 4) % len(arm_ids)]
                desired[0, demo_joint] = np.deg2rad(8) * np.sin(np.pi * (time_s % 4) / 4) ** 2
            if gripper_panel.demo.get_value_as_bool():
                closing = GRIPPER.jaw_stroke_m * (.5 - .5 * np.cos(2 * np.pi * demo_step * control_dt / 24))
                gripper_panel.show_closing(closing)
                demo_step += 1
            else:
                closing, demo_step = gripper_panel.closing_m(), 0
            desired[0, jaw_ids] = closing
            if gripper_panel.torque_limit_nm() != torque_limit:
                torque_limit = gripper_panel.torque_limit_nm()
                robot.set_max_efforts(np.array([[torque_limit / GRIPPER.pinion_radius_m]], dtype=np.float32),
                                      joint_indices=[jaw_ids[0]])
        if args.validate:
            jaw_target, arm_target = validation_targets(time_s)
            desired[0, jaw_ids] = jaw_target
            if arm_target is not None:
                desired[0, arm_ids] = arm_target
        # Rate-limit targets: arm joints at 30 deg/s, jaws at the gripper's target speed.
        max_change = np.full(robot.num_dof, JOINT_TARGET_RATE_RAD_S * control_dt)
        max_change[jaw_ids] = min(GRIPPER.target_jaw_speed_m_s, GRIPPER.pinion_radius_m * GRIPPER.max_speed_rad_s) * control_dt
        targets += np.clip(desired - targets, -max_change, max_change)
        targets[0, pinion_id] = targets[0, jaw_ids[0]] / GRIPPER.pinion_radius_m
        robot.set_joint_position_targets(targets)
        world.step(render=not args.headless)

        positions, velocities = robot.get_joint_positions(), robot.get_joint_velocities()
        assert np.isfinite(positions).all() and np.isfinite(velocities).all()
        arm_error = float(np.max(np.abs(positions[0, arm_ids] - targets[0, arm_ids])))
        jaw_error = float(np.max(np.abs(positions[0, jaw_ids] - targets[0, jaw_ids])))
        coupling_error = float(np.max(np.abs(positions[0, jaw_ids] - GRIPPER.pinion_radius_m * positions[0, pinion_id])))
        result['max_arm_error_rad'] = max(result['max_arm_error_rad'], arm_error)
        result['max_jaw_error_m'] = max(result['max_jaw_error_m'], jaw_error)
        result['max_coupling_error_m'] = max(result['max_coupling_error_m'], coupling_error)
        if joint_panel and step % round(.1 / control_dt) == 0:
            joint_panel.update(positions[:, arm_ids], tracking_error=arm_error)
            gripper_panel.update(positions[0, jaw_ids], positions[0, pinion_id], jaw_error)
        step += 1
        if step == round(1 / control_dt):
            base_positions, _ = robot.get_world_poses()
            np.testing.assert_allclose(base_positions[0, 2], args.mount_lift_m, atol=1e-5)
            print(f'ARM_MOUNT_HEIGHT_VERIFIED: lift={args.mount_lift_m:.3f} m, ground_z={ground_z:.3f} m', flush=True)
        if args.validate and step in checkpoint_steps:
            checkpoint = dict(step=step, jaw_target_m=desired[0, jaw_ids].tolist(),
                              jaw_actual_m=positions[0, jaw_ids].tolist(),
                              arm_actual_deg=np.rad2deg(positions[0, arm_ids]).tolist(), jaw_error_m=jaw_error,
                              arm_error_rad=arm_error, pinion_angle_rad=float(positions[0, pinion_id]),
                              coupling_error_m=coupling_error, contacts=sorted(contacts.pairs))
            result['checkpoints'].append(checkpoint)
            print('ARM_CHECKPOINT: ' + json.dumps(checkpoint), flush=True)
    result['steps'] = step
    if args.validate:
        checkpoints = result['checkpoints']
        assert len(checkpoints) == len(CHECKPOINT_SECONDS)
        assert all(checkpoint['jaw_error_m'] < .002 for checkpoint in checkpoints), 'A jaw did not reach its target'
        assert all(checkpoint['arm_error_rad'] < .15 for checkpoint in checkpoints), 'The arm failed to hold or move'
        assert result['max_coupling_error_m'] < .0001, 'Rack-pinion coupling failed'
        assert any(any('jaw_0' in first and 'jaw_1' in second for first, second in checkpoint['contacts'])
                   for checkpoint in checkpoints), 'Full closure never brought the jaws into contact'
        result['passed'] = True
        (output_dir(args.motor) / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
        print('ARM_VALIDATED: ' + json.dumps({key: result[key] for key in
                                              ('max_arm_error_rad', 'max_jaw_error_m', 'max_coupling_error_m')}),
              flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    print('ARM_FAILED', flush=True)
    sys.exit(1)
finally:
    app.close()
