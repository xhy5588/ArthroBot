"""Physics check of the loose arm model (arthrobot_tasks.arm_planning.loose); writes a JSON report.

One environment per looseness level, each with the rigid arm next to the loose one.

1. Structure: the loose arm has servo, gearbox, flex and play joints for all six arm
   joints, and PhysX applied each level's gaps and bracket stiffness.
2. Hold the zero pose for 3 s. Gravity must push the loaded joints (2 and 3) to one
   end of their gearbox or bracket play, the output encoder must read servo + gearbox, the brackets must bend by gravity torque / stiffness (within 5%),
   nothing may touch, and the 'no_looseness' level must put the gripper where the
   rigid arm puts it.
3. 5 s of random servo targets: the state must stay finite.

    python scripts/arm/check_loose_arm.py
"""
import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--physics-hz', type=float, default=480.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from arthrobot import paths  # noqa: E402
from arthrobot_tasks.arm_planning.kinematics import ArmModel  # noqa: E402
from arthrobot_tasks.arm_planning.loose import (AXIS_INERTIA_ZERO_POSE, FLEX_JOINTS, GEAR_JOINTS, LEVELS,  # noqa: E402
                                                PLAY_JOINTS, apply_looseness, flex_damping)
from arthrobot_tasks.arm_planning.scene import ENV_SPACING, MountFrame, PairSceneCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402


def step(sim, scene, seconds):
    for _ in range(round(seconds * args.physics_hz)):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())


def main():
    levels = list(LEVELS.values())
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / args.physics_hz, device=args.device))
    cfg = PairSceneCfg(num_envs=len(levels), env_spacing=ENV_SPACING)
    cfg.robot_loose.spawn.activate_contact_sensors = True
    cfg.contacts = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/RobotLoose/.*')
    scene = InteractiveScene(cfg)
    sim.reset()
    rigid, loose = scene['robot'], scene['robot_loose']
    apply_looseness(loose, levels)
    report = {'physics_hz': args.physics_hz, 'levels': [level.name for level in levels]}

    play_ids = loose.find_joints(PLAY_JOINTS, preserve_order=True)[0]
    gear_ids = loose.find_joints(GEAR_JOINTS, preserve_order=True)[0]
    flex_ids = loose.find_joints(FLEX_JOINTS, preserve_order=True)[0]
    report['loose_joints'] = loose.joint_names
    report['loose_bodies'] = loose.body_names
    gear_limits = loose.root_physx_view.get_dof_limits()[:, gear_ids].numpy()
    np.testing.assert_allclose(gear_limits[..., 1] - gear_limits[..., 0],
                               np.radians([level.gear_deg for level in levels])[:, None].repeat(6, 1), rtol=1e-4)
    limits = loose.root_physx_view.get_dof_limits()[:, play_ids].numpy()
    stiffness = loose.root_physx_view.get_dof_stiffnesses()[:, flex_ids].numpy()
    damping = loose.root_physx_view.get_dof_dampings()[:, flex_ids].numpy()
    expected_gap = np.radians([level.play_deg for level in levels])
    np.testing.assert_allclose(limits[..., 1] - limits[..., 0], expected_gap[:, None].repeat(6, 1), rtol=1e-4)
    expected_k = np.array([level.flex_nm_per_rad for level in levels])[:, None].repeat(6, 1)
    np.testing.assert_allclose(stiffness, expected_k, rtol=1e-4)
    np.testing.assert_allclose(damping, flex_damping(torch.tensor(expected_k)).numpy(), rtol=1e-4)

    # Hold zero.
    frames = MountFrame(rigid), MountFrame(loose)
    for robot, frame in zip((rigid, loose), frames):
        robot.set_joint_position_target(torch.zeros(len(levels), 6, device=robot.device), joint_ids=frame.servo_ids)
    step(sim, scene, 1.0)
    contacts = scene['contacts']
    worst_force = torch.zeros_like(contacts.data.net_forces_w[..., 0])
    for _ in range(round(2.0 * args.physics_hz)):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.get_physics_dt())
        worst_force = torch.maximum(worst_force, contacts.data.net_forces_w.norm(dim=-1))
    model = ArmModel(device=args.device)
    inertia = model.axis_inertia(torch.zeros(1, 6, device=args.device))[0].cpu().numpy()
    np.testing.assert_allclose(inertia, AXIS_INERTIA_ZERO_POSE, atol=1e-5)  # The constants have 5 decimals.
    q_rigid = frames[0].encoders()[0]
    q_loose = frames[1].encoders()[0]
    play = loose.data.joint_pos[:, play_ids]
    gear = loose.data.joint_pos[:, gear_ids]
    flex = loose.data.joint_pos[:, flex_ids]
    q_output = frames[1].output_encoders()[0]
    output_error = (q_output - (q_loose + gear)).abs().max()
    report['output_encoder_max_error_deg'] = math.degrees(float(output_error))
    tcp_rigid, tcp_loose = frames[0].tcp(), frames[1].tcp()
    believed_loose = model.tcp(q_loose).float()
    touching = sorted({loose.body_names[i] for i in torch.nonzero(worst_force > 1e-3)[:, 1].tolist()})
    hold = {}
    for e, level in enumerate(levels):
        hold[level.name] = {
            'encoder_deg': np.degrees(q_loose[e].cpu().numpy()).round(3).tolist(),
            'flex_deg': np.degrees(flex[e].cpu().numpy()).round(4).tolist(),
            'gear_deg': np.degrees(gear[e].cpu().numpy()).round(4).tolist(),
            'play_deg': np.degrees(play[e].cpu().numpy()).round(4).tolist(),
            'gear_fraction_of_half_gap': (gear[e] / (math.radians(level.gear_deg) / 2)).cpu().numpy().round(3).tolist(),
            'play_fraction_of_half_gap': (play[e] / (math.radians(level.play_deg) / 2)).cpu().numpy().round(3).tolist(),
            # Statics: servo and bracket carry the same gravity torque, so flex / servo = 80 / k.
            'flex_over_servo_joints_2_3': (flex[e, 1:3] / q_loose[e, 1:3]).cpu().numpy().round(4).tolist(),
            'expected_flex_over_servo': round(80. / level.flex_nm_per_rad, 4),
            'tcp_offset_from_rigid_mm': float((tcp_loose[e] - tcp_rigid[e]).norm() * 1e3),
            'tcp_offset_from_encoder_estimate_mm': float((tcp_loose[e] - believed_loose[e]).norm() * 1e3),
        }
    report['hold'] = hold
    report['rigid_encoder_deg'] = np.degrees(q_rigid[0].cpu().numpy()).round(3).tolist()
    report['rigid_tcp_vs_model_mm'] = float((model.tcp(q_rigid).float() - tcp_rigid).norm(dim=-1).max() * 1e3)
    report['bodies_in_contact'] = touching
    print('LOOSE_HOLD: ' + json.dumps(hold), flush=True)

    assert report['rigid_tcp_vs_model_mm'] < 0.5, 'Torch FK does not match the rigid arm'
    # Half a count of the 14-bit encoder is 0.011 deg.
    assert report['output_encoder_max_error_deg'] < 0.012, 'Output encoder does not read servo + gearbox'
    # The loose arm's servo settles about 0.06 deg lower on joint 2 than the rigid arm's (1% of its gravity
    # sag; the two articulations use 16 and 8 solver iterations), about 0.5 mm at the gripper.
    assert hold['no_looseness']['tcp_offset_from_rigid_mm'] < 1.0, 'Loose model without looseness differs from rigid'
    assert not touching, f'Contacts at rest: {touching}'
    for e, level in enumerate(levels):
        if level.flex_nm_per_rad <= 2000:
            ratios = hold[level.name]['flex_over_servo_joints_2_3']
            assert all(abs(r / (80. / level.flex_nm_per_rad) - 1) < 0.05 for r in ratios), (level.name, ratios)
        for field in ('gear', 'play'):
            if getattr(level, f'{field}_deg') >= 0.25:
                # Joints 2 and 3 carry the arm's weight, so their gap sits at one end.
                fraction = hold[level.name][f'{field}_fraction_of_half_gap']
                assert all(abs(fraction[j]) > 0.95 for j in (1, 2)), (level.name, field, fraction)

    # Random servo targets.
    generator = torch.Generator(device=args.device).manual_seed(0)
    peak = 0.
    for _ in range(25):
        targets = (torch.rand(len(levels), 6, device=args.device, generator=generator) * 2 - 1)
        for robot, frame in zip((rigid, loose), frames):
            robot.set_joint_position_target(targets, joint_ids=frame.servo_ids)
        step(sim, scene, 0.2)
        peak = max(peak, float(loose.data.joint_vel.abs().max()))
    finite = bool(torch.isfinite(loose.data.joint_pos).all() and torch.isfinite(loose.data.joint_vel).all())
    report['random_targets'] = {'finite_state': finite, 'peak_joint_speed_rad_s': round(peak, 3)}
    print('LOOSE_RANDOM: ' + json.dumps(report['random_targets']), flush=True)
    assert finite, 'Non-finite state under random targets'
    report['passed'] = True
    (paths.build_dir('arm_planning') / 'check_loose_arm.json').write_text(json.dumps(report, indent=2) + '\n')
    print('LOOSE_CHECK_PASSED', flush=True)


exit_code = 1
try:
    main()
    exit_code = 0
except BaseException:
    import traceback
    traceback.print_exc()
    print('LOOSE_CHECK_FAILED', flush=True)
finally:
    # Kit's shutdown can hang after a contact sensor was used; the results are written.
    sys.stdout.flush()
    os._exit(exit_code)
