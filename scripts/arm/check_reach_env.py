"""Physics check of the arm reach training environment; writes ``build/arm/reach_env_check.json``.

1. Hold the zero pose for 3 s with zero actions. Each joint's sag must match its
   static gravity torque divided by the 80 N m/rad gain, positions must stop
   drifting, and no body may touch anything (the training convex hulls must not
   collide at rest). PhysX reports a small constant velocity on drive-held joints
   under gravity (~0.07 rad/s on joint 2 at 120 Hz); it is proportional to the
   timestep and vanishes without gravity, so it is recorded but not tested.
2. PhysX must apply the +/-180 deg training limits and the drive gains.
3. 5 s of random actions: the state must stay finite and within the speed cap.
   Contacts seen here are reported, which shows that the contact sensor works.
"""
import argparse
import json
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--num_envs', type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402

from arthrobot import paths  # noqa: E402
from arthrobot.motors import POSITION_GAIN_NM_PER_RAD  # noqa: E402
from arthrobot_assets.arm import ARM_JOINTS, JAW_JOINTS, PINION_JOINT  # noqa: E402
from arthrobot_tasks.arm_reach.env_cfg import COMMAND, ArmReachEnvCfg  # noqa: E402
from arthrobot_tasks.arm_reach.mount import JOINT_LIMIT_DEG  # noqa: E402

# Static gravity torque (N m) of each joint at the zero pose with joint 1 vertical (URDF masses).
ZERO_POSE_GRAVITY_TORQUE_NM = (0., -7.73, -2.97, .06, 0., 0.)
SAG_TOLERANCE_DEG = .5          # gravity torque changes slightly as the arm sags
CONTACT_THRESHOLD_N = 1e-3
SPEED_CAP_WITH_OVERSHOOT_RAD_S = 12.


def make_env(num_envs):
    cfg = ArmReachEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.commands.tcp_target.debug_vis = False
    cfg.events.reset_arm.params['position_range'] = (0., 0.)
    cfg.observations.policy.enable_corruption = False
    cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.scene.contacts = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*')
    return ManagerBasedRLEnv(cfg)


def hold_check(env, report):
    robot = env.scene['robot']
    arm_ids = robot.find_joints(list(ARM_JOINTS), preserve_order=True)[0]
    jaw_ids = robot.find_joints(list(JAW_JOINTS), preserve_order=True)[0]
    pinion_id = robot.find_joints(PINION_JOINT)[0][0]
    limits = robot.data.joint_pos_limits[0, arm_ids].cpu().numpy()
    limit_rad = math.radians(JOINT_LIMIT_DEG)
    np.testing.assert_allclose(limits, [[-limit_rad, limit_rad]] * len(ARM_JOINTS), atol=1e-4)
    np.testing.assert_allclose(robot.data.joint_stiffness[0, arm_ids].cpu().numpy(), POSITION_GAIN_NM_PER_RAD, rtol=1e-5)
    env.reset()
    zero_actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    peak_contact_force = torch.zeros(env.num_envs, robot.num_bodies, device=env.device)
    for step in range(round(3. / env.step_dt)):
        env.step(zero_actions)
        if step == round(2. / env.step_dt):
            settled = robot.data.joint_pos[:, arm_ids].clone()
        if step >= round(1. / env.step_dt):
            forces = env.scene['contacts'].data.net_forces_w.norm(dim=-1)
            peak_contact_force = torch.maximum(peak_contact_force, forces)
    joint_positions = robot.data.joint_pos[0]
    predicted_sag = np.array(ZERO_POSE_GRAVITY_TORQUE_NM) / POSITION_GAIN_NM_PER_RAD
    command = env.command_manager.get_term(COMMAND)
    touching = sorted({robot.body_names[i] for i in torch.nonzero(peak_contact_force > CONTACT_THRESHOLD_N)[:, 1].tolist()})
    report['hold'] = dict(
        arm_sag_deg=np.degrees(joint_positions[arm_ids].cpu().numpy()).round(3).tolist(),
        predicted_sag_deg=np.degrees(predicted_sag).round(3).tolist(),
        jaw_m=joint_positions[jaw_ids].cpu().numpy().round(6).tolist(),
        pinion_rad=round(float(joint_positions[pinion_id]), 6),
        last_second_drift_deg=math.degrees(float((robot.data.joint_pos[:, arm_ids] - settled).abs().max())),
        reported_arm_speed_rad_s=float(robot.data.joint_vel[:, arm_ids].abs().max()),
        tcp_in_mount_m=command.world_to_mount(command.tcp_pos_w())[0].cpu().numpy().round(4).tolist(),
        bodies_in_contact=touching, max_contact_force_n=float(peak_contact_force.max()),
        training_limits_rad=limits.round(5).tolist())
    print('ARM_REACH_HOLD: ' + json.dumps(report['hold']), flush=True)
    sag_deg = np.degrees(robot.data.joint_pos[:, arm_ids].cpu().numpy())
    np.testing.assert_allclose(sag_deg, np.broadcast_to(np.degrees(predicted_sag), sag_deg.shape), atol=SAG_TOLERANCE_DEG)
    assert max(abs(value) for value in report['hold']['jaw_m']) < 1e-3, 'The gripper did not stay open'
    assert report['hold']['last_second_drift_deg'] < .01, 'The arm did not settle'
    assert not touching, f'Bodies in contact at rest: {touching}'


def random_action_check(env, report):
    robot = env.scene['robot']
    arm_ids = robot.find_joints(list(ARM_JOINTS), preserve_order=True)[0]
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    touched, peak_speed = set(), 0.
    for _ in range(round(5. / env.step_dt)):
        env.step(actions.uniform_(-2, 2))
        forces = env.scene['contacts'].data.net_forces_w.norm(dim=-1)
        touched |= {robot.body_names[i] for i in torch.nonzero(forces > CONTACT_THRESHOLD_N)[:, 1].tolist()}
        peak_speed = max(peak_speed, float(robot.data.joint_vel[:, arm_ids].abs().max()))
    finite = bool(torch.isfinite(robot.data.joint_pos).all() and torch.isfinite(robot.data.joint_vel).all())
    report['random_actions'] = dict(finite_state=finite, peak_arm_speed_rad_s=round(peak_speed, 3),
                                    bodies_that_made_contact=sorted(touched))
    print('ARM_REACH_RANDOM: ' + json.dumps(report['random_actions']), flush=True)
    assert finite, 'Non-finite state under random actions'
    assert peak_speed < SPEED_CAP_WITH_OVERSHOOT_RAD_S, f'Joint speed {peak_speed:.2f} rad/s exceeds the cap'


def main():
    report = {}
    env = make_env(args.num_envs)
    hold_check(env, report)
    random_action_check(env, report)
    env.close()
    report['passed'] = True
    (paths.build_dir('arm') / 'reach_env_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print('ARM_REACH_CHECK_PASSED', flush=True)


try:
    main()
finally:
    app.close()
