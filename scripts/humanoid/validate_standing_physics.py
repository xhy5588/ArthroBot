"""Check that the get-up simulation settings keep the standing dynamics.

The standing policy was trained on the full asset (convex decomposition, PGS 32/8).
This runs it from the upright nominal pose in the get-up setup instead: the light
asset (one convex hull per mesh, self-collision joint limits) with TGS 8/1. If it
still balances, the simplified contact geometry and solver preserve the
balance-relevant dynamics.

Pass: >= 90% of trials keep tilt < 20 deg, torso > 0.46 m, both wheels on the
floor >= 80% of the time and COM drift < 0.25 m for the whole run.

    python scripts/humanoid/validate_standing_physics.py --headless
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import traceback

from arthrobot import paths
from arthrobot_tasks.humanoid.standing.policy import CHECKPOINT as STANDING_CHECKPOINT

MAX_DRIFT_M = .25
MIN_PASSED_FRACTION = .9
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--standing', type=Path, default=STANDING_CHECKPOINT, help='Standing checkpoint (rsl_rl).')
parser.add_argument('--num-envs', type=int, default=32)
parser.add_argument('--seconds', type=float, default=30.)
parser.add_argument('--position-iterations', type=int, default=8)
parser.add_argument('--velocity-iterations', type=int, default=1)
parser.add_argument('--randomize', action='store_true', help='Also apply the get-up domain randomization.')
parser.add_argument('--report', type=Path, default=paths.BUILD_DIR / 'humanoid_training/standing_physics_validation.json')
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app


def upright_pose_bank(settings: dict) -> dict:
    """A one-pose bank: the standing policy's nominal pose and start height."""
    from arthrobot_assets.humanoid import MOTOR_JOINTS
    pose = settings['nominal_pose']
    joints = [pose['joint_positions'].get(name, 0.) for name in MOTOR_JOINTS]
    root_pose = [0., 0., pose['initial_height_m'], *pose['initial_quaternion_wxyz']]
    return dict(joint_order=list(MOTOR_JOINTS), families=['upright'],
                poses=[dict(family='upright', joints=joints, root_pose=root_pose)])


def main():
    import torch
    from arthrobot_tasks.humanoid.assets import ensure_getup_usd
    from arthrobot_tasks.humanoid.getup.env import GetupEnvCfg
    from arthrobot_tasks.humanoid.getup.handover import HandoverEnv, StandingTracker
    from arthrobot_tasks.humanoid.standing.control import POSITION_SCALE_RAD
    from arthrobot_tasks.humanoid.standing.policy import load_standing_actor, load_standing_settings

    settings = load_standing_settings(args.standing)
    assert settings['control_mode'] == 'hybrid' and settings['position_scale_rad'] == POSITION_SCALE_RAD
    bank_file = Path(tempfile.mkdtemp()) / 'upright_pose.json'
    bank_file.write_text(json.dumps(upright_pose_bank(settings)))
    usd_path, _ = ensure_getup_usd()
    cfg = GetupEnvCfg()
    cfg.asset_path, cfg.pose_bank = str(usd_path), str(bank_file)
    cfg.standing_settings = str(Path(args.standing).parent / 'settings.json')
    cfg.scene.num_envs, cfg.sim.device = args.num_envs, args.device
    cfg.episode_length_s = args.seconds + 1.
    cfg.unactuated_s = 0.
    cfg.randomize, cfg.obs_noise = args.randomize, False
    cfg.robot.spawn.articulation_props.solver_position_iteration_count = args.position_iterations
    cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = args.velocity_iterations
    env = HandoverEnv(cfg)
    standing_policy = load_standing_actor(args.standing, env.device)

    env.reset()
    env.hand_over(torch.arange(env.num_envs, device=env.device))
    tracker = StandingTracker(env)
    tracked = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    max_drift = torch.zeros(env.num_envs, device=env.device)
    with torch.no_grad():
        for _ in range(int(args.seconds / env.step_dt)):
            env.step(standing_policy(env.standing_policy_observation()))
            tracker.update(tracked)
            max_drift = torch.maximum(max_drift, (env.center_of_mass()[:, :2] - env.handover_com[:, :2]).norm(dim=-1))
    passed = tracker.stayed_standing() & (max_drift < MAX_DRIFT_M)
    passed_fraction = float(passed.float().mean())
    report = dict(policy=str(args.standing), seconds=args.seconds, trials=env.num_envs, randomized=args.randomize,
                  solver=f'TGS {args.position_iterations}/{args.velocity_iterations}', asset=str(usd_path),
                  passed_fraction=passed_fraction, passed=passed_fraction >= MIN_PASSED_FRACTION,
                  max_tilt_deg=dict(mean=float(tracker.max_tilt_deg.mean()), max=float(tracker.max_tilt_deg.max())),
                  min_height_m=float(tracker.min_height_m.min()), max_drift_m=float(max_drift.max()),
                  wheel_support_fraction_min=float(tracker.wheel_support_fraction().min()))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('STANDING_PHYSICS_VALIDATION: ' + json.dumps(report), flush=True)


exit_code = 1
try:
    main()
    exit_code = 0
except BaseException:
    traceback.print_exc()
finally:
    # Kit can hang while shutting down after headless runs; the report is already written.
    os._exit(exit_code)
