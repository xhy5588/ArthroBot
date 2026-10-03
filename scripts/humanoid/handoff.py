"""Get up with the get-up policy, then hand over to the standing policy.

Even-numbered robots switch to the standing policy once they have stood for
--stand-hold seconds; odd-numbered robots keep the get-up policy for the whole
run (the control group). Both groups are then checked for staying upright:
tilt < 20 deg, torso > 0.46 m and both wheels on the floor >= 80% of the time.
The report also gives the arm posture at hand-over (deviation from the standing
pose and gripper positions in the torso frame). Deterministic policies, no pull
force, held-out lying poses.

    python scripts/humanoid/handoff.py --headless
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import traceback

from arthrobot import paths
from arthrobot_tasks.humanoid.getup.evaluation import CHECKPOINT as GETUP_CHECKPOINT
from arthrobot_tasks.humanoid.getup.poses import POSE_BANK_DIR
from arthrobot_tasks.humanoid.standing.policy import CHECKPOINT as STANDING_CHECKPOINT

GRIPPER_BODIES = ('left_arm_6_body', 'right_arm_6_body')
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--checkpoint', type=Path, default=GETUP_CHECKPOINT, help='Get-up checkpoint.')
parser.add_argument('--standing', type=Path, default=STANDING_CHECKPOINT, help='Standing checkpoint (rsl_rl).')
parser.add_argument('--pose-bank', type=Path, default=POSE_BANK_DIR / 'held_out_seed1042_32.json')
parser.add_argument('--num-envs', type=int, default=640)
parser.add_argument('--seconds', type=float, default=30.)
parser.add_argument('--stand-hold', type=float, default=1., help='Seconds of standing before the hand-over.')
parser.add_argument('--unactuated', type=float, help='Override the passive settle time, s.')
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--report', type=Path, default=paths.BUILD_DIR / 'humanoid_training/handoff.json')
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app


def main():
    import torch
    from isaaclab.utils.math import quat_apply_inverse
    from arthrobot_assets.humanoid import ARM_JOINTS, LIMB_JOINT_COUNT
    from arthrobot_tasks.humanoid.assets import ensure_getup_usd
    from arthrobot_tasks.humanoid.getup.env import GetupEnvCfg
    from arthrobot_tasks.humanoid.getup.evaluation import action_bound_at, apply_run_settings, load_getup_policy
    from arthrobot_tasks.humanoid.getup.handover import HandoverEnv, StandingTracker
    from arthrobot_tasks.humanoid.standing.policy import load_standing_actor

    usd_path, _ = ensure_getup_usd()
    cfg = GetupEnvCfg()
    cfg.asset_path, cfg.pose_bank = str(usd_path), str(args.pose_bank)
    cfg.scene.num_envs, cfg.seed, cfg.sim.device, cfg.obs_noise = args.num_envs, args.seed, args.device, False
    cfg.episode_length_s = args.seconds + 2.
    if args.unactuated is not None:
        cfg.unactuated_s = args.unactuated
    getup_policy, settings, state = load_getup_policy(args.checkpoint, args.device)
    apply_run_settings(cfg, settings)
    env = HandoverEnv(cfg)
    standing_policy = load_standing_actor(args.standing, env.device)
    env.action_bound = action_bound_at(state['iteration'], settings['final_action_bound'], settings['bound_iterations'])
    env.pull_force_n = 0.
    n, device = env.num_envs, env.device
    env.forced_pose = torch.arange(n, device=device) % len(env.bank_root)
    gripper_ids = env.robot.find_bodies(list(GRIPPER_BODIES), preserve_order=True)[0]
    hands_over = torch.arange(n, device=device) % 2 == 0
    handover_time = torch.full((n,), float('nan'), device=device)
    tracker = StandingTracker(env)
    posture = {}

    obs, _ = env.reset()
    with torch.no_grad():
        for step in range(int(args.seconds / env.step_dt)):
            actions = torch.where(env.standing_mode[:, None], standing_policy(env.standing_policy_observation()),
                                  getup_policy.mean(obs['policy']).clamp(-1., 1.))
            obs, *_ = env.step(actions)
            new = hands_over & ~env.standing_mode & (env.standing_time >= args.stand_hold)
            if new.any():
                ids = new.nonzero().flatten()
                data = env.robot.data
                limb_angles = data.joint_pos[ids][:, env.motor_ids[:LIMB_JOINT_COUNT]]
                gripper_offsets = data.body_pos_w[ids][:, gripper_ids] - data.root_pos_w[ids, None]
                grippers_in_torso = quat_apply_inverse(data.root_quat_w[ids, None].expand(-1, 2, -1).reshape(-1, 4),
                                                       gripper_offsets.reshape(-1, 3)).reshape(-1, 2, 3)
                for row, env_id in enumerate(ids.tolist()):
                    posture[env_id] = dict(
                        arm_deviation_rad=(limb_angles[row, :12] - env.nominal[:12]).abs().tolist(),
                        leg_deviation_rad=(limb_angles[row, 12:] - env.nominal[12:LIMB_JOINT_COUNT]).abs().tolist(),
                        gripper_in_torso_m=grippers_in_torso[row].tolist())
                env.hand_over(ids)
                handover_time[ids] = step * env.step_dt
            # Both groups are tracked from the moment they could have been handed over.
            tracker.update(~torch.isnan(handover_time) | (~hands_over & (env.best_standing >= args.stand_hold)))
    stayed = tracker.stayed_standing()

    def group_report(mask):
        tracked = mask & (tracker.steps > 0)
        if not tracked.any():
            return dict(robots=0, stayed_standing=None, mean_tracked_s=None, max_tilt_deg_mean=None)
        return dict(robots=int(tracked.sum()), stayed_standing=float(stayed[tracked].float().mean()),
                    mean_tracked_s=float((tracker.steps[tracked] * env.step_dt).mean()),
                    max_tilt_deg_mean=float(tracker.max_tilt_deg[tracked].mean()))

    def mean_or_none(values):
        return statistics.mean(values) if values else None

    records = list(posture.values())
    left_forward = [record['gripper_in_torso_m'][0][0] for record in records]
    right_forward = [record['gripper_in_torso_m'][1][0] for record in records]
    opposite = sum((left > .05 and right < -.05) or (left < -.05 and right > .05)
                   for left, right in zip(left_forward, right_forward))
    report = dict(getup_checkpoint=str(args.checkpoint), standing_checkpoint=str(args.standing), seed=args.seed,
                  seconds=args.seconds, handed_over=len(records), handover=group_report(hands_over),
                  getup_policy_only=group_report(~hands_over),
                  arm_deviation_from_nominal_rad_mean={
                      name: mean_or_none([record['arm_deviation_rad'][index] for record in records])
                      for index, name in enumerate(ARM_JOINTS)},
                  outside_standing_policy_range_fraction=sum(max(record['arm_deviation_rad']) > .25
                                                             for record in records) / max(1, len(records)),
                  gripper_forward_m=dict(left_mean=mean_or_none(left_forward), right_mean=mean_or_none(right_forward)),
                  one_gripper_forward_one_back_fraction=opposite / max(1, len(records)))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('GETUP_HANDOFF: ' + json.dumps(report), flush=True)


exit_code = 1
try:
    main()
    exit_code = 0
except BaseException:
    traceback.print_exc()
finally:
    # Kit can hang while shutting down after headless runs; the report is already written.
    os._exit(exit_code)
