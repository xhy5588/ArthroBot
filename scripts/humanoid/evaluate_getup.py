"""Evaluate a get-up checkpoint on held-out lying poses, with a self-contact breakdown.

Deterministic policy, no pull force, no observation noise. Every robot runs:
robot i replays pose i % (bank size) under its own randomized physics.
Success = standing (torso > 0.46 m, tilt < 15 deg, both wheels down) for 2 s.

    python scripts/humanoid/evaluate_getup.py --headless
    python scripts/humanoid/evaluate_getup.py --headless --pose-bank source/arthrobot_tasks/humanoid/getup/data/pose_banks/fresh_seed2042_32.json
"""
import argparse
import json
import os
from pathlib import Path
import traceback

from arthrobot import paths
from arthrobot_tasks.humanoid.getup.evaluation import CHECKPOINT
from arthrobot_tasks.humanoid.getup.poses import POSE_BANK_DIR

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
parser.add_argument('--pose-bank', type=Path, default=POSE_BANK_DIR / 'held_out_seed1042_32.json')
parser.add_argument('--num-envs', type=int, default=320)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--report', type=Path, default=paths.BUILD_DIR / 'humanoid_training/getup_evaluation.json')
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app


def main():
    import torch
    from arthrobot_tasks.humanoid.assets import ensure_getup_usd
    from arthrobot_tasks.humanoid.getup.env import GetupEnv, GetupEnvCfg
    from arthrobot_tasks.humanoid.getup.evaluation import (action_bound_at, apply_run_settings, evaluate_held_out,
                                                           load_getup_policy, self_contact_by_body, summarize)
    usd_path, _ = ensure_getup_usd()
    cfg = GetupEnvCfg()
    cfg.asset_path, cfg.pose_bank = str(usd_path), str(args.pose_bank)
    cfg.scene.num_envs, cfg.seed, cfg.sim.device, cfg.obs_noise = args.num_envs, args.seed, args.device, False
    model, settings, state = load_getup_policy(args.checkpoint, args.device)
    apply_run_settings(cfg, settings)
    env = GetupEnv(cfg)
    env.action_bound = action_bound_at(state['iteration'], settings['final_action_bound'], settings['bound_iterations'])
    records = sorted(evaluate_held_out(env, model, 0, len(env.bank_root)), key=lambda record: record['env'])
    speeds = sorted(record['peak_limb_speed'] for record in records)
    report = dict(checkpoint=str(args.checkpoint), pose_bank=str(args.pose_bank), seed=args.seed,
                  update=state['iteration'], action_bound=env.action_bound, **summarize(records, env.families),
                  peak_limb_speed_p90=speeds[int(.9 * len(speeds))],
                  self_contact_by_body=self_contact_by_body(records), episodes_detail=records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('GETUP_EVALUATION: ' + json.dumps({key: value for key, value in report.items()
                                             if key not in ('episodes_detail', 'self_contact_by_body')}), flush=True)


exit_code = 1
try:
    main()
    exit_code = 0
except BaseException:
    traceback.print_exc()
finally:
    # Kit can hang while shutting down after headless get-up runs; the report is already written.
    os._exit(exit_code)
