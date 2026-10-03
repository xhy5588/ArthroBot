"""Score an arm reach policy with the deterministic policy; writes a JSON report.

Every arm tracks three 4-second targets per 12-second episode. A target's final
error is the TCP distance just before the next target appears. No observation
noise. By default this scores the included checkpoint.

    python scripts/arm/evaluate.py [--checkpoint logs/rsl_rl/arm_reach/<run>/model_550.pt]
"""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from arthrobot import paths

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--checkpoint', type=Path, default=paths.CHECKPOINTS_DIR / 'arm_reach/model_550.pt')
parser.add_argument('--num_envs', type=int, default=256)
parser.add_argument('--episodes', type=int, default=2)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--out', type=Path, help='Report path (default: next to the checkpoint is avoided; build/arm/).')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from arthrobot_tasks.arm_reach.agents import ArmReachPPORunnerCfg  # noqa: E402
from arthrobot_tasks.arm_reach.env_cfg import COMMAND, ArmReachEnvCfg_PLAY  # noqa: E402

REACHED_DISTANCE_M = .02


def main():
    agent_cfg = ArmReachPPORunnerCfg()
    env_cfg = ArmReachEnvCfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands.tcp_target.debug_vis = False
    env_cfg.seed = args.seed
    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    command = env.unwrapped.command_manager.get_term(COMMAND)
    step_dt = env.unwrapped.step_dt

    observations = env.get_observations()
    target = command.command.clone()
    distance = torch.norm(command.tcp_pos_w() - command.target_pos_w(), dim=-1)
    steps_on_target = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    time_to_reach = torch.full((env.num_envs,), float('nan'), device=env.device)
    final_errors, reach_times, targets = [], [], []
    for _ in range(args.episodes * round(env_cfg.episode_length_s / step_dt)):
        with torch.inference_mode():
            observations, _, _, _ = env.step(policy(observations))
        changed = (command.command != target).any(dim=-1)
        if changed.any():
            # Close out finished targets with the error measured before this step.
            final_errors.append(distance[changed].cpu())
            reach_times.append(time_to_reach[changed].cpu())
            targets.append(target[changed].cpu())
            steps_on_target[changed] = 0
            time_to_reach[changed] = float('nan')
        target = command.command.clone()
        steps_on_target += 1
        distance = torch.norm(command.tcp_pos_w() - command.target_pos_w(), dim=-1)
        first_reach = (distance < REACHED_DISTANCE_M) & torch.isnan(time_to_reach)
        time_to_reach[first_reach] = steps_on_target[first_reach].float() * step_dt

    final = torch.cat(final_errors).numpy()
    reach = torch.cat(reach_times).numpy()
    points = torch.cat(targets).numpy()
    report = dict(
        checkpoint=str(args.checkpoint), targets=int(final.size), seed=args.seed,
        final_error_mm={'median': float(np.median(final) * 1e3), 'p90': float(np.percentile(final, 90) * 1e3),
                        'p99': float(np.percentile(final, 99) * 1e3), 'max': float(final.max() * 1e3)},
        within_1cm=float(np.mean(final < .01)), within_2cm=float(np.mean(final < .02)),
        reached_2cm_fraction=float(np.mean(~np.isnan(reach))), median_time_to_2cm_s=float(np.nanmedian(reach)),
        worst_targets_mount_m=np.round(points[np.argsort(final)[-5:]], 3).tolist())
    output = args.out or paths.build_dir('arm') / 'reach_evaluation.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('ARM_REACH_EVALUATION: ' + json.dumps(report), flush=True)
    env.close()


try:
    main()
finally:
    app.close()
