"""Train the get-up policy: HoST-style multi-critic PPO from lying poses (headless).

Curricula:
- Pull force: an upward force on the torso starts at --pull-force newtons and
  drops by --pull-step after every held-out evaluation in which at least
  --pull-eval-success of the starts stood for 2 s. It never increases again.
- Action bound: joint targets are q + bound x action; the bound decays linearly
  from 1 rad to --final-action-bound over --bound-iterations.
Every --eval-interval updates the policy is evaluated on the held-out bank
(deterministic, no pull force); see arthrobot_tasks.humanoid.getup.evaluation.
Runs are written to logs/humanoid_getup/<time>/ (settings, metrics.csv,
TensorBoard events, evaluations/, model_<update>.pt and best.pt).

First stage (defaults):
    python scripts/humanoid/train_getup.py --num-envs 4096
Pipeline check:
    python scripts/humanoid/train_getup.py --smoke --num-envs 64 --iterations 2 --eval-interval 2

The included checkpoint (update 10,000) was trained from scratch in two stages, with the
HoST-style ending pose (near standing, a wide Gaussian of the arm and leg errors from the
standing policy's pose) instead of the narrow all-limb posture term, which pays nothing
when the arms are far away. Every value is in checkpoints/humanoid_getup/settings.json.
    # Stage 1, updates 0-5,700
    python scripts/humanoid/train_getup.py --num-envs 4096 --iterations 5700 --entropy 0.005 --max-std 0.6 \\
        --min-std 0.2 --gamma 0.997 --strength-min 0.85 --strength-max 1.0 --safety-weight 1.5 \\
        --self-contact-weight 3 --inherited-contact-weight 1 --joint-speed-weight 1 --torque-weight 0.25 \\
        --torso-rate-weight 1 --posture-weight 0 --arm-pose-width 0.1 --leg-pose-width 1 --height-schedule 3 \\
        --episode-length 12 --standing-fraction 0.25 \\
        --standing-bank source/arthrobot_tasks/humanoid/getup/data/pose_banks/standing_seed7_800.json
    # Stage 2, updates 5,700-12,000: the arms rested on their joint stops, so add a per-joint linear
    # arm pull, a lower arm-noise cap and a joint-stop penalty near standing (the included policy is update 10,000)
    python scripts/humanoid/train_getup.py --checkpoint <run>/model_5700.pt --iterations 6300 <stage 1 flags> \\
        --arm-pose-linear-weight 1 --arm-max-std 0.25 --standing-joint-limit-weight 10
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import arthrobot_tasks.humanoid.getup as getup_package
from arthrobot import paths
from arthrobot_tasks.humanoid.getup.poses import POSE_BANK_DIR

RUNS_DIR = paths.LOGS_DIR / 'humanoid_getup'
GETUP_SOURCES = [Path(__file__).resolve(), *sorted(Path(getup_package.__file__).parent.glob('*.py'))]
TRAINING_SUMMARY_COLUMNS = tuple(f'train_{key}' for key in (
    'episodes', 'stood_2s', 'standing_at_end', 'max_height_m', 'first_standing_s', 'clean_success',
    'self_contact_episodes', 'ready_success', 'clean_new_success', 'handover_ready_success', 'peak_limb_speed',
    'saturated_s', 'standing_start_episodes', 'standing_start_ready', 'standing_start_stayed', 'arm_ready_success',
    'final_arm_error_rad'))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--num-envs', type=int, default=512)
parser.add_argument('--iterations', type=int, default=6000)
parser.add_argument('--steps-per-env', type=int, default=24)
parser.add_argument('--run-dir', type=Path, help='Default: logs/humanoid_getup/<time>.')
parser.add_argument('--checkpoint', type=Path, help='Resume the model, optimizer and curricula.')
parser.add_argument('--training-bank', type=Path, default=POSE_BANK_DIR / 'training_seed42_400.json')
parser.add_argument('--held-out-bank', type=Path, default=POSE_BANK_DIR / 'held_out_seed1042_32.json')
parser.add_argument('--standing-bank', type=Path, help='Extra upright training starts (family "standing").')
parser.add_argument('--standing-fraction', type=float, default=0., help='Share of training resets from --standing-bank.')
parser.add_argument('--pull-force', type=float, default=65., help='Initial upward pull on the torso, N (~50%% of body weight).')
parser.add_argument('--pull-step', type=float, default=13.)
parser.add_argument('--pull-eval-success', type=float, default=.15)
parser.add_argument('--final-action-bound', type=float, default=.25)
parser.add_argument('--bound-iterations', type=int, default=3000)
parser.add_argument('--eval-interval', type=int, default=250)
parser.add_argument('--save-interval', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--entropy', type=float, default=.005)
parser.add_argument('--gamma', type=float, default=.99)
parser.add_argument('--min-std', type=float, default=.05)
parser.add_argument('--max-std', type=float, default=1.)
parser.add_argument('--safety-weight', type=float, default=1., help='Advantage weight of the safety reward group.')
parser.add_argument('--self-contact-weight', type=float, default=0., help='Per body in self-contact created by the policy.')
parser.add_argument('--inherited-contact-weight', type=float, default=0., help='Per body still touching since the settle.')
parser.add_argument('--joint-speed-weight', type=float, default=0.)
parser.add_argument('--joint-speed-soft', type=float, default=.5, help='Fraction of 74 rpm.')
parser.add_argument('--torque-weight', type=float, default=0.)
parser.add_argument('--torque-soft', type=float, default=9., help='N m.')
parser.add_argument('--torso-rate-weight', type=float, default=0.)
parser.add_argument('--torso-rate-soft', type=float, default=2., help='rad/s.')
parser.add_argument('--posture-l1-weight', type=float, default=0., help='Legs, near standing.')
parser.add_argument('--arm-posture-l1-weight', type=float, default=0., help='Arms, near standing.')
parser.add_argument('--posture-progress-weight', type=float, default=0.)
parser.add_argument('--height-schedule', type=float, default=0., help='Stand-up duration target, s (0: off).')
parser.add_argument('--strength-min', type=float, default=.9, help='Motor strength randomization, lower bound.')
parser.add_argument('--strength-max', type=float, default=1.1)
parser.add_argument('--posture-weight', type=float, default=1., help='Narrow all-limb posture term near standing.')
parser.add_argument('--arm-pose-width', type=float, default=0., help='HoST-style arm ending pose (0: off; HoST uses 0.1).')
parser.add_argument('--leg-pose-width', type=float, default=0., help='HoST-style leg ending pose (0: off).')
parser.add_argument('--arm-pose-linear-weight', type=float, default=0.,
                    help='Near standing: per-joint linear pull of the arms to the standing pose (works from the stops).')
parser.add_argument('--standing-joint-limit-weight', type=float, default=0.,
                    help='Safety group, near standing: per rad beyond 90%% of a joint range.')
parser.add_argument('--arm-max-std', type=float, help='Separate action-noise cap for the 12 arm actions.')
parser.add_argument('--episode-length', type=float, default=10., help='Episode length, s.')
parser.add_argument('--smoke', action='store_true', help='Pipeline check: log every update; run folder suffix _smoke.')
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
stop = {'requested': False}
app = AppLauncher(args).app
# Registered after Kit starts (Kit installs its own handlers, and its shutdown can hang
# mid-run). These only set a flag; the loop then saves a checkpoint and exits.
signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
signal.signal(signal.SIGINT, lambda *_: stop.update(requested=True))


def write_json(path: Path, data) -> None:
    """Write through a temporary file so readers never see a partial file."""
    temporary = Path(f'{path}.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def environment_settings() -> dict:
    """GetupEnvCfg fields set from the command line (recorded under the same names in settings.json)."""
    return dict(self_contact_weight=args.self_contact_weight, inherited_contact_weight=args.inherited_contact_weight,
                joint_speed_weight=args.joint_speed_weight, joint_speed_soft=args.joint_speed_soft,
                torque_weight=args.torque_weight, torque_soft_nm=args.torque_soft,
                torso_rate_weight=args.torso_rate_weight, torso_rate_soft=args.torso_rate_soft,
                posture_l1_weight=args.posture_l1_weight, arm_posture_l1_weight=args.arm_posture_l1_weight,
                posture_progress_weight=args.posture_progress_weight, height_schedule_s=args.height_schedule,
                strength_range=(args.strength_min, args.strength_max), standing_fraction=args.standing_fraction,
                posture_weight=args.posture_weight, arm_pose_width=args.arm_pose_width,
                leg_pose_width=args.leg_pose_width, arm_pose_linear_weight=args.arm_pose_linear_weight,
                standing_joint_limit_weight=args.standing_joint_limit_weight)


def main():
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from arthrobot_tasks.humanoid.assets import ensure_getup_usd
    from arthrobot_tasks.humanoid.getup.env import REWARD_GROUPS, GetupEnv, GetupEnvCfg
    from arthrobot_tasks.humanoid.getup.evaluation import action_bound_at, apply_run_settings, evaluate_held_out, summarize
    from arthrobot_tasks.humanoid.getup.poses import merge_banks
    from arthrobot_tasks.humanoid.getup.ppo import ActorCritic, MultiCriticPPO

    torch.manual_seed(args.seed)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = args.run_dir or RUNS_DIR / (stamp + ('_smoke' if args.smoke else ''))
    run_dir.mkdir(parents=True, exist_ok=args.checkpoint is not None and args.run_dir is not None)
    usd_path, limits = ensure_getup_usd()
    training_bank = json.loads(args.training_bank.read_text())
    held_out_bank = json.loads(args.held_out_bank.read_text())
    if args.standing_bank:
        training_bank = dict(merge_banks(training_bank, json.loads(args.standing_bank.read_text())),
                             standing_bank=str(args.standing_bank))
    write_json(run_dir / 'training_bank.json', training_bank)
    write_json(run_dir / 'held_out_bank.json', held_out_bank)

    cfg = GetupEnvCfg()
    cfg.asset_path, cfg.pose_bank = str(usd_path), str(run_dir / 'training_bank.json')
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, args.seed, args.device
    cfg.standing_fraction = args.standing_fraction
    cfg.episode_length_s = args.episode_length
    apply_run_settings(cfg, environment_settings())
    env = GetupEnv(cfg)
    held_out_first_row = env.append_poses(held_out_bank)
    held_out_count = len(held_out_bank['poses'])

    group_weights = (2.5, .1, 1., 1., args.safety_weight)
    model = ActorCritic(cfg.observation_space, cfg.state_space, cfg.action_space, len(REWARD_GROUPS)).to(env.device)
    ppo = MultiCriticPPO(model, group_weights, env.num_envs, args.steps_per_env, env.device, entropy=args.entropy,
                         max_std=args.max_std, min_std=args.min_std, gamma=args.gamma, group_names=REWARD_GROUPS,
                         arm_max_std=args.arm_max_std)
    state = dict(iteration=0, pull_force=args.pull_force, pull_changes=[], best=None)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
        ppo.load_state_dict(saved['ppo'])
        state.update(saved['state'])
        if args.run_dir is None:
            # A new run folder tracks its own best checkpoint; the source run keeps its best.pt.
            state['resumed_best'], state['best'] = state['best'], None

    # Apply the std bounds before the first rollout so the first update's KL is not inflated by the clamp.
    ppo.clamp_std()

    settings = dict(task='humanoid_getup', started=stamp, num_envs=env.num_envs, steps_per_env=args.steps_per_env,
                    iterations=args.iterations, observation=cfg.observation_space, critic=cfg.state_space,
                    actions=cfg.action_space, reward_groups=REWARD_GROUPS, group_weights=group_weights,
                    pull_force_initial_n=args.pull_force, pull_step_n=args.pull_step,
                    pull_gate=f'held-out evaluation stood_2s >= {args.pull_eval_success}',
                    final_action_bound=args.final_action_bound, bound_iterations=args.bound_iterations,
                    entropy=args.entropy, gamma=args.gamma, min_action_std=args.min_std, max_action_std=args.max_std,
                    arm_max_action_std=args.arm_max_std,
                    **environment_settings(), episode_length_s=args.episode_length,
                    self_contact_threshold_n=cfg.self_contact_n,
                    ready_tolerance=cfg.ready_tolerance, standing_bank=str(args.standing_bank),
                    training_bank=str(args.training_bank), held_out_bank=str(args.held_out_bank),
                    torque_limit_nm=cfg.torque_limit, kp=cfg.kp, kd=cfg.kd, physics='TGS 8/1 at 240 Hz, 60 Hz control',
                    asset=str(usd_path), joint_limits=limits['limits'], resumed_from=str(args.checkpoint),
                    source_sha256={source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in GETUP_SOURCES})
    write_json(run_dir / 'settings.json', settings)
    (run_dir / 'source').mkdir(exist_ok=True)
    for source in GETUP_SOURCES:
        shutil.copy2(source, run_dir / 'source' / source.name)
    writer = SummaryWriter(str(run_dir))
    metrics_file = (run_dir / 'metrics.csv').open('a', newline='')
    metrics_writer = None

    def save(path: Path):
        torch.save(dict(ppo=ppo.state_dict(), state=state, settings=settings), f'{path}.tmp')
        Path(f'{path}.tmp').replace(path)
        (RUNS_DIR / 'latest.txt').write_text(f'{path}\n')

    def set_curriculum():
        env.action_bound = action_bound_at(state['iteration'], args.final_action_bound, args.bound_iterations)
        env.pull_force_n = state['pull_force']

    def reset_all():
        obs, _ = env.reset()
        # Spread episode phases so that resets do not stay synchronized (HoST: init_at_random_ep_len).
        env.episode_length_buf[:] = torch.randint(0, int(env.max_episode_length), (env.num_envs,), device=env.device)
        return obs

    def evaluate(tag: str) -> dict:
        records = evaluate_held_out(env, model, held_out_first_row, held_out_count)
        report = dict(update=state['iteration'], tag=tag, pull_force_n=0., action_bound=env.action_bound,
                      **summarize(records, env.families), episodes_detail=records)
        (run_dir / 'evaluations').mkdir(exist_ok=True)
        write_json(run_dir / 'evaluations' / f'{tag}.json', report)
        return report

    set_curriculum()
    obs = reset_all()
    first_new_record = 0          # index into env.completed of the first episode since the last evaluation
    start_iteration = state['iteration']
    while state['iteration'] < start_iteration + args.iterations and not stop['requested']:
        started = time.perf_counter()
        set_curriculum()
        for _ in range(args.steps_per_env):
            actions = ppo.act(obs['policy'], obs['critic'])
            next_obs, _, terminated, truncated, extras = env.step(actions)
            ppo.process(extras['reward_groups'], terminated, truncated, next_obs['policy'], next_obs['critic'],
                        env.terminal_critic_obs)
            obs = next_obs
        ppo.returns(obs['critic'])
        collection_s = time.perf_counter() - started
        stats = ppo.update()
        state['iteration'] += 1
        update = state['iteration']

        recent = env.completed[first_new_record:][-512:]
        train = summarize([record for record in recent if record['family'] != 'standing'], env.families)
        upright = [record for record in recent if record['family'] == 'standing']
        if upright:
            train['standing_start_episodes'] = len(upright)
            train['standing_start_ready'] = sum(record['best_ready_s'] >= 1. for record in upright) / len(upright)
            train['standing_start_stayed'] = sum(record['standing_at_end'] for record in upright) / len(upright)
        if len(env.completed) > 3000:
            trimmed = len(env.completed) - 2048
            env.completed = env.completed[trimmed:]
            first_new_record = max(0, first_new_record - trimmed)
        control_steps = max(1, env.control_stat_steps)
        row = dict(update=update, transitions_per_s=args.steps_per_env * env.num_envs / (time.perf_counter() - started),
                   collection_s=collection_s, pull_force_n=state['pull_force'], action_bound=env.action_bound,
                   torque_saturation=float(env.control_stats[0] / control_steps),
                   torque_rms_nm=float(env.control_stats[1] / control_steps),
                   self_contact_step_fraction=float(env.self_contact_steps / control_steps), **ppo.diagnostics,
                   **{f'action_std_{part}': float(model.log_std[joints].exp().mean())
                      for part, joints in (('arms', slice(0, 12)), ('legs', slice(12, 18)), ('wheels', slice(18, 20)))},
                   **stats, **{f'train_{key}': value for key, value in train.items()},
                   **{f'group_reward_{group}': float(ppo.storage['rewards'][..., i].mean() / env.step_dt)
                      for i, group in enumerate(REWARD_GROUPS)})
        env.control_stats.zero_()
        env.self_contact_steps.zero_()
        env.control_stat_steps = 0
        for key, value in {**row, **env.extras.get('log', {})}.items():
            if isinstance(value, (int, float)) and key != 'update':
                writer.add_scalar(key if key.startswith('Reward/') else f'Train/{key}', value, update)
        if metrics_writer is None:
            family_columns = {f'train_{family}_{suffix}' for family in env.families for suffix in ('stood_2s', 'episodes')}
            columns = sorted((set(row) | set(TRAINING_SUMMARY_COLUMNS) | family_columns) - {'update'})
            metrics_writer = csv.DictWriter(metrics_file, fieldnames=['update', *columns], extrasaction='ignore')
            if metrics_file.tell() == 0:
                metrics_writer.writeheader()
        metrics_writer.writerow(row)
        metrics_file.flush()
        if update % 10 == 0 or args.smoke:
            print(f'GETUP_TRAIN: update {update} {row["transitions_per_s"]:.0f} transitions/s '
                  f'pull {state["pull_force"]:.0f} N bound {env.action_bound:.2f} stood_2s {train.get("stood_2s")} '
                  f'max_height {train.get("max_height_m")} arm_ready {train.get("arm_ready_success")} '
                  f'kl {stats["kl"]:.4f} std {stats["action_std"]:.3f} '
                  f'lr {stats["learning_rate"]:.2e}', flush=True)
        write_json(run_dir / 'status.json', dict(state='training', update=update, pull_force_n=state['pull_force'],
                                                 action_bound=env.action_bound, train=train, pid=os.getpid(),
                                                 time=time.time()))
        if update % args.save_interval == 0:
            save(run_dir / f'model_{update}.pt')
        if args.eval_interval and update % args.eval_interval == 0:
            report = evaluate(f'update_{update:06d}')
            for key, value in report.items():
                if isinstance(value, (int, float)) and key != 'update':
                    writer.add_scalar(f'Evaluation/{key}', value, update)
            print(f'GETUP_EVAL: update {update} stood_2s {report.get("stood_2s")} '
                  f'end {report.get("standing_at_end")} arm_ready {report.get("arm_ready_success")} '
                  f'final_arm_error {report.get("final_arm_error_rad")} ready {report.get("ready_success")} '
                  f'clean_new {report.get("clean_new_success")} handover_ready {report.get("handover_ready_success")} '
                  f'time_to_stand {report.get("first_standing_s")} peak_speed {report.get("peak_limb_speed")}', flush=True)
            if state['pull_force'] > 0 and report.get('stood_2s', 0.) >= args.pull_eval_success:
                state['pull_force'] = max(0., state['pull_force'] - args.pull_step)
                state['pull_changes'].append(dict(update=update, pull_force_n=state['pull_force'],
                                                  eval_stood_2s=report['stood_2s']))
                print(f'GETUP_CURRICULUM: update {update} pull force -> {state["pull_force"]:.1f} N', flush=True)
            writer.add_scalar('Evaluation/pull_force_after_n', state['pull_force'], update)
            if args.arm_pose_width > 0:
                # Stood up and settled into the ending pose first, then got up at all.
                score = (report.get('arm_ready_success', 0.), report.get('stood_2s', 0.),
                         report.get('clean_new_success', 0.))
            else:
                score = (report.get('handover_ready_success', 0.), report.get('clean_new_success', 0.),
                         report.get('stood_2s', 0.))
            if state['best'] is None or score > tuple(state['best']['score']):
                state['best'] = dict(update=update, score=score)
                save(run_dir / 'best.pt')
            save(run_dir / f'model_{update}.pt')
            set_curriculum()
            obs = reset_all()
            first_new_record = len(env.completed)

    save(run_dir / f'model_{state["iteration"]}.pt')
    write_json(run_dir / 'status.json', dict(state='stopped' if stop['requested'] else 'finished',
                                             update=state['iteration'], pull_force_n=state['pull_force'],
                                             best=state['best'], time=time.time()))
    print('GETUP_DONE ' + json.dumps(dict(update=state['iteration'], directory=str(run_dir))), flush=True)
    writer.close()
    metrics_file.close()


exit_code = 1
try:
    main()
    exit_code = 0
except BaseException:
    traceback.print_exc()
finally:
    # Kit can hang while shutting down after headless runs; checkpoints and logs are already written.
    os._exit(exit_code)
