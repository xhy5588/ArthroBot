"""Train, evaluate or watch the hybrid standing policy of the humanoid.

Modes:
  check     runtime checks of the environment (headless), no learning
  train     PPO from scratch (or resume with --checkpoint); runs go to logs/humanoid_standing/
  evaluate  deterministic trials of a checkpoint; reports the qualification (>= 90% of >= 16 trials stand)
  preview   watch a checkpoint in the GUI (default: the included checkpoint)

    python scripts/humanoid/train_standing.py --mode train --headless --num-envs 128 --iterations 5000 --eval-interval 500
    python scripts/humanoid/train_standing.py --mode evaluate --headless --num-envs 16 --seconds 30
    python scripts/humanoid/train_standing.py --mode preview
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from arthrobot import paths
from arthrobot_tasks.humanoid.standing.policy import CHECKPOINT as INCLUDED_CHECKPOINT

RUNS_DIR = paths.LOGS_DIR / 'humanoid_standing'
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--mode', choices=['check', 'train', 'evaluate', 'preview'], default='preview')
parser.add_argument('--checkpoint', type=Path, help='Resume (train) or load (evaluate/preview) this checkpoint.')
parser.add_argument('--num-envs', type=int)
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--eval-interval', type=int, default=0, help='Updates between held-out evaluations (0: none).')
parser.add_argument('--action-rate-weight', type=float, help='Action-change penalty (default 0.10, or the checkpoint value).')
parser.add_argument('--seconds', type=float, default=30., help='Evaluation length.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--render-every', type=int, default=1, help='Render once per this many control steps.')
parser.add_argument('--report', type=Path, help='Report path for check/evaluate.')
parser.add_argument('--screenshot', type=Path, help='Save a viewport image during visible training.')
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.render_every < 1 or args.iterations < 1 or args.eval_interval < 0 or args.seconds <= 0:
    parser.error('--render-every, --iterations and --seconds must be positive; --eval-interval nonnegative')
if args.mode in ('evaluate', 'preview') and args.checkpoint is None:
    args.checkpoint = INCLUDED_CHECKPOINT
if args.checkpoint and not args.checkpoint.is_file():
    parser.error(f'Missing checkpoint: {args.checkpoint}')
if args.eval_interval and args.mode != 'train':
    parser.error('--eval-interval applies to training only')
checkpoint_settings = {}
if args.checkpoint and (args.checkpoint.parent / 'settings.json').is_file():
    checkpoint_settings = json.loads((args.checkpoint.parent / 'settings.json').read_text())
    if checkpoint_settings.get('control_mode', 'hybrid') != 'hybrid':
        parser.error('This checkpoint uses another action meaning (not hybrid control)')
if args.action_rate_weight is None:
    args.action_rate_weight = checkpoint_settings.get('action_rate_weight', .10)
if args.mode == 'check':
    args.headless = True
if not args.headless:
    from arthrobot.sim.rtx_compat import enable
    enable()
app = AppLauncher(args, width=1600, height=1000).app


def write_report(data: dict, default_name: str) -> None:
    path = args.report or paths.build_dir('humanoid_training') / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')
    print('STANDING_REPORT: ' + json.dumps({key: value for key, value in data.items() if key != 'episodes'}), flush=True)


def make_env(model):
    from arthrobot_assets.humanoid.nominal_pose import solve_pose
    from arthrobot_tasks.humanoid.standing.env import HumanoidStandingEnv, HumanoidStandingEnvCfg
    cfg = HumanoidStandingEnvCfg()
    cfg.action_rate_weight = args.action_rate_weight
    cfg.set_nominal_pose(checkpoint_settings.get('nominal_pose') or solve_pose())
    for field, saved_name in (('position_scale', 'position_scale_rad'), ('position_kp', 'position_kp'),
                              ('position_kd', 'position_kd')):
        if saved_name in checkpoint_settings:
            setattr(cfg, field, checkpoint_settings[saved_name])
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.sim.render_interval = cfg.decimation * args.render_every
    cfg.scene.num_envs = args.num_envs or ((128 if args.headless else 8) if args.mode == 'train' else 4 if args.headless else 1)
    cfg.viewer.eye, cfg.viewer.lookat = (2.3, -2.3, 1.6), (.4, 0., .45)
    visible_training = args.mode == 'train' and not args.headless
    if visible_training:
        _render_only_first_robot(cfg)
    env = HumanoidStandingEnv(cfg)
    if visible_training:
        _show_first_robot_only(env, model)
    return env


def _render_only_first_robot(cfg):
    """Hide the source robot's visual meshes before cloning, so the renderer never loads 128 CAD copies."""
    cfg.robot.spawn.visible = False
    spawn = cfg.robot.spawn.func

    def spawn_without_visuals(*spawn_args, **spawn_kwargs):
        prim = spawn(*spawn_args, **spawn_kwargs)
        # Visibility alone still fills the render index; inactive visual subtrees do not.
        for body in prim.GetChildren():
            visuals = body.GetChild('visuals')
            if visuals.IsValid():
                visuals.SetActive(False)
        return prim

    cfg.robot.spawn.func = spawn_without_visuals
    cfg.viewer.origin_type, cfg.viewer.asset_name, cfg.viewer.env_index = 'asset_root', 'robot', 0
    cfg.viewer.eye, cfg.viewer.lookat = (2.3, -2.3, 1.1), (0., 0., -.08)


def _show_first_robot_only(env, model):
    from pxr import Gf, UsdGeom
    stage = env.scene.stage
    for path in env.scene.env_prim_paths[1:]:
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
        for name in model['body_masses']:   # author overrides before re-activating the source visuals
            visuals = stage.GetPrimAtPath(f'{path}/Robot/{name}/visuals')
            if visuals.IsValid():
                visuals.SetActive(False)
    for name in model['body_masses']:
        visuals = stage.GetPrimAtPath(f'/World/envs/env_0/Robot/{name}/visuals')
        if visuals.IsValid():
            visuals.SetActive(True)
    UsdGeom.Imageable(stage.GetPrimAtPath('/World/envs/env_0/Robot')).MakeVisible()
    floor = stage.GetPrimAtPath('/World/ground/Environment')
    if floor.IsValid():
        floor.GetAttribute('xformOp:scale').Set(Gf.Vec3d(10., 10., 1.))   # visual floor only
    grid = stage.GetPrimAtPath('/World/Grid')
    if grid.IsValid():
        UsdGeom.Xformable(grid).AddTranslateOp().Set(Gf.Vec3d(*env.scene.env_origins[0].cpu().tolist()))


def check_physics_setup(env, model):
    import torch
    from arthrobot_tasks.humanoid.standing.control import TORQUE_LIMIT_NM
    assert not env.robot.is_fixed_base
    assert torch.count_nonzero(env.robot.root_physx_view.get_dof_stiffnesses()) == 0
    assert torch.count_nonzero(env.robot.root_physx_view.get_dof_dampings()) == 0
    limits = env.robot.data.joint_effort_limits
    assert torch.allclose(limits, torch.full_like(limits, TORQUE_LIMIT_NM))
    mass = float(env.robot.root_physx_view.get_masses()[0].sum())
    assert abs(mass - model['total_mass_kg']) < 1e-4
    return mass


def main():
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from arthrobot_assets.humanoid import MOTOR_JOINTS
    from arthrobot_tasks.humanoid.assets import ensure_training_usd
    from arthrobot_tasks.humanoid.standing.agents import StandingPPORunnerCfg
    torch.set_num_threads(4)
    _, model = ensure_training_usd()
    env = make_env(model)
    observations, _ = env.reset()
    assert observations['policy'].shape == (env.num_envs, env.cfg.observation_space)
    mass = check_physics_setup(env, model)
    metadata = dict(task='standing', control_mode='hybrid', motor_action_order=list(MOTOR_JOINTS), actions=20,
                    observations=env.cfg.observation_space, action_rate_weight=args.action_rate_weight,
                    nominal_pose=env.cfg.nominal_pose, position_scale_rad=env.cfg.position_scale,
                    position_kp=env.cfg.position_kp, position_kd=env.cfg.position_kd, render_every=args.render_every,
                    grippers='parked open', motor_torque_limit_nm=13., motor_speed_limit_rpm=74., mass_kg=mass,
                    episode_seconds=env.cfg.episode_length_s, fall_tilt_deg=30., seed=args.seed)
    print('STANDING_READY: ' + json.dumps({key: metadata[key] for key in ('actions', 'observations', 'mass_kg')}),
          flush=True)
    if args.mode == 'check':
        from arthrobot_tasks.humanoid.standing.checks import check_hybrid_control
        write_report(dict(**metadata, **check_hybrid_control(env)), 'standing_check.json')
        env.close()
        return

    runner_cfg = StandingPPORunnerCfg()
    runner_cfg.seed, runner_cfg.device = args.seed, env.device
    wrapper = RslRlVecEnvWrapper(env, clip_actions=1.)
    run_dir = None
    if args.mode == 'train':
        run_dir = RUNS_DIR / datetime.now().strftime('%Y%m%d_%H%M%S')
        _save_run_settings(run_dir, metadata, runner_cfg, env)
    runner = OnPolicyRunner(wrapper, runner_cfg.to_dict(), log_dir=str(run_dir) if run_dir else None, device=env.device)
    if args.checkpoint:
        runner.load(str(args.checkpoint), load_optimizer=args.mode == 'train')
        if args.mode == 'train':
            runner.current_learning_iteration += 1   # the saved iteration is already complete
    if args.mode == 'train':
        train(runner, wrapper, env, run_dir)
    else:
        rows = run_policy(runner, wrapper, env)
        from arthrobot_tasks.humanoid.standing.evaluation import qualify_standing
        write_report(dict(**metadata, checkpoint=str(args.checkpoint), completed_trials=len(rows),
                          survived_trials=sum(row['survived'] for row in rows),
                          mean_episode_seconds=sum(row['seconds'] for row in rows) / max(1, len(rows)),
                          episodes=rows, qualification=qualify_standing(rows)), 'standing_evaluation.json')
    env.close()


def _save_run_settings(run_dir, metadata, runner_cfg, env):
    run_dir.mkdir(parents=True)
    source_dir = Path(__file__).resolve().parents[2] / 'source/arthrobot_tasks/humanoid/standing'
    sources = sorted(source_dir.glob('*.py')) + [Path(__file__).resolve()]
    (run_dir / 'settings.json').write_text(json.dumps(dict(
        **metadata, runner=runner_cfg.to_dict(), num_envs=env.num_envs, checkpoint=str(args.checkpoint),
        iterations_requested=args.iterations, eval_interval=args.eval_interval,
        total_training_transitions=args.iterations * env.num_envs * runner_cfg.num_steps_per_env,
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}), indent=2))
    (run_dir / 'source').mkdir()
    for path in sources:
        shutil.copy2(path, run_dir / 'source' / path.name)


def train(runner, wrapper, env, run_dir):
    from arthrobot_tasks.humanoid.standing.diagnostics import install_diagnostics
    from arthrobot_tasks.humanoid.standing.evaluation import evaluate_standing
    from arthrobot_tasks.humanoid.standing.monitor import install_monitor
    install_diagnostics(runner.alg)
    install_monitor(runner, env, run_dir)
    print(f'TRAINING_RUN: {run_dir}', flush=True)
    save_checkpoint = runner.save

    def save_and_point_latest(path, *save_args, **save_kwargs):
        save_checkpoint(path, *save_args, **save_kwargs)
        pointer = RUNS_DIR / 'latest.tmp'
        pointer.write_text(str(Path(path).resolve()) + '\n')
        pointer.replace(RUNS_DIR / 'latest.txt')

    runner.save = save_and_point_latest
    stop_requested = {'value': False}
    if not args.headless:
        _install_training_panel(runner, wrapper, env, stop_requested)
    stopped = False
    try:
        remaining = args.iterations
        while remaining:
            chunk = min(remaining, args.eval_interval or remaining)
            runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=False)
            remaining -= chunk
            completed = runner.current_learning_iteration + 1
            if args.eval_interval:
                print(f'STANDING_EVALUATION_START: update {completed}; learning paused', flush=True)
                evaluate_standing(runner, wrapper, env, run_dir, completed)
            if remaining:
                runner.current_learning_iteration += 1
    except _StopTraining:
        stopped = True
        runner.save(str(run_dir / f'model_{runner.current_learning_iteration}.pt'))
    latest = max(run_dir.glob('model_*.pt'), key=lambda path: int(path.stem.split('_')[-1]))
    print(f'{"TRAINING_STOPPED" if stopped else "TRAINING_COMPLETE"}: {latest}', flush=True)


class _StopTraining(Exception):
    pass


def _install_training_panel(runner, wrapper, env, stop_requested):
    """Live panel for visible training: robot 0's state and a save-and-stop button."""
    import omni.ui as ui
    import torch
    window = ui.Window('Live standing training', width=430, height=320)
    window.position_x, window.position_y = 1040, 120
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label(f'PPO standing | {env.num_envs} parallel robots', height=24)
            ui.Label('Watching robot 0 with exploration enabled.', word_wrap=True)
            ui.Label('18 joint targets + 2 wheel torques. Grippers open.', word_wrap=True)
            status = ui.Label('', height=140, word_wrap=True)
            ui.Button('Save and stop training', height=28, clicked_fn=lambda: stop_requested.update(value=True))
    step_env = wrapper.step
    steps = {'count': 0}

    def visible_step(actions):
        if stop_requested['value']:
            raise _StopTraining()
        result = step_env(actions)
        steps['count'] += 1
        if steps['count'] % 12 == 0:
            _, velocity = env.com_state()
            tilt = torch.acos((-env.robot.data.projected_gravity_b[0, 2]).clamp(-1., 1.)) * 180 / torch.pi
            standing = runner.standing_status
            status.text = (f'Update: {runner.current_learning_iteration}\n'
                           f'Robot 0 speed: {float(velocity[0, 0]):.3f} m/s\n'
                           f'Torso tilt: {float(tilt):.1f} degrees\n'
                           f'Full trials survived: {env.total_successes}/{env.total_resets}\n'
                           f'Total PPO loss: {standing.get("total_loss", float("nan")):.4f}\n'
                           f'100-update mean: {standing.get("total_loss_mean_100", float("nan")):.4f}')
        if args.screenshot and steps['count'] == 60:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            capture_viewport_to_file(get_active_viewport(), str(args.screenshot.resolve()))
        return result

    wrapper.step = visible_step
    print('LIVE_TRAINING_PANEL_READY: watching robot 0; policy updates are active', flush=True)


def run_policy(runner, wrapper, env):
    """Run the deterministic policy (evaluate: for --seconds; preview: until the window closes)."""
    import torch
    policy = runner.get_inference_policy(device=env.device)
    label = None
    if not args.headless:
        import omni.ui as ui
        window = ui.Window('Standing policy', width=410, height=200)
        with window.frame:
            with ui.VStack(spacing=8):
                ui.Label('18 learned joint targets + 2 wheel torques. Grippers open.', word_wrap=True)
                ui.Button('Reset robot', clicked_fn=lambda: env.reset())
                label = ui.Label('', word_wrap=True)
    with torch.inference_mode():
        step = 0
        while app.is_running() and (args.mode == 'preview' or step < int(args.seconds / env.step_dt)):
            start = time.monotonic()
            observations, reward, *_ = env.step(policy(wrapper.get_observations()))
            assert torch.isfinite(observations['policy']).all() and torch.isfinite(reward).all()
            if label is not None and step % 6 == 0:
                _, velocity = env.com_state()
                tilt = torch.acos((-env.robot.data.projected_gravity_b[0, 2]).clamp(-1., 1.)) * 180 / torch.pi
                label.text = (f'Forward speed: {float(velocity[0, 0]):.3f} m/s\nTorso tilt: {float(tilt):.1f} degrees\n'
                              f'Completed trials: {env.total_resets}')
                x, y, _ = env.robot.data.root_pos_w[0].cpu().tolist()
                env.sim.set_camera_view(eye=(x + 2.3, y - 2.3, 1.6), target=(x, y, .45))
            if not args.headless:
                time.sleep(max(0., env.step_dt - (time.monotonic() - start)))
            step += 1
    return env.completed_episode_records


try:
    main()
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    app.close()
