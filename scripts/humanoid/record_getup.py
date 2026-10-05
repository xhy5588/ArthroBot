"""Record get-up rollouts to an MP4 (needs rendering; see the README's RTX note).

Robots start from the first held-out poses of each start family (all five families and
one pose each by default), so clips are comparable across checkpoints. Evaluation
conditions: deterministic policy, no pull force, no randomization or observation
noise, the checkpoint's action bound. Each robot has its own tracking camera.

- By default every tile shows live measurements and an extra tile shows the run
  information; --plain keeps only the labels (for presentation clips).
- --handover switches each robot to the standing policy once it has stood for
  --stand-hold seconds, as in handoff.py. The standing policy brings the limbs back to
  the standing pose. The summary reports, per robot, whether it stayed standing and
  whether it ended in the standing pose.
- --tensorboard-dir also adds an animated GIF and per-robot scalars to TensorBoard.

Memory: every rendered robot carries all 1,750 visual meshes, about 1.5 GB of host
memory each. On a 32 GB machine keep a run to about 5-8 robots (15 were killed by
the out-of-memory killer); record more poses with one --families value per run.

    python scripts/humanoid/record_getup.py
    python scripts/humanoid/record_getup.py --plain --families front --tile-width 800 --tile-height 450 --seconds 7
    python scripts/humanoid/record_getup.py --plain --handover --poses-per-family 3 --seconds 12
    python scripts/humanoid/record_getup.py --checkpoint logs/humanoid_getup/<run>/model_6000.pt --tensorboard-dir logs/humanoid_getup/<run>/rollouts
"""
import argparse
import io
import json
import os
from pathlib import Path
import traceback

from arthrobot import paths
from arthrobot_tasks.humanoid.getup.evaluation import CHECKPOINT
from arthrobot_tasks.humanoid.getup.poses import POSE_BANK_DIR
from arthrobot_tasks.humanoid.standing.policy import CHECKPOINT as STANDING_CHECKPOINT

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
parser.add_argument('--pose-bank', type=Path, default=POSE_BANK_DIR / 'held_out_seed1042_32.json')
parser.add_argument('--output-dir', type=Path, default=paths.BUILD_DIR / 'humanoid_training/getup_videos')
parser.add_argument('--tensorboard-dir', type=Path, help='Also write an animated GIF and scalars here.')
parser.add_argument('--families', nargs='+', help='Start families to record (default: every family in the bank).')
parser.add_argument('--poses-per-family', type=int, default=1, help='Record the first N held-out poses of each family.')
parser.add_argument('--plain', action='store_true', help='Only labels on the tiles; no measurements.')
parser.add_argument('--handover', action='store_true', help='Switch to the standing policy after standing.')
parser.add_argument('--standing', type=Path, default=STANDING_CHECKPOINT, help='Standing checkpoint for --handover.')
parser.add_argument('--stand-hold', type=float, default=1., help='Seconds of standing before the hand-over.')
parser.add_argument('--seconds', type=float, help='Clip length (default: the whole 10 s episode).')
parser.add_argument('--camera-offset', type=float, nargs=3, default=(1.25, -1.25, .7), metavar=('X', 'Y', 'Z'),
                    help='Camera position relative to the torso, m.')
parser.add_argument('--tile-width', type=int, default=400)
parser.add_argument('--tile-height', type=int, default=300)
parser.add_argument('--fps', type=int, default=15)
parser.add_argument('--gif-scale', type=float, default=.5, help='GIF size relative to the MP4.')
parser.add_argument('--gif-fps', type=int, default=8)
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.poses_per_family < 1:
    parser.error('--poses-per-family must be at least 1')
args.headless, args.enable_cameras = True, True
from arthrobot.sim.rtx_compat import enable  # noqa: E402

enable()
app = AppLauncher(args).app

GRID_COLUMNS = 3
STANDING_POLICY_COLOR = (60, 220, 60)


def short_body_name(name: str) -> str:
    return name.replace('_body', '').replace('_arm_6', ' gripper').replace('_', ' ')


def main():
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCamera, TiledCameraCfg
    from arthrobot_tasks.humanoid.assets import ensure_getup_usd
    from arthrobot_tasks.humanoid.getup.env import GetupEnvCfg
    from arthrobot_tasks.humanoid.getup.evaluation import action_bound_at, load_getup_policy
    from arthrobot_tasks.humanoid.getup.handover import HandoverEnv, StandingTracker
    from arthrobot_tasks.humanoid.standing.policy import load_standing_actor

    model, settings, state = load_getup_policy(args.checkpoint, args.device)
    update = int(state['iteration'])
    action_bound = action_bound_at(update, settings['final_action_bound'], settings['bound_iterations'])
    bank = json.loads(args.pose_bank.read_text())
    families = [family for family in bank['families'] if any(pose['family'] == family for pose in bank['poses'])]
    if args.families:
        unknown = sorted(set(args.families) - set(families))
        if unknown:
            raise ValueError(f'Families {unknown} are not in {args.pose_bank.name}; available: {families}')
        families = list(args.families)
    robots = []   # (label, family, bank row), in tile order
    for family in families:
        rows = [index for index, pose in enumerate(bank['poses']) if pose['family'] == family][:args.poses_per_family]
        robots += [(family if args.poses_per_family == 1 else f'{family} {number + 1}', family, row)
                   for number, row in enumerate(rows)]
    width, height = args.tile_width, args.tile_height
    usd_path, _ = ensure_getup_usd(with_visuals=True)

    cfg = GetupEnvCfg()
    cfg.asset_path, cfg.pose_bank = str(usd_path), str(args.pose_bank)
    cfg.scene.num_envs, cfg.scene.env_spacing, cfg.sim.device = len(robots), 20., args.device  # no other robot in view
    cfg.randomize, cfg.obs_noise = False, False
    if args.seconds:
        cfg.episode_length_s = max(cfg.episode_length_s, args.seconds + 1.)
    camera_cfg = TiledCameraCfg(prim_path='/World/envs/env_.*/Camera', data_types=['rgb'], width=width, height=height,
                                offset=TiledCameraCfg.OffsetCfg(pos=(1.8, -1.8, 1.), rot=(1., 0., 0., 0.),
                                                                convention='world'),
                                spawn=sim_utils.PinholeCameraCfg(focal_length=16., clipping_range=(.05, 40.)))

    class RecordingEnv(HandoverEnv):
        """Get-up environment with a camera per robot. Without hand-overs it behaves exactly as GetupEnv."""

        def _add_sensors(self):
            # One camera per environment, created before the environments are cloned.
            self.camera = TiledCamera(camera_cfg)
            self.scene.sensors['camera'] = self.camera

    env = RecordingEnv(cfg)
    env.action_bound, env.pull_force_n = action_bound, 0.
    env.forced_pose = torch.tensor([row for _, _, row in robots], device=env.device)
    standing_policy = load_standing_actor(args.standing, env.device) if args.handover else None
    obs, _ = env.reset()
    camera_offset = torch.tensor(args.camera_offset, device=env.device)
    handover_time = torch.full((env.num_envs,), float('nan'), device=env.device)
    after_handover = StandingTracker(env)

    def aim_cameras():
        target = env.robot.data.root_pos_w.clone()
        target[:, 2] = env.scene.env_origins[:, 2] + .3
        env.camera.set_world_poses_from_view(target + camera_offset, target)

    def policy_name(index):
        return 'standing policy' if bool(env.standing_mode[index]) else 'get-up policy'

    def robot_tile(image, index, label, torso_height, tilt, standing_s, contacts, contact_s):
        tile = np.ascontiguousarray(image)
        cv2.rectangle(tile, (0, 0), (width, 44), (0, 0, 0), -1)
        cv2.putText(tile, f'{label}   h {torso_height:.2f} m   tilt {tilt:.0f} deg', (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        status = f'STANDING {standing_s:.1f} s' if standing_s > 0 else 'not standing'
        if args.handover:
            status += f'   {policy_name(index)}'
        cv2.putText(tile, status, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (60, 220, 60) if standing_s > 0 else (200, 200, 200), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (0, height - 24), (width, height), (0, 0, 0), -1)
        touch = 'SELF-CONTACT: ' + ', '.join(contacts[:3]) if contacts else 'no self-contact'
        cv2.putText(tile, f'{touch}  (total {contact_s:.2f} s)', (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (255, 80, 80) if contacts else (160, 160, 160), 1, cv2.LINE_AA)
        return tile

    def outlined_text(tile, text, origin, scale, color):
        cv2.putText(tile, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    def plain_tile(image, index, label):
        tile = np.ascontiguousarray(image)
        outlined_text(tile, label, (12, 30), .8, (255, 255, 255))
        if args.handover:
            standing = bool(env.standing_mode[index])
            outlined_text(tile, policy_name(index), (12, 58), .6, STANDING_POLICY_COLOR if standing else (255, 255, 255))
        return tile

    def caption_tile():
        tile = np.full((height, width, 3), 24, np.uint8)
        lines = ['ArthroBot humanoid, v0', f'get-up policy, update {update:,}']
        if args.handover:
            lines.append(f'-> standing policy after {args.stand_hold:g} s up')
        lines += ['deterministic, no assistance', 'held-out start poses']
        for row, line in enumerate(lines):
            cv2.putText(tile, line, (16, height // 2 - 16 * len(lines) + 32 * row), cv2.FONT_HERSHEY_SIMPLEX, .65,
                        (230, 230, 230), 1, cv2.LINE_AA)
        return tile

    def grid(tiles):
        columns = min(GRID_COLUMNS, len(tiles))
        tiles = tiles + [np.full_like(tiles[0], 24)] * (-len(tiles) % columns)
        rows = [np.hstack(tiles[start:start + columns]) for start in range(0, len(tiles), columns)]
        return np.vstack(rows)

    def info_tile(time_s):
        tile = np.full((height, width, 3), 24, np.uint8)
        lines = [f'update {update}', f't = {time_s:4.1f} s' + ('  (motors off)' if time_s < cfg.unactuated_s else ''),
                 f'action bound {action_bound:.2f} rad', 'pull force 0 N, deterministic',
                 f'training pull force {state["pull_force"]:.0f} N', 'success = stand 2 s',
                 '(h > .46 m, tilt < 15, wheels down)']
        for row, line in enumerate(lines):
            cv2.putText(tile, line, (12, 30 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 230, 230), 1, cv2.LINE_AA)
        return tile

    aim_cameras()
    for _ in range(8):   # renderer warm-up, so that the first frames are not black
        env.sim.render()
    frame_stride = max(1, round(1 / (env.step_dt * args.fps)))
    frames = []
    max_height = torch.zeros(env.num_envs, device=env.device)
    best_standing = torch.zeros(env.num_envs, device=env.device)
    # The timeout fires when the episode counter reaches max - 1; stopping one step earlier
    # keeps the final frames (a reset would show the start pose again).
    steps = int(env.max_episode_length) - 2
    if args.seconds:
        steps = min(steps, round(args.seconds / env.step_dt))
    with torch.no_grad():
        for step in range(steps):
            aim_cameras()
            actions = model.mean(obs['policy']).clamp(-1., 1.)
            if args.handover:
                actions = torch.where(env.standing_mode[:, None],
                                      standing_policy(env.standing_policy_observation()), actions)
            obs, *_ = env.step(actions)
            max_height = torch.maximum(max_height, env.torso_height())
            best_standing = torch.maximum(best_standing, env.standing_time)
            if args.handover:
                new = ~env.standing_mode & (env.standing_time >= args.stand_hold)
                if new.any():
                    ids = new.nonzero().flatten()
                    env.hand_over(ids)
                    handover_time[ids] = step * env.step_dt
                after_handover.update(env.standing_mode)
            if step % frame_stride:
                continue
            images = env.camera.data.output['rgb'][..., :3].cpu().numpy().astype(np.uint8)
            if args.plain:
                tiles = [plain_tile(images[i], i, label) for i, (label, _, _) in enumerate(robots)]
                frames.append(grid(tiles + [caption_tile()] if len(tiles) > 1 else tiles))
                continue
            tilt = env.tilt_deg().tolist()
            touching = ((env.self_contact_forces() > env.cfg.self_contact_n) & env.actuated()[:, None]).cpu()
            contacts = [[short_body_name(env.contacts.body_names[j]) for j in torch.nonzero(row).flatten().tolist()]
                        for row in touching]
            tiles = [robot_tile(images[i], i, label, float(env.torso_height()[i]), tilt[i], float(env.standing_time[i]),
                                contacts[i], float(env.self_contact_time[i])) for i, (label, _, _) in enumerate(robots)]
            tiles.append(info_tile((step + 1) * env.step_dt))
            frames.append(grid(tiles))

    in_standing_pose = env.ready_pose() & env.standing()
    stayed_standing = after_handover.stayed_standing()
    extra_tile = not args.plain or len(robots) > 1   # the info or caption tile
    columns = min(GRID_COLUMNS, len(robots) + extra_tile)
    summary = dict(update=update, checkpoint=str(args.checkpoint), action_bound=action_bound, pull_force_n=0.,
                   training_pull_force_n=state['pull_force'], frames=len(frames), fps=args.fps,
                   tile_size_px=[width, height], handover=args.handover,
                   standing_checkpoint=str(args.standing) if args.handover else None,
                   robots=[dict(label=label, family=family, pose_index=row, tile_origin_px=[
                                    width * (i % columns), height * (i // columns)],
                                best_standing_s=float(best_standing[i]), stood_2s=float(best_standing[i]) >= 2.,
                                max_height_m=float(max_height[i]), self_contact_s=float(env.self_contact_time[i]),
                                handover_s=None if torch.isnan(handover_time[i]) else float(handover_time[i]),
                                stayed_standing_after_handover=bool(stayed_standing[i]) if args.handover else None,
                                in_standing_pose_at_end=bool(in_standing_pose[i]))
                           for i, (label, family, row) in enumerate(robots)])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = (f'update_{update:06d}' + ('_' + '_'.join(args.families) if args.families else '')
            + (f'_x{args.poses_per_family}' if args.poses_per_family > 1 else '')
            + ('_handover' if args.handover else '') + ('_plain' if args.plain else ''))
    video = args.output_dir / f'{name}.mp4'
    imageio.mimwrite(video, frames, fps=args.fps, codec='libx264', quality=7, macro_block_size=8)
    (args.output_dir / f'{name}.json').write_text(json.dumps(summary, indent=2) + '\n')
    if args.tensorboard_dir:
        write_tensorboard(frames, summary, update)
    print('GETUP_ROLLOUT: ' + json.dumps(dict(
        video=str(video), **{key: value for key, value in summary.items() if key != 'robots'},
        stood=[robot['label'] for robot in summary['robots'] if robot['stood_2s']],
        in_standing_pose=[robot['label'] for robot in summary['robots'] if robot['in_standing_pose_at_end']])),
        flush=True)


def write_tensorboard(frames, summary: dict, update: int) -> None:
    """An animated GIF image summary (what add_video produces) plus per-robot scalars."""
    from PIL import Image
    from tensorboard.compat.proto.summary_pb2 import Summary
    from torch.utils.tensorboard import SummaryWriter
    keep_every = max(1, round(args.fps / args.gif_fps))
    small = [Image.fromarray(frame).resize((int(frame.shape[1] * args.gif_scale), int(frame.shape[0] * args.gif_scale)))
             for frame in frames[::keep_every]]
    buffer = io.BytesIO()
    small[0].save(buffer, format='GIF', save_all=True, append_images=small[1:],
                  duration=int(1000 * keep_every / args.fps), loop=0, optimize=True)
    image = Summary.Image(height=small[0].height, width=small[0].width, colorspace=3,
                          encoded_image_string=buffer.getvalue())
    writer = SummaryWriter(str(args.tensorboard_dir))
    writer._get_file_writer().add_summary(Summary(value=[Summary.Value(tag='Rollout/held_out_families', image=image)]),
                                          update)
    for robot in summary['robots']:
        tag = robot['label'].replace(' ', '_')
        for key in ('best_standing_s', 'max_height_m', 'self_contact_s'):
            writer.add_scalar(f'Rollout/{tag}_{key}', robot[key], update)
    writer.add_scalar('Rollout/families_stood_2s', sum(robot['stood_2s'] for robot in summary['robots']), update)
    writer.close()


exit_code = 1
try:
    main()
    exit_code = 0
except BaseException:
    traceback.print_exc()
finally:
    # Kit can hang while shutting down after headless runs; the video is already written.
    os._exit(exit_code)
