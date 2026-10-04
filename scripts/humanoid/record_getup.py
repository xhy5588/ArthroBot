"""Record one get-up rollout per lying family to an MP4 (needs rendering; see the README's RTX note).

One robot per start family (all five by default), each from the first held-out pose
of its family, so clips are comparable across checkpoints. Evaluation conditions:
deterministic policy, no pull force, no randomization or observation noise, the
checkpoint's action bound. Each robot has its own tracking camera. By default every
tile shows live measurements and an extra tile shows the run information; --plain
keeps only the family names (for presentation clips). Optionally adds an animated
GIF and per-family scalars to TensorBoard.

    python scripts/humanoid/record_getup.py
    python scripts/humanoid/record_getup.py --plain --families front --tile-width 800 --tile-height 450 --seconds 7
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

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
parser.add_argument('--pose-bank', type=Path, default=POSE_BANK_DIR / 'held_out_seed1042_32.json')
parser.add_argument('--output-dir', type=Path, default=paths.BUILD_DIR / 'humanoid_training/getup_videos')
parser.add_argument('--tensorboard-dir', type=Path, help='Also write an animated GIF and scalars here.')
parser.add_argument('--families', nargs='+', help='Start families to record (default: every family in the bank).')
parser.add_argument('--plain', action='store_true', help='Only family names on the tiles; no measurements.')
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
args.headless, args.enable_cameras = True, True
from arthrobot.sim.rtx_compat import enable  # noqa: E402

enable()
app = AppLauncher(args).app

GRID_COLUMNS = 3


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
    from arthrobot_tasks.humanoid.getup.env import GetupEnv, GetupEnvCfg
    from arthrobot_tasks.humanoid.getup.evaluation import action_bound_at, load_getup_policy

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
    first_pose = [next(i for i, pose in enumerate(bank['poses']) if pose['family'] == family) for family in families]
    width, height = args.tile_width, args.tile_height
    usd_path, _ = ensure_getup_usd(with_visuals=True)

    cfg = GetupEnvCfg()
    cfg.asset_path, cfg.pose_bank = str(usd_path), str(args.pose_bank)
    cfg.scene.num_envs, cfg.scene.env_spacing, cfg.sim.device = len(families), 20., args.device  # no other robot in view
    cfg.randomize, cfg.obs_noise = False, False
    camera_cfg = TiledCameraCfg(prim_path='/World/envs/env_.*/Camera', data_types=['rgb'], width=width, height=height,
                                offset=TiledCameraCfg.OffsetCfg(pos=(1.8, -1.8, 1.), rot=(1., 0., 0., 0.),
                                                                convention='world'),
                                spawn=sim_utils.PinholeCameraCfg(focal_length=16., clipping_range=(.05, 40.)))

    class RecordingEnv(GetupEnv):
        def _add_sensors(self):
            # One camera per environment, created before the environments are cloned.
            self.camera = TiledCamera(camera_cfg)
            self.scene.sensors['camera'] = self.camera

    env = RecordingEnv(cfg)
    env.action_bound, env.pull_force_n = action_bound, 0.
    env.forced_pose = torch.tensor(first_pose, device=env.device)
    obs, _ = env.reset()
    camera_offset = torch.tensor(args.camera_offset, device=env.device)

    def aim_cameras():
        target = env.robot.data.root_pos_w.clone()
        target[:, 2] = env.scene.env_origins[:, 2] + .3
        env.camera.set_world_poses_from_view(target + camera_offset, target)

    def robot_tile(image, family, torso_height, tilt, standing_s, contacts, contact_s):
        tile = np.ascontiguousarray(image)
        cv2.rectangle(tile, (0, 0), (width, 44), (0, 0, 0), -1)
        cv2.putText(tile, f'{family}   h {torso_height:.2f} m   tilt {tilt:.0f} deg', (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, f'STANDING {standing_s:.1f} s' if standing_s > 0 else 'not standing', (8, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (60, 220, 60) if standing_s > 0 else (200, 200, 200), 1, cv2.LINE_AA)
        cv2.rectangle(tile, (0, height - 24), (width, height), (0, 0, 0), -1)
        touch = 'SELF-CONTACT: ' + ', '.join(contacts[:3]) if contacts else 'no self-contact'
        cv2.putText(tile, f'{touch}  (total {contact_s:.2f} s)', (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (255, 80, 80) if contacts else (160, 160, 160), 1, cv2.LINE_AA)
        return tile

    def plain_tile(image, family):
        tile = np.ascontiguousarray(image)
        cv2.putText(tile, family, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, family, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
        return tile

    def caption_tile():
        tile = np.full((height, width, 3), 24, np.uint8)
        lines = ['ArthroBot humanoid, v0', f'get-up policy, update {update:,}', 'deterministic, no assistance',
                 'held-out start poses']
        for row, line in enumerate(lines):
            cv2.putText(tile, line, (16, height // 2 - 45 + 32 * row), cv2.FONT_HERSHEY_SIMPLEX, .65, (230, 230, 230),
                        1, cv2.LINE_AA)
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
            obs, *_ = env.step(model.mean(obs['policy']).clamp(-1., 1.))
            max_height = torch.maximum(max_height, env.torso_height())
            best_standing = torch.maximum(best_standing, env.standing_time)
            if step % frame_stride:
                continue
            images = env.camera.data.output['rgb'][..., :3].cpu().numpy().astype(np.uint8)
            if args.plain:
                tiles = [plain_tile(images[i], family) for i, family in enumerate(families)]
                frames.append(grid(tiles + [caption_tile()] if len(tiles) > 1 else tiles))
                continue
            tilt = torch.rad2deg(torch.acos((-env.robot.data.projected_gravity_b[:, 2]).clamp(-1., 1.))).tolist()
            touching = ((env.self_contact_forces() > env.cfg.self_contact_n) & env.actuated()[:, None]).cpu()
            contacts = [[short_body_name(env.contacts.body_names[j]) for j in torch.nonzero(row).flatten().tolist()]
                        for row in touching]
            tiles = [robot_tile(images[i], family, float(env.torso_height()[i]), tilt[i], float(env.standing_time[i]),
                                contacts[i], float(env.self_contact_time[i])) for i, family in enumerate(families)]
            tiles.append(info_tile((step + 1) * env.step_dt))
            frames.append(grid(tiles))

    summary = dict(update=update, checkpoint=str(args.checkpoint), action_bound=action_bound, pull_force_n=0.,
                   training_pull_force_n=state['pull_force'], frames=len(frames), fps=args.fps,
                   families={family: dict(pose_index=first_pose[i], best_standing_s=float(best_standing[i]),
                                          stood_2s=float(best_standing[i]) >= 2., max_height_m=float(max_height[i]),
                                          self_contact_s=float(env.self_contact_time[i]))
                             for i, family in enumerate(families)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = f'update_{update:06d}' + ('_' + '_'.join(args.families) if args.families else '') + (
        '_plain' if args.plain else '')
    video = args.output_dir / f'{name}.mp4'
    imageio.mimwrite(video, frames, fps=args.fps, codec='libx264', quality=7, macro_block_size=8)
    (args.output_dir / f'{name}.json').write_text(json.dumps(summary, indent=2) + '\n')
    if args.tensorboard_dir:
        write_tensorboard(frames, summary, update)
    print('GETUP_ROLLOUT: ' + json.dumps(dict(video=str(video), **{key: value for key, value in summary.items()
                                                                     if key != 'families'},
                                              stood=[family for family, values in summary['families'].items()
                                                     if values['stood_2s']])), flush=True)


def write_tensorboard(frames, summary: dict, update: int) -> None:
    """An animated GIF image summary (what add_video produces) plus per-family scalars."""
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
    for family, values in summary['families'].items():
        for key in ('best_standing_s', 'max_height_m', 'self_contact_s'):
            writer.add_scalar(f'Rollout/{family}_{key}', values[key], update)
    writer.add_scalar('Rollout/families_stood_2s', sum(values['stood_2s'] for values in summary['families'].values()),
                      update)
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
