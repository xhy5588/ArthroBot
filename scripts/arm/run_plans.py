"""Replay planned trajectories on simulated arms with a physical table and wall; count contacts.

    python scripts/arm/run_plans.py
    python scripts/arm/run_plans.py --arms rigid:curobo,loose:curobo:none,loose:curobo:model \
        --levels typical_bracket,loose_bracket,very_loose_bracket --hold 2

Reads a plans file from scripts/arm/plan_trajectories.py. --arms lists the arms in
every environment as model:planner[:correction], where model is 'rigid' (the reach
task's arm) or 'loose' (arthrobot_tasks.arm_planning.loose) and planner is
'ik_joint' (IK + minimum-jerk joint interpolation) or 'curobo'. Each arm gets its
own table and wall (arthrobot_tasks.arm_planning.obstacles). With loose arms, environments cover every --levels entry
for every target sequence; all arms of an environment replay the same sequence.

Servo targets are the planned joint angles offset by gravity torque / servo
stiffness, as in arthrobot_tasks.arm_planning.controllers. Corrections (--correct, or per arm
as model:planner:correction):
  camera  during the --hold after each move, add the measured TCP and gripper-rotation
          error (as from a camera) into the IK goal
  track   a camera all along the path: the camera's gripper pose minus the pose the motor
          encoders imply is the deflection no encoder sees; add it back through the Jacobian
  model   no sensor: compensate the gravity sag of the play and bracket bending of each
          gravity-loaded joint, knowing the arm's looseness (as after measuring it)
Contact sensors record any touch, with the table, the wall or the arm itself,
during the move and during the hold.

Writes <out>/<name>.json; with --video also <out>/<name>.mp4 (keep the environment count small).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

from arthrobot import paths

OUT_DIR = paths.LOGS_DIR / 'arm_planning'
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--plans', type=Path, default=OUT_DIR / 'plans.npz')
parser.add_argument('--arms', default='rigid:ik_joint,rigid:curobo')
parser.add_argument('--levels', default='', help='Looseness levels for loose arms (arm_planning.loose.LEVELS).')
parser.add_argument('--correct', choices=('none', 'camera', 'track', 'model'), default='none',
                    help='Default correction; an arm can override it as model:planner:correction.')
parser.add_argument('--hold', type=float, default=1.0, help='Seconds to hold each target after the move.')
parser.add_argument('--sequences', type=int, default=0, help='Replay only N target sequences (0: all).')
parser.add_argument('--first-sequence', type=int, default=0, help='Index of the first sequence to replay.')
parser.add_argument('--targets', type=int, default=0, help='Replay only the first N targets of each sequence (0: all).')
parser.add_argument('--physics-hz', type=float, default=480.)
parser.add_argument('--out', type=Path, default=OUT_DIR)
parser.add_argument('--name', default='run')
parser.add_argument('--video', action='store_true')
parser.add_argument('--camera', choices=('wide', 'close', 'high'), default='wide', help='Video framing.')
parser.add_argument('--inset', action='store_true', help='Video: add a 2.5x zoom around the target.')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.video:
    args.enable_cameras = True
    from arthrobot.sim.rtx_compat import enable  # Isaac Sim 5.1's RTX renderer needs this on NVIDIA 595.x.
    enable()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402

from arthrobot_tasks.arm_planning.controllers import CAMERA_GAIN, SERVO_STIFFNESS  # noqa: E402
from arthrobot_tasks.arm_planning.kinematics import ArmModel, rotation_exp, rotation_log  # noqa: E402
from arthrobot_tasks.arm_planning.loose import LEVELS, apply_looseness, loose_arm_cfg  # noqa: E402
from arthrobot_tasks.arm_planning.obstacles import cuboids  # noqa: E402
from arthrobot_tasks.arm_planning.scene import MountFrame  # noqa: E402
from arthrobot_tasks.arm_reach.env_cfg import ARM_CFG, GROUND_Z, PEDESTAL_TOP  # noqa: E402
from arthrobot_tasks.arm_reach.mount import BASE_POSITION  # noqa: E402

PLANNER_LABELS = {'ik_joint': 'IK + joint interpolation', 'curobo': 'cuRobo'}
CORRECTION_LABELS = {'none': '', 'camera': ' + camera at target', 'track': ' + camera tracking',
                     'model': ' + measured-sag compensation'}
TRACK_FILTER_S = 0.2  # s, low-pass time constant of the measured deflection.
LOADED_TORQUE = 0.3  # N m: joints with less gravity torque float in their play and are not compensated.
ARM_OFFSET_Y = 2.0  # m between the arms of an environment, so their scenes do not overlap.
CONTACT_N = 0.05
FLIP = torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.float64))  # Same grasp, turned 180 deg about z.


def arm_specs():
    """[(name, model, planner, correction)] from --arms."""
    specs = []
    for i, item in enumerate(args.arms.split(',')):
        model, planner, *rest = item.split(':')
        correction = rest[0] if rest else args.correct
        assert model in ('rigid', 'loose') and planner in PLANNER_LABELS and correction in CORRECTION_LABELS, item
        specs.append((f'{model}_{planner}_{correction}_{i}', model, planner, correction))
    return specs


def build_scene_cfg(num_envs, specs, video):
    cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.0 + ARM_OFFSET_Y * len(specs))
    cfg.ground = AssetBaseCfg(prim_path='/World/ground', spawn=sim_utils.GroundPlaneCfg(),
                              init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, GROUND_Z)))
    cfg.light = AssetBaseCfg(prim_path='/World/light',
                             spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0))
    for a, (name, model, _, _) in enumerate(specs):
        y = a * ARM_OFFSET_Y
        position = (BASE_POSITION[0], BASE_POSITION[1] + y, BASE_POSITION[2])
        if model == 'rigid':
            robot = ARM_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot_' + name,
                                          init_state=ARM_CFG.init_state.replace(pos=position))
        else:
            robot = loose_arm_cfg('{ENV_REGEX_NS}/Robot_' + name, position)
        robot.spawn = robot.spawn.replace(activate_contact_sensors=True)
        setattr(cfg, f'robot_{name}', robot)
        setattr(cfg, f'contacts_{name}', ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot_' + name + '/.*'))
        setattr(cfg, f'pedestal_{name}', AssetBaseCfg(
            prim_path='{ENV_REGEX_NS}/Pedestal_' + name,
            spawn=sim_utils.CylinderCfg(radius=0.05, height=PEDESTAL_TOP - GROUND_Z,
                                        collision_props=sim_utils.CollisionPropertiesCfg(),
                                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38))),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, y, (PEDESTAL_TOP + GROUND_Z) / 2))))
        for obstacle, (center, dims) in cuboids().items():
            color = (0.55, 0.42, 0.3) if obstacle.startswith('table') else (0.3, 0.45, 0.75)
            setattr(cfg, f'{obstacle}_{name}', AssetBaseCfg(
                prim_path='{ENV_REGEX_NS}/' + f'{obstacle}_{name}',
                spawn=sim_utils.CuboidCfg(size=dims, collision_props=sim_utils.CollisionPropertiesCfg(),
                                          visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(center[0], center[1] + y, center[2]))))
        if video:
            from isaaclab.sensors import TiledCameraCfg
            setattr(cfg, f'camera_{name}', TiledCameraCfg(
                prim_path='{ENV_REGEX_NS}/Camera_' + name, data_types=['rgb'], width=800, height=600,
                update_latest_camera_pose=True,
                spawn=sim_utils.PinholeCameraCfg(focal_length=20., clipping_range=(0.05, 30.))))
    return cfg


class HoldCorrector:
    """During the hold: integrate the true TCP and gripper-rotation error into the pose IK goal."""

    def __init__(self, model, num_envs, device):
        self.model = model
        z = lambda *shape: torch.zeros(num_envs, *shape, dtype=torch.float64, device=device)
        self.bias, self.rotation_bias, self.q_goal = z(3), z(3), z(6)
        self.target, self.target_rot = z(3), z(3, 3)

    def start(self, q_final, target, target_rot):
        """q_final: the plan's last joint angles; target_rot: the grasp variant the plan reached."""
        self.q_goal, self.target, self.target_rot = q_final.double(), target.double(), target_rot
        self.bias.zero_()
        self.rotation_bias.zero_()

    def step(self, true_tcp, true_rot, dt):
        self.bias += CAMERA_GAIN * dt * (self.target - true_tcp.double())
        error = rotation_log(self.target_rot @ true_rot.double().transpose(1, 2))
        self.rotation_bias += CAMERA_GAIN * dt * error
        goal_rot = rotation_exp(self.rotation_bias) @ self.target_rot
        self.q_goal = self.model.ik_pose(self.target + self.bias, goal_rot, self.q_goal, use_seeds=False,
                                         symmetric=False, iterations=3)[0]
        return self.q_goal.float()


class Tracker:
    """Camera tracking along the whole path. The camera's gripper pose minus the pose the motor encoders imply is
    the deflection no encoder sees (play and bracket bending); motion lag shows up in the encoders too, so it is
    not in this difference. Low-pass filter it and add it to the planned joints through the Jacobian.
    (Integrating plan-minus-camera instead wound up on the lag of every move and overshot.)"""

    def __init__(self, model, num_envs, device, rotation_scale=0.1, damping=0.02):
        self.model, self.rotation_scale, self.damping = model, rotation_scale, damping
        self.deflection = torch.zeros(num_envs, 6, dtype=torch.float64, device=device)  # Position (m), rotation.

    def step(self, q_plan, q_encoder, true_tcp, true_rot, dt):
        believed_tcp, believed_rot = self.model.pose(q_encoder.double())
        measured = torch.cat([believed_tcp - true_tcp.double(),
                              rotation_log(believed_rot @ true_rot.double().transpose(1, 2))], -1)
        self.deflection += (measured - self.deflection) * min(1., dt / TRACK_FILTER_S)
        q = q_plan.double()
        scale = torch.tensor([1., 1., 1.] + [self.rotation_scale] * 3, dtype=torch.float64, device=q.device)
        jac = self.model.jacobian6(q) * scale[:, None]
        eye = torch.eye(6, dtype=torch.float64, device=q.device)
        delta = (jac.transpose(1, 2) @ torch.linalg.solve(jac @ jac.transpose(1, 2) + self.damping**2 * eye,
                                                         (self.deflection * scale)[..., None]))[..., 0]
        return (q + delta).float()


def main():
    plans = np.load(args.plans)
    specs = arm_specs()
    loose = any(model == 'loose' for _, model, _, _ in specs)
    levels = [LEVELS[name] for name in args.levels.split(',')] if loose else [None]
    control_hz = float(plans['control_hz'])
    decimation = round(args.physics_hz / control_hz)
    physics_dt, control_dt = 1 / args.physics_hz, 1 / control_hz
    n_sequences, n_targets = plans['curobo_steps'].shape
    n_sequences -= args.first_sequence
    if args.sequences:
        n_sequences = min(n_sequences, args.sequences)
    if args.targets:
        n_targets = min(n_targets, args.targets)
    num_envs = n_sequences * len(levels)
    env_level = torch.arange(num_envs) // n_sequences
    env_sequence = torch.arange(num_envs) % n_sequences + args.first_sequence
    device = args.device
    model = ArmModel(device=device)
    hold_steps = round(args.hold * control_hz)
    planners = sorted({planner for _, _, planner, _ in specs})
    segment_steps = int(max(plans[f'{p}_steps'][:, :n_targets].max() for p in planners)) + hold_steps
    sequence = env_sequence.to(device)
    trajectories = {p: torch.tensor(plans[p], device=device)[sequence, :n_targets] for p in planners}  # [E, K, T, 6]
    steps = {p: torch.tensor(plans[f'{p}_steps'], device=device)[sequence, :n_targets] for p in planners}
    targets = torch.tensor(plans['positions'], device=device, dtype=torch.float32)[sequence, :n_targets]  # [E, K, 3]
    target_rots = torch.tensor(plans['rotations'], device=device)[sequence, :n_targets]  # [E, K, 3, 3]
    feasible = torch.tensor(plans['curobo_success'], device=device)[sequence, :n_targets]
    # Correct only after plans that reached their target: a failed cuRobo plan stays put, and pulling the arm
    # toward its (unreachable, usually walled-in) target drove it into the wall.
    planned = {'curobo': feasible, 'ik_joint': torch.ones_like(feasible)}

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=physics_dt, device=device, render_interval=decimation))
    scene = InteractiveScene(build_scene_cfg(num_envs, specs, args.video))
    sim.reset()
    robots = {name: scene[f'robot_{name}'] for name, _, _, _ in specs}
    for name, model_kind, _, _ in specs:
        if model_kind == 'loose':
            apply_looseness(robots[name], [levels[i] for i in env_level.tolist()])
    frames = {name: MountFrame(robot) for name, robot in robots.items()}
    home = torch.tensor(plans['home'], device=device, dtype=torch.float32).expand(num_envs, 6)
    correctors = {name: HoldCorrector(model, num_envs, device) for name, _, _, _ in specs}
    trackers = {name: Tracker(model, num_envs, device) for name, _, _, _ in specs}
    loose_params = None
    if loose:
        level_of_env = [levels[i] for i in env_level.tolist()]
        half_gap = torch.tensor([math.radians(l.gear_deg + l.play_deg) / 2 for l in level_of_env], device=device)
        stiffness = torch.tensor([l.flex_nm_per_rad for l in level_of_env], device=device)
        loose_params = (half_gap[:, None], stiffness[:, None])

    def sag_compensation(q, params):
        """Joint offsets that cancel the gravity sag of play and bracket bending on loaded joints."""
        half_gap, stiffness = params
        torque = model.gravity_torque(q).float()
        sag = torque / stiffness + torch.sign(torque) * half_gap
        return torch.where(torque.abs() > LOADED_TORQUE, -sag, torch.zeros_like(sag))

    def command(q):
        return q - (model.gravity_torque(q) / SERVO_STIFFNESS).float()

    def gripper_rotation(name):
        return frames[name].tool_rotation().double() @ model.gripper_in_tool

    for name, robot in robots.items():
        joint_pos = robot.data.default_joint_pos.clone()
        joint_pos[:, frames[name].servo_ids] = home
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        robot.set_joint_position_target(command(home), joint_ids=frames[name].servo_ids)
    for _ in range(round(1.0 * args.physics_hz)):  # Settle at home.
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(physics_dt)

    shape = (num_envs, n_targets)
    touched_move = {name: torch.zeros(shape, dtype=torch.bool, device=device) for name in robots}
    touched_hold = {name: torch.zeros(shape, dtype=torch.bool, device=device) for name in robots}
    bodies_touched = {name: {} for name in robots}
    final_error = {name: torch.zeros(shape, device=device) for name in robots}
    final_angle = {name: torch.zeros(shape, device=device) for name in robots}
    video = VideoWriter(args.out / f'{args.name}.mp4', control_hz, specs, levels, env_level) if args.video else None
    rows = torch.arange(num_envs, device=device)
    for k in range(n_targets):
        for t in range(segment_steps):
            holding = {}
            for name, model_kind, planner, correction in specs:
                last = steps[planner][:, k] - 1
                q = trajectories[planner][rows, k, torch.clamp(torch.full_like(last, t), max=last)]
                holding[name] = t > last
                if correction == 'track':
                    q = trackers[name].step(q, frames[name].encoders()[0], frames[name].tcp(), gripper_rotation(name),
                                            control_dt)
                elif correction == 'model' and model_kind == 'loose':
                    q = q + sag_compensation(q, loose_params)
                if correction == 'camera':
                    correctable = planned[planner][:, k]
                    started = (t == last + 1) & correctable
                    if bool(started.any()):
                        # Start correcting from the plan's end, toward the grasp variant it reached.
                        _, reached = model.pose(q.double())
                        flipped = target_rots[:, k] @ FLIP.to(device)
                        use_flip = ((reached - flipped).abs().sum((1, 2)) < (reached - target_rots[:, k]).abs().sum((1, 2)))
                        wanted = torch.where(use_flip[:, None, None], flipped, target_rots[:, k])
                        corrector = correctors[name]
                        fresh = HoldCorrector(model, num_envs, device)
                        fresh.start(q, targets[:, k], wanted)
                        for attr in ('q_goal', 'target', 'target_rot', 'bias', 'rotation_bias'):
                            getattr(corrector, attr)[started] = getattr(fresh, attr)[started]
                    corrected = correctors[name].step(frames[name].tcp(), gripper_rotation(name), control_dt)
                    q = torch.where((holding[name] & ~started & correctable)[:, None], corrected, q)
                robots[name].set_joint_position_target(command(q), joint_ids=frames[name].servo_ids)
            contact_now = {name: torch.zeros(num_envs, dtype=torch.bool, device=device) for name in robots}
            for _ in range(decimation):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(physics_dt)
                for name in robots:
                    sensor = scene[f'contacts_{name}']
                    force = sensor.data.net_forces_w.norm(dim=-1)
                    contact_now[name] |= force.amax(-1) > CONTACT_N
                    for body in torch.nonzero((force > CONTACT_N).any(0))[:, 0].tolist():
                        body_name = sensor.body_names[body]  # The sensor's own body order.
                        bodies_touched[name][body_name] = bodies_touched[name].get(body_name, 0) + 1
            for name in robots:
                touched_move[name][:, k] |= contact_now[name] & ~holding[name]
                touched_hold[name][:, k] |= contact_now[name] & holding[name]
            if video is not None:
                video.add(scene, frames, targets[:, k], contact_now, k)
        for name in robots:
            final_error[name][:, k] = (frames[name].tcp() - targets[:, k]).norm(dim=-1)
            rot = gripper_rotation(name)
            angle = lambda r: torch.acos((((rot * r).sum((1, 2)) - 1) / 2).clamp(-1., 1.))
            final_angle[name][:, k] = torch.minimum(angle(target_rots[:, k]),
                                                    angle(target_rots[:, k] @ FLIP.to(device))).float()
        print(f'RUN_PLANS: target {k + 1}/{n_targets} done', flush=True)

    report = {'plans': str(args.plans), 'arms': args.arms, 'correct': args.correct, 'hold_s': args.hold,
              'sequences': n_sequences, 'targets_per_sequence': n_targets, 'results': []}
    for name, model_kind, planner, correction in specs:
        for v, level in enumerate(levels):
            envs = env_level.to(device) == v
            mask = feasible & envs[:, None]
            pick = lambda x: x[mask]
            report['results'].append({
                'arm': name, 'model': model_kind, 'planner': planner, 'correction': correction,
                'level': level.name if (level is not None and model_kind == 'loose') else 'rigid',
                'feasible_targets': int(mask.sum()),
                'contact_during_move': round(float(pick(touched_move[name]).float().mean()), 3),
                'contact_during_hold': round(float(pick(touched_hold[name]).float().mean()), 3),
                'final_error_mm_median': round(float(pick(final_error[name]).median() * 1e3), 2),
                'final_error_mm_p90': round(float(pick(final_error[name]).quantile(0.9) * 1e3), 2),
                'final_rotation_deg_median': round(float(pick(final_angle[name]).median() * 180 / math.pi), 3),
            })
    report['contact_physics_steps_per_body'] = bodies_touched
    np.savez_compressed(args.out / f'{args.name}.npz', env_level=env_level.numpy(), env_sequence=env_sequence.numpy(),
                        **{f'{name}_contact_move': touched_move[name].cpu().numpy() for name in robots},
                        **{f'{name}_contact_hold': touched_hold[name].cpu().numpy() for name in robots},
                        **{f'{name}_final_error': final_error[name].cpu().numpy() for name in robots},
                        targets=targets.cpu().numpy(), feasible=feasible.cpu().numpy())
    print('RUN_PLANS_RESULTS: ' + json.dumps(report['results']), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'{args.name}.json').write_text(json.dumps(report, indent=2) + '\n')
    if video is not None:
        video.close()


class VideoWriter:
    """One tile per arm and environment, target ring, and a red frame while the arm touches something."""
    VIEWS = {'wide': ((0.85, -0.75, 0.55), (0.22, 0.0, 0.06)),  # Eye and look-at, mount frame of each arm, m.
             'close': ((0.62, -0.50, 0.36), (0.20, 0.0, 0.07)),
             'high': ((0.74, -0.40, 0.68), (0.19, 0.0, 0.07))}

    def __init__(self, path, fps, specs, levels, env_level):
        import imageio.v2 as imageio
        from PIL import ImageFont
        font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        self.font = ImageFont.truetype(font_path, 18) if os.path.exists(font_path) else ImageFont.load_default()
        self.path, self.specs, self.levels, self.env_level = path, specs, levels, env_level
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(path, fps=round(fps), codec='libx264', quality=7, macro_block_size=8)
        self.posed = False

    def label(self, e, model, planner, correction):
        arm = 'rigid arm' if model == 'rigid' else self.levels[int(self.env_level[e])].label.split(',')[0]
        return f'{PLANNER_LABELS[planner]}{CORRECTION_LABELS[correction]} | {arm}'

    def add(self, scene, frames, targets, contact_now, k):
        from PIL import Image, ImageDraw
        num_envs = scene.num_envs
        if not self.posed:
            for name, _, _, _ in self.specs:
                eye, look = (torch.tensor(p, device=targets.device).expand(num_envs, 3) for p in VideoWriter.VIEWS[args.camera])
                scene[f'camera_{name}'].set_world_poses_from_view(frames[name].to_world(eye), frames[name].to_world(look))
            self.posed = True
        sim_utils.SimulationContext.instance().render()
        tiles = []
        for name, model, planner, correction in self.specs:
            camera = scene[f'camera_{name}']
            camera.update(0., force_recompute=True)
            rgb = camera.data.output['rgb'][..., :3].cpu().numpy().astype(np.uint8)
            data = camera.data
            p = quat_apply_inverse(data.quat_w_ros, frames[name].to_world(targets) - data.pos_w)
            uv = (data.intrinsic_matrices @ p[..., None])[..., 0].cpu().numpy()
            error_mm = ((frames[name].tcp() - targets).norm(dim=-1) * 1e3).cpu().numpy()
            column = []
            for e in range(num_envs):
                image = Image.fromarray(rgb[e])
                draw = ImageDraw.Draw(image)
                x, y = uv[e, 0] / uv[e, 2], uv[e, 1] / uv[e, 2]
                draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline=(40, 200, 60), width=4)
                if args.inset:  # 2.5x zoom around the target.
                    half = 60
                    zoom = image.crop((int(x) - half, int(y) - half, int(x) + half, int(y) + half)).resize((300, 300))
                    image.paste(zoom, (8, 38))
                    draw.rectangle((7, 37, 309, 339), outline='white', width=2)
                draw.rectangle((0, image.height - 34, image.width, image.height), fill=(0, 0, 0))
                draw.text((10, image.height - 29), f'gripper {error_mm[e]:6.1f} mm from target', fill='white',
                          font=self.font)
                if bool(contact_now[name][e]):
                    draw.rectangle((0, 0, image.width - 1, image.height - 1), outline=(230, 30, 30), width=10)
                    draw.text((image.width - 140, 40), 'CONTACT', fill=(230, 30, 30), font=self.font)
                draw.rectangle((0, 0, image.width, 30), fill=(0, 0, 0))
                draw.text((10, 5), f'{self.label(e, model, planner, correction)} | target {k + 1}', fill='white',
                          font=self.font)
                column.append(image)
            tiles.append(column)
        width, height = tiles[0][0].size
        canvas = Image.new('RGB', (width * len(tiles), height * num_envs))
        for a, column in enumerate(tiles):
            for e, image in enumerate(column):
                canvas.paste(image, (a * width, e * height))
        self.writer.append_data(np.asarray(canvas))

    def close(self):
        self.writer.close()
        print(f'RUN_PLANS_VIDEO: {self.path}', flush=True)


exit_code = 1
try:
    main()
    exit_code = 0
except BaseException:
    import traceback
    traceback.print_exc()
finally:
    sys.stdout.flush()
    os._exit(exit_code)
