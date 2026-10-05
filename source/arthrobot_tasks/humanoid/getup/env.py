"""Get-up environment for the wheel-legged humanoid (Isaac Lab direct env).

- Episodes start from a floor-placed pose from a pose bank. The first 0.6 s are
  unactuated so the robot settles before the policy acts (upright "standing"
  starts are actuated immediately).
- 18 limb actions set PD targets relative to the current joint angle:
  target = q + action_bound x action, clipped to the self-collision limits; the
  runner shrinks ``action_bound`` over training. 2 wheel actions are torques.
  Every motor is capped at 13 N m and 74 rpm.
- Rewards come in five groups (task, regularization, style, target, safety), one
  critic each; see :mod:`.ppo`. Near standing, the target group can reward the
  ending pose as HoST does (``arm_pose_width``, ``leg_pose_width``): a wide Gaussian
  of the squared joint errors from the standing policy's pose, so it still pulls
  when the arms are far away.
- An optional upward pull on the torso (``pull_force_n``, set by the runner) helps
  early learning and is removed during training.
- Domain randomization: friction, restitution, link masses, torso payload and COM
  (at startup); PD gains, motor strength, joint offsets and action delay (per
  reset); observation noise.
No scripted motion, reference trajectory or balance controller is used.
Physics: 240 Hz, TGS 8/1 on the light get-up asset, 60 Hz control.
"""
import json
import math
from pathlib import Path

import torch
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from arthrobot.motors import load_motor
from arthrobot_assets.humanoid import LIMB_JOINT_COUNT as LIMBS, MOTOR_JOINTS, WHEEL_BODIES
from arthrobot_tasks.humanoid.standing.policy import CHECKPOINT as STANDING_CHECKPOINT

MOTOR = load_motor('mg5010')
MAX_SPEED = MOTOR.max_speed_rad_s
MOTORS = len(MOTOR_JOINTS)
REWARD_GROUPS = ('task', 'regularization', 'style', 'target', 'safety')
HISTORY = 6                                   # observation frames stacked for the actor
ONE_STEP_OBS = 3 + 3 + LIMBS + MOTORS + MOTORS  # angular velocity, gravity, limb angles, speeds, actions
PRIVILEGED_OBS = 1 + 3 + 21 + 4               # height, linear velocity, body contacts, curriculum/time state
NOMINAL_WHEEL_TRACK_M = .419                  # wheel-center distance in the standing pose (URDF FK)
WHEEL_CONTACT_FORCE_N = 3.


@configclass
class GetupEvents:
    material = EventTermCfg(func=mdp.randomize_rigid_body_material, mode='startup', params=dict(
        asset_cfg=SceneEntityCfg('robot', body_names='.*'), static_friction_range=(.4, 1.1),
        dynamic_friction_range=(.3, .9), restitution_range=(0., .3), num_buckets=64, make_consistent=True))
    link_mass = EventTermCfg(func=mdp.randomize_rigid_body_mass, mode='startup', params=dict(
        asset_cfg=SceneEntityCfg('robot', body_names='.*'), mass_distribution_params=(.9, 1.1),
        operation='scale', recompute_inertia=True))
    payload = EventTermCfg(func=mdp.randomize_rigid_body_mass, mode='startup', params=dict(
        asset_cfg=SceneEntityCfg('robot', body_names='torso'), mass_distribution_params=(-.3, .8),
        operation='add', recompute_inertia=True))
    com = EventTermCfg(func=mdp.randomize_rigid_body_com, mode='startup', params=dict(
        asset_cfg=SceneEntityCfg('robot', body_names='torso'),
        com_range={'x': (-.02, .02), 'y': (-.02, .02), 'z': (-.02, .02)}))


def floor_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(static_friction=.9, dynamic_friction=.8, restitution=0.,
                                          friction_combine_mode='average')


@configclass
class GetupEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 10.
    unactuated_s = .6
    action_space = MOTORS
    observation_space = HISTORY * ONE_STEP_OBS
    state_space = HISTORY * ONE_STEP_OBS + PRIVILEGED_OBS
    asset_path = ''                    # set from ensure_getup_usd()
    pose_bank = ''                     # JSON pose bank of start poses
    standing_settings = str(STANDING_CHECKPOINT.parent / 'settings.json')   # nominal standing pose
    kp, kd, torque_limit = 80., 4., 13.
    soft_limit_fraction = .9
    randomize = True
    kp_range, kd_range, strength_range = (.85, 1.15), (.85, 1.15), (.9, 1.1)
    offset_range = .03
    max_delay_substeps = 6
    obs_noise = True
    # Safety reward group: self-contact, joint speed, torque, torso rotation rate.
    self_contact_n = 1.                # non-floor contact force that counts as self-contact
    self_contact_weight = 0.           # per body in contact the policy created
    inherited_contact_weight = 0.      # per body still touching since the passive settle
    joint_speed_soft = .5              # fraction of 74 rpm; limbs only (wheels need speed to balance)
    joint_speed_weight = 0.            # per rad/s above the soft limit, summed over limbs
    torque_soft_nm = 9.                # ~70% of the 13 N m continuous rating
    torque_weight = 0.                 # per N m above torque_soft_nm, summed over limbs
    torso_rate_soft = 2.               # rad/s
    torso_rate_weight = 0.             # per rad/s of torso angular speed above torso_rate_soft
    posture_l1_weight = 0.             # near standing: per rad of |q - nominal|, legs
    arm_posture_l1_weight = 0.         # near standing: per rad of |q - nominal|, arms
    posture_progress_weight = 0.       # near standing: per rad of arm error removed
    posture_weight = 1.                # near standing: exp(-2 x squared error of all 18 limbs); too narrow to pull far
    # Ending pose as in HoST's target_upper_dof_pos (0 = off): near standing, reward
    # exp(-width x sum of squared joint errors from the standing pose). HoST uses 0.1 for the upper body.
    arm_pose_width = 0.                # 12 arm joints
    leg_pose_width = 0.                # 6 leg joints (they stay closer to the pose, so a narrower Gaussian)
    # Near standing: per arm joint, reward rising linearly from 0 (pi rad off) to 1 (at the pose), averaged.
    # Unlike the Gaussians it still pulls when every arm joint rests on a joint stop.
    arm_pose_linear_weight = 0.
    # Safety group, near standing: per rad beyond 90% of a limb joint's range, so the arms cannot rest on the stops.
    standing_joint_limit_weight = 0.
    ready_tolerance = .25              # rad; the standing policy's command range around nominal
    standing_fraction = 0.             # share of training resets from the 'standing' bank family
    # Stand-up schedule (0 = off): height and uprightness are rewarded for following a smooth
    # ramp from their values at motor-on to standing over this many seconds; the standing
    # bonus only counts once the ramp has finished.
    height_schedule_s = 0.
    height_track_sigma = .05           # m
    upright_track_sigma = .25          # in -gravity_z units (1 = upright)
    standing_height_m = .46
    standing_tilt_deg = 15.
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 240, render_interval=decimation,
        physx=sim_utils.PhysxCfg(solver_type=1, gpu_max_rigid_contact_count=2**22, gpu_max_rigid_patch_count=2**19,
                                gpu_found_lost_pairs_capacity=2**22),
        physics_material=floor_material())
    terrain = TerrainImporterCfg(prim_path='/World/ground', terrain_type='plane', collision_group=-1,
                                 physics_material=floor_material())
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=3., replicate_physics=True)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path='/World/envs/env_.*/Robot',
        spawn=sim_utils.UsdFileCfg(
            usd_path='', activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=1)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., .5), joint_pos={'.*': 0.}),
        actuators={'motors': ImplicitActuatorCfg(joint_names_expr=list(MOTOR_JOINTS), stiffness=0., damping=0.,
                                                 effort_limit_sim=13., velocity_limit_sim=MAX_SPEED,
                                                 armature=MOTOR.reflected_inertia_kg_m2)},
        soft_joint_pos_limit_factor=1.)
    contacts: ContactSensorCfg = ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/.*', history_length=1,
                                                  update_period=1 / 240)
    events: GetupEvents = GetupEvents()


class GetupEnv(DirectRLEnv):
    cfg: GetupEnvCfg

    def __init__(self, cfg: GetupEnvCfg, render_mode=None, **kwargs):
        if not cfg.randomize:
            cfg.events = None
        super().__init__(cfg, render_mode, **kwargs)
        n, device = self.num_envs, self.device
        self.motor_ids, names = self.robot.find_joints(list(MOTOR_JOINTS), preserve_order=True)
        assert names == list(MOTOR_JOINTS)
        self.torso_id = self.robot.find_bodies('torso')[0][0]
        self.wheel_body_ids = self.robot.find_bodies(list(WHEEL_BODIES), preserve_order=True)[0]
        self.knee_body_ids = self.robot.find_bodies(['left_knee_body', 'right_knee_body'], preserve_order=True)[0]
        self.wheel_sensor_ids = self.contacts.find_bodies(list(WHEEL_BODIES), preserve_order=True)[0]
        self.body_sensor_ids = [i for i, name in enumerate(self.contacts.body_names) if not name.endswith('wheel_body')]
        assert len(self.contacts.body_names) == 21
        assert self.floor_contacts.body_names == self.contacts.body_names
        pose = json.loads(Path(cfg.standing_settings).read_text())['nominal_pose']
        self.nominal = torch.tensor([pose['joint_positions'].get(name, 0.) for name in MOTOR_JOINTS], device=device)
        self.standing_height = float(pose['initial_height_m'])
        limits = self.robot.data.joint_pos_limits[0, self.motor_ids[:LIMBS]]
        self.low, self.high = limits[:, 0].clone(), limits[:, 1].clone()
        middle, half_range = (self.low + self.high) / 2, (self.high - self.low) / 2
        self.soft_low = middle - cfg.soft_limit_fraction * half_range
        self.soft_high = middle + cfg.soft_limit_fraction * half_range
        bank = json.loads(Path(cfg.pose_bank).read_text())
        assert bank['joint_order'] == list(MOTOR_JOINTS)
        self.bank_root = torch.tensor([pose['root_pose'] for pose in bank['poses']], device=device)
        self.bank_q = torch.tensor([pose['joints'] for pose in bank['poses']], device=device)
        self.bank_family = torch.tensor([bank['families'].index(pose['family']) for pose in bank['poses']], device=device)
        self.families = tuple(bank['families'])
        self.training_rows = len(self.bank_root)
        self.forced_pose = None            # evaluation: a bank index per environment
        self.arm_ready_time = torch.zeros(n, device=device)
        self.best_arm_ready = torch.zeros(n, device=device)
        # Curriculum values, set by the runner.
        self.action_bound = 1.
        self.pull_force_n = 0.
        self.unactuated_steps = int(round(cfg.unactuated_s / self.step_dt))
        # Lying starts settle passively first; standing starts are actuated from the first step.
        self.unactuated_env = torch.full((n,), self.unactuated_steps, dtype=torch.long, device=device)
        self.standing_family = self.families.index('standing') if 'standing' in self.families else None
        self.start_height = torch.zeros(n, device=device)
        self.start_upright = torch.zeros(n, device=device)
        self.noise_scale = torch.cat((torch.full((3,), .05), torch.full((3,), .03), torch.full((LIMBS,), .01),
                                      torch.full((MOTORS,), .03), torch.zeros(MOTORS))).to(device)
        # Control buffers.
        self.actions = torch.zeros(n, MOTORS, device=device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.previous_actions2 = torch.zeros_like(self.actions)
        self.targets = torch.zeros(n, LIMBS, device=device)
        self.previous_targets = torch.zeros_like(self.targets)
        self.torque = torch.zeros(n, MOTORS, device=device)
        self.previous_joint_vel = torch.zeros(n, MOTORS, device=device)
        self.substep = 0
        # Per-reset randomization.
        self.kp = torch.full((n, LIMBS), cfg.kp, device=device)
        self.kd = torch.full((n, LIMBS), cfg.kd, device=device)
        self.strength = torch.ones(n, MOTORS, device=device)
        self.offset = torch.zeros(n, LIMBS, device=device)
        self.delay = torch.zeros(n, 1, dtype=torch.long, device=device)
        # Observation history and episode statistics.
        self.history = torch.zeros(n, HISTORY, ONE_STEP_OBS, device=device)
        self.family = torch.zeros(n, dtype=torch.long, device=device)
        self.pose_index = torch.zeros(n, dtype=torch.long, device=device)
        self.start_xy = torch.zeros(n, 2, device=device)
        self.standing_time = torch.zeros(n, device=device)
        self.best_standing = torch.zeros(n, device=device)
        self.max_height = torch.zeros(n, device=device)
        self.first_standing_s = torch.full((n,), float('nan'), device=device)
        self.self_contact_time = torch.zeros(n, device=device)
        self.self_contact_standing_time = torch.zeros(n, device=device)
        self.body_self_contact_time = torch.zeros(n, 21, device=device)
        self.self_contact_steps = torch.zeros((), device=device)
        self.inherited = torch.zeros(n, 21, dtype=torch.bool, device=device)
        self.new_touch = torch.zeros(n, 21, dtype=torch.bool, device=device)
        self.inherited_touch = torch.zeros(n, 21, dtype=torch.bool, device=device)
        self.new_contact_time = torch.zeros(n, device=device)
        self.inherited_time = torch.zeros(n, device=device)
        self.ready_time = torch.zeros(n, device=device)
        self.best_ready = torch.zeros(n, device=device)
        self.peak_limb_speed = torch.zeros(n, device=device)
        self.over_speed_time = torch.zeros(n, device=device)
        self.saturated_time = torch.zeros(n, device=device)
        self.peak_torso_rate = torch.zeros(n, device=device)
        self.previous_arm_error = torch.zeros(n, device=device)
        self.previous_near = torch.zeros(n, dtype=torch.bool, device=device)
        self.group_sums = torch.zeros(n, len(REWARD_GROUPS), device=device)
        self.term_names = None             # reward-term names, in the column order of term_sums
        self.term_sums = None              # [N, terms] per-episode sums of every reward term (for logging)
        self.terminal_critic_obs = torch.zeros(n, cfg.state_space, device=device)
        self.completed = []
        self.control_stats = torch.zeros(3, device=device)   # torque saturation, torque RMS, action change
        self.control_stat_steps = 0

    def append_poses(self, bank: dict) -> int:
        """Add a bank's poses after the training rows, where training never samples them; return the first new row.

        Used for held-out evaluation poses (selected through ``forced_pose``).
        """
        assert bank['joint_order'] == list(MOTOR_JOINTS)
        first_row = len(self.bank_root)
        poses = bank['poses']
        self.bank_root = torch.cat((self.bank_root,
                                    torch.tensor([pose['root_pose'] for pose in poses], device=self.device)))
        self.bank_q = torch.cat((self.bank_q, torch.tensor([pose['joints'] for pose in poses], device=self.device)))
        self.bank_family = torch.cat((self.bank_family, torch.tensor(
            [self.families.index(pose['family']) for pose in poses], device=self.device)))
        return first_row

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.cfg.robot.spawn.usd_path = self.cfg.asset_path
        self.robot = Articulation(self.cfg.robot)
        self.contacts = ContactSensor(self.cfg.contacts)
        self._add_sensors()
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations['robot'] = self.robot
        self.scene.sensors['contacts'] = self.contacts
        # Floor-only contact per body. The ground is one shared plane, so a single sensor over
        # all bodies can filter against it; self-contact = total contact - floor contact.
        floor = next(str(prim.GetPath()) for prim in self.scene.stage.Traverse()
                     if str(prim.GetPath()).startswith('/World/ground') and prim.GetTypeName() == 'Plane')
        self.floor_contacts = ContactSensor(ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/.*',
                                                             filter_prim_paths_expr=[floor], history_length=1,
                                                             update_period=1 / 240))
        self.scene.sensors['floor_contacts'] = self.floor_contacts
        light = sim_utils.DomeLightCfg(intensity=2000., color=(.75, .75, .75))
        light.func('/World/Light', light)

    def _add_sensors(self):
        """Hook for extra sensors that must exist before cloning (e.g. cameras)."""

    # ------------------------------------------------------------------ action
    def actuated(self) -> torch.Tensor:
        return self.episode_length_buf >= self.unactuated_env

    def _pre_physics_step(self, actions):
        self.previous_actions2[:] = self.previous_actions
        self.previous_actions[:] = self.actions
        self.actions[:] = torch.nan_to_num(actions).clamp(-1., 1.)
        limb_angles = self.robot.data.joint_pos[:, self.motor_ids[:LIMBS]]
        self.previous_targets[:] = self.targets
        self.targets[:] = (limb_angles + self.action_bound * self.actions[:, :LIMBS]).clamp(self.low, self.high)
        self.substep = 0
        # The external wrench persists across physics steps, so it is set once per control step.
        force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        force[:, 0, 2] = self.pull_force_n * self.actuated()
        self.robot.set_external_force_and_torque(force, torch.zeros_like(force), body_ids=[self.torso_id],
                                                 is_global=True)

    def _apply_action(self):
        data = self.robot.data
        limb_angles = data.joint_pos[:, self.motor_ids[:LIMBS]] + self.offset
        velocities = data.joint_vel[:, self.motor_ids]
        # Action delay: the previous command stays active for the first `delay` physics steps.
        delayed = self.substep < self.delay
        targets = torch.where(delayed, self.previous_targets, self.targets)
        limb_torques = self.kp * (targets - limb_angles) - self.kd * velocities[:, :LIMBS]
        wheel_torques = self.cfg.torque_limit * torch.where(delayed, self.previous_actions[:, LIMBS:],
                                                            self.actions[:, LIMBS:])
        limit = self.cfg.torque_limit * self.strength
        torque = torch.cat((limb_torques, wheel_torques), -1)
        torque = torch.maximum(torch.minimum(torque, limit), -limit)
        self.torque[:] = torque * self.actuated()[:, None]
        self.robot.set_joint_effort_target(self.torque, joint_ids=self.motor_ids)
        self.substep += 1

    # ------------------------------------------------------------------ observation
    def torso_height(self) -> torch.Tensor:
        return self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]

    def one_step_obs(self) -> torch.Tensor:
        data = self.robot.data
        limb_angles = data.joint_pos[:, self.motor_ids[:LIMBS]] - self.nominal[:LIMBS]
        observation = torch.cat((data.root_ang_vel_b * .25, data.projected_gravity_b, limb_angles,
                                 data.joint_vel[:, self.motor_ids] / MAX_SPEED, self.actions), -1)
        if self.cfg.obs_noise:
            observation = observation + (torch.rand_like(observation) * 2 - 1) * self.noise_scale
        return observation

    def privileged_obs(self) -> torch.Tensor:
        data = self.robot.data
        contact = (self.contacts.data.net_forces_w.norm(dim=-1) > 1.).float()
        return torch.cat((self.torso_height()[:, None], data.root_lin_vel_b, contact,
                          torch.full((self.num_envs, 1), self.pull_force_n / 100., device=self.device),
                          (~self.actuated()).float()[:, None],
                          (1. - self.episode_length_buf / self.max_episode_length)[:, None],
                          (self.standing_time / 2.).clamp(max=1.)[:, None]), -1)

    def critic_obs(self) -> torch.Tensor:
        return torch.cat((self.history.flatten(1), self.privileged_obs()), -1)

    def _get_observations(self):
        return {'policy': self.history.flatten(1).clamp(-10., 10.), 'critic': self.critic_obs().clamp(-10., 10.)}

    # ------------------------------------------------------------------ reward
    def floor_forces(self) -> torch.Tensor:
        return self.floor_contacts.data.force_matrix_w[:, :, 0]

    def self_contact_forces(self) -> torch.Tensor:
        """Per-body contact force not explained by the floor (jointed neighbors never collide)."""
        return (self.contacts.data.net_forces_w - self.floor_forces()).norm(dim=-1)

    def standing(self) -> torch.Tensor:
        """High, upright and on both wheels."""
        wheels_down = self.floor_forces()[:, self.wheel_sensor_ids, 2] > WHEEL_CONTACT_FORCE_N
        upright = -self.robot.data.projected_gravity_b[:, 2] > math.cos(math.radians(self.cfg.standing_tilt_deg))
        return (self.torso_height() > self.cfg.standing_height_m) & upright & wheels_down.all(-1)

    def ending_pose_terms(self, deviation: torch.Tensor, near: torch.Tensor) -> dict[str, torch.Tensor]:
        """HoST-style ending-pose rewards near standing; only the enabled ones are returned."""
        terms = {}
        if self.cfg.arm_pose_width > 0:
            terms['arm_pose'] = near * torch.exp(-self.cfg.arm_pose_width * deviation[:, :12].square().sum(-1))
        if self.cfg.leg_pose_width > 0:
            terms['leg_pose'] = near * torch.exp(-self.cfg.leg_pose_width * deviation[:, 12:].square().sum(-1))
        if self.cfg.arm_pose_linear_weight > 0:
            closeness = (1. - deviation[:, :12] / math.pi).clamp(min=0.)
            terms['arm_pose_linear'] = self.cfg.arm_pose_linear_weight * near * closeness.mean(-1)
        return terms

    def arm_error_max(self) -> torch.Tensor:
        """[N] largest distance of an arm joint from the standing pose, rad."""
        return (self.robot.data.joint_pos[:, self.motor_ids[:12]] - self.nominal[:12]).abs().amax(-1)

    def ready_pose(self) -> torch.Tensor:
        """All 18 limb joints within the standing policy's command range of the nominal pose."""
        limb_angles = self.robot.data.joint_pos[:, self.motor_ids[:LIMBS]]
        return (limb_angles - self.nominal[:LIMBS]).abs().amax(-1) < self.cfg.ready_tolerance

    def reward_terms(self) -> dict[str, dict[str, torch.Tensor]]:
        cfg, data = self.cfg, self.robot.data
        height = self.torso_height()
        gravity = data.projected_gravity_b
        upright = -gravity[:, 2]
        angles, velocities = data.joint_pos[:, self.motor_ids], data.joint_vel[:, self.motor_ids]
        accelerations = (velocities - self.previous_joint_vel) / self.step_dt
        limb_angles = angles[:, :LIMBS]
        floor = self.floor_forces()
        wheels_down = floor[:, self.wheel_sensor_ids, 2] > WHEEL_CONTACT_FORCE_N
        body_floor_contacts = (floor[:, self.body_sensor_ids].norm(dim=-1) > 5.).float().sum(-1)
        near_mask = (height > .9 * self.standing_height) & (upright > math.cos(math.radians(30.)))
        near = near_mask.float()
        deviation = (limb_angles - self.nominal[:LIMBS]).abs()
        arm_error = deviation[:, :12].sum(-1)
        # Progress towards the nominal arm pose while near standing. It telescopes to the total
        # error removed, so a slow return earns as much as a fast one.
        progress = torch.where(near_mask & self.previous_near, (self.previous_arm_error - arm_error) / self.step_dt,
                               torch.zeros_like(arm_error)).clamp(-20., 20.)
        self._arm_error, self._near = arm_error, near_mask
        # Shank orientation: knee -> wheel points down when standing.
        shank = data.body_pos_w[:, self.wheel_body_ids] - data.body_pos_w[:, self.knee_body_ids]
        shank_down = (-shank[..., 2] / shank.norm(dim=-1).clamp_min(1e-6)).clamp(0., 1.).mean(-1)
        posture = (limb_angles - self.nominal[:LIMBS]).square().sum(-1)
        hip_abduction = (limb_angles[:, [13, 16]] - self.nominal[[13, 16]]).abs().sum(-1)
        wheel_spacing_xy = (data.body_pos_w[:, self.wheel_body_ids[0]] - data.body_pos_w[:, self.wheel_body_ids[1]])[:, :2]
        limit_violation = ((self.soft_low - limb_angles).clamp(min=0.) + (limb_angles - self.soft_high).clamp(min=0.)).sum(-1)
        speed_violation = (velocities.abs() - .9 * MAX_SPEED).clamp(min=0.).sum(-1)
        linear_speed = data.root_lin_vel_b[:, :2].square().sum(-1)
        angular_speed = data.root_ang_vel_b.square().sum(-1)
        if cfg.height_schedule_s > 0:
            since_motor_on = ((self.episode_length_buf - self.unactuated_env).float() * self.step_dt).clamp(min=0.)
            ramp = (since_motor_on / cfg.height_schedule_s).clamp(max=1.)
            ramp = ramp * ramp * (3. - 2. * ramp)   # smoothstep
            height_reference = self.start_height + (self.standing_height - self.start_height) * ramp
            upright_reference = self.start_upright + (1. - self.start_upright) * ramp
            task = dict(height=torch.exp(-((height - height_reference) / cfg.height_track_sigma).square()),
                        orientation=torch.exp(-((upright - upright_reference) / cfg.upright_track_sigma).square()),
                        standing=self.standing().float() * (since_motor_on >= cfg.height_schedule_s))
        else:
            task = dict(height=(height / self.standing_height).clamp(0., 1.), orientation=((upright + 1.) / 2.).square(),
                        standing=self.standing().float())
        return dict(
            task=task,
            regularization=dict(
                dof_acc=-2.5e-7 * accelerations[:, :LIMBS].square().sum(-1),
                action_rate=-.01 * (self.actions - self.previous_actions).square().sum(-1),
                smoothness=-.01 * (self.actions - 2 * self.previous_actions + self.previous_actions2).square().sum(-1),
                torques=-1e-4 * self.torque.square().sum(-1),
                joint_power=-2e-4 * (self.torque * velocities).abs().sum(-1),
                dof_vel=-1e-3 * velocities[:, :LIMBS].square().sum(-1),
                dof_pos_limits=-10. * limit_violation,
                dof_vel_limits=-1. * speed_violation),
            style=dict(
                hip_abduction=-1. * hip_abduction,
                shank_orientation=shank_down,
                wheel_track=-2. * (wheel_spacing_xy.norm(dim=-1) - NOMINAL_WHEEL_TRACK_M).abs() * near,
                ang_vel_xy=torch.exp(-2. * data.root_ang_vel_b[:, :2].square().sum(-1)),
                body_contact=-.5 * body_floor_contacts * near),
            target=dict(
                lin_vel=near * torch.exp(-linear_speed / .05),
                ang_vel=near * torch.exp(-angular_speed / .25),
                posture=cfg.posture_weight * near * torch.exp(-posture / .5),
                # The exponential is ~0 far from nominal; this keeps pulling back.
                posture_l1=-near * (cfg.posture_l1_weight * deviation[:, 12:].sum(-1) + cfg.arm_posture_l1_weight * arm_error),
                posture_progress=cfg.posture_progress_weight * progress,
                ready=near * self.ready_pose().float(),
                orientation=near * torch.exp(-gravity[:, :2].square().sum(-1) / .02),
                height=near * torch.exp(-(height - self.standing_height).square() / .002),
                wheel_contact=near * wheels_down.all(-1).float(),
                **self.ending_pose_terms(deviation, near)),
            safety=dict(
                self_contact_new=-cfg.self_contact_weight * self.new_touch.float().sum(-1),
                self_contact_inherited=-cfg.inherited_contact_weight * self.inherited_touch.float().sum(-1),
                joint_speed=-cfg.joint_speed_weight * (
                    velocities[:, :LIMBS].abs() - cfg.joint_speed_soft * MAX_SPEED).clamp(min=0.).sum(-1),
                torque=-cfg.torque_weight * (self.torque[:, :LIMBS].abs() - cfg.torque_soft_nm).clamp(min=0.).sum(-1),
                torso_rate=-cfg.torso_rate_weight * (data.root_ang_vel_b.norm(dim=-1) - cfg.torso_rate_soft).clamp(min=0.),
                **({'joint_limits': -cfg.standing_joint_limit_weight * near * limit_violation}
                   if cfg.standing_joint_limit_weight > 0 else {})))

    def update_contacts(self) -> torch.Tensor:
        """Split self-contact into contact left over from the passive settle and new contact."""
        touching = self.self_contact_forces() > self.cfg.self_contact_n
        motors_switch_on = (self.episode_length_buf == self.unactuated_env.clamp(min=1))[:, None]
        # Inherited = touching at motor-on and continuously since; released contact never becomes inherited again.
        self.inherited = torch.where(motors_switch_on, touching, self.inherited & touching)
        actuated = self.actuated()[:, None]
        self.new_touch = touching & ~self.inherited & actuated
        self.inherited_touch = self.inherited & actuated
        return touching & actuated

    def _get_rewards(self):
        touching = self.update_contacts()
        motors_switch_on = self.episode_length_buf == self.unactuated_env.clamp(min=1)
        self.start_height = torch.where(motors_switch_on, self.torso_height(), self.start_height)
        self.start_upright = torch.where(motors_switch_on, -self.robot.data.projected_gravity_b[:, 2], self.start_upright)
        terms = self.reward_terms()
        self.previous_arm_error[:], self.previous_near[:] = self._arm_error, self._near
        actuated = self.actuated().float()
        groups = torch.stack([sum(terms[group].values()) for group in REWARD_GROUPS], -1) * actuated[:, None] * self.step_dt
        self.group_sums += groups
        # All reward terms in one tensor: one GPU operation per step instead of one per term (logging only).
        if self.term_names is None:
            self.term_names = [f'{group}/{name}' for group, values in terms.items() for name in values]
            self.term_sums = torch.zeros(self.num_envs, len(self.term_names), device=self.device)
        values = torch.stack([value for group in terms.values() for value in group.values()], -1)
        self.term_sums += values * (actuated * self.step_dt)[:, None]
        self.extras['reward_groups'] = groups
        self.previous_joint_vel[:] = self.robot.data.joint_vel[:, self.motor_ids]
        self._update_statistics(touching)
        # The history advances here (after physics, before reset), so timeouts bootstrap from the true final state.
        self.history = torch.cat((self.history[:, 1:], self.one_step_obs()[:, None]), 1)
        self.terminal_critic_obs = self.critic_obs().clamp(-10., 10.)
        return groups.sum(-1)

    def _update_statistics(self, touching):
        dt, actuated = self.step_dt, self.actuated()
        standing = self.standing() & actuated
        self.standing_time = torch.where(standing, self.standing_time + dt, torch.zeros_like(self.standing_time))
        self.best_standing = torch.maximum(self.best_standing, self.standing_time)
        self.max_height = torch.maximum(self.max_height, self.torso_height())
        self.body_self_contact_time += touching.float() * dt
        self.self_contact_time += touching.any(-1).float() * dt
        self.self_contact_standing_time += (touching.any(-1) & standing).float() * dt
        self.self_contact_steps += touching.any(-1).float().mean()
        self.new_contact_time += self.new_touch.any(-1).float() * dt
        self.inherited_time += self.inherited_touch.any(-1).float() * dt
        ready = standing & self.ready_pose()
        self.ready_time = torch.where(ready, self.ready_time + dt, torch.zeros_like(self.ready_time))
        self.best_ready = torch.maximum(self.best_ready, self.ready_time)
        arms_ready = standing & (self.arm_error_max() < self.cfg.ready_tolerance)
        self.arm_ready_time = torch.where(arms_ready, self.arm_ready_time + dt, torch.zeros_like(self.arm_ready_time))
        self.best_arm_ready = torch.maximum(self.best_arm_ready, self.arm_ready_time)
        limb_speed = self.robot.data.joint_vel[:, self.motor_ids[:LIMBS]].abs()
        self.peak_limb_speed = torch.where(actuated, torch.maximum(self.peak_limb_speed, limb_speed.amax(-1)),
                                           self.peak_limb_speed)
        self.over_speed_time += ((limb_speed > self.cfg.joint_speed_soft * MAX_SPEED).any(-1) & actuated).float() * dt
        saturated = (self.torque[:, :LIMBS].abs() >= .99 * self.cfg.torque_limit * self.strength[:, :LIMBS]).any(-1)
        self.saturated_time += (saturated & actuated).float() * dt
        torso_rate = self.robot.data.root_ang_vel_b.norm(dim=-1)
        self.peak_torso_rate = torch.where(actuated, torch.maximum(self.peak_torso_rate, torso_rate), self.peak_torso_rate)
        first = standing & torch.isnan(self.first_standing_s)
        self.first_standing_s = torch.where(first, (self.episode_length_buf - self.unactuated_env).float() * dt,
                                            self.first_standing_s)
        self.control_stats += torch.stack(((self.torque.abs() >= .99 * self.cfg.torque_limit).float().mean(),
                                           self.torque.square().mean().sqrt(),
                                           (self.actions - self.previous_actions).square().mean()))
        self.control_stat_steps += 1

    # ------------------------------------------------------------------ termination and reset
    def _get_dones(self):
        data = self.robot.data
        invalid = (~torch.isfinite(data.root_state_w).all(-1) | ~torch.isfinite(data.joint_pos).all(-1)
                   | ~torch.isfinite(data.joint_vel).all(-1))
        escaped = (data.root_pos_w[:, :2] - self.start_xy).norm(dim=-1) > 3.
        below_floor = self.torso_height() < -.2
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        failed = invalid | escaped | below_floor
        return failed, timeout & ~failed

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if hasattr(self, 'history'):
            self.record_episodes(env_ids)
        super()._reset_idx(env_ids)
        if not hasattr(self, 'history'):   # still inside DirectRLEnv construction
            return
        n = len(env_ids)
        if self.forced_pose is not None:
            index = self.forced_pose[env_ids]
        elif self.standing_family is not None and self.cfg.standing_fraction > 0:
            standing_rows = torch.nonzero(self.bank_family == self.standing_family).flatten()
            lying_rows = torch.nonzero(self.bank_family[:self.training_rows] != self.standing_family).flatten()
            use_standing = torch.rand(n, device=self.device) < self.cfg.standing_fraction
            index = torch.where(use_standing, standing_rows[torch.randint(len(standing_rows), (n,), device=self.device)],
                                lying_rows[torch.randint(len(lying_rows), (n,), device=self.device)])
        else:
            index = torch.randint(self.training_rows, (n,), device=self.device)
        self.pose_index[env_ids] = index
        self.family[env_ids] = self.bank_family[index]
        root = torch.zeros(n, 13, device=self.device)
        root[:, :7] = self.bank_root[index]
        root[:, :3] += self.scene.env_origins[env_ids]
        joint_positions = self.robot.data.default_joint_pos[env_ids].clone()
        joint_positions[:, self.motor_ids] = self.bank_q[index]
        self.robot.write_root_state_to_sim(root, env_ids)
        self.robot.write_joint_state_to_sim(joint_positions, torch.zeros_like(joint_positions), env_ids=env_ids)
        self.robot.set_joint_effort_target(torch.zeros_like(joint_positions), env_ids=env_ids)
        self.start_xy[env_ids] = root[:, :2]
        if self.cfg.randomize:
            def uniform(bounds, shape):
                return torch.empty(shape, device=self.device).uniform_(*bounds)
            self.kp[env_ids] = self.cfg.kp * uniform(self.cfg.kp_range, (n, LIMBS))
            self.kd[env_ids] = self.cfg.kd * uniform(self.cfg.kd_range, (n, LIMBS))
            self.strength[env_ids] = uniform(self.cfg.strength_range, (n, MOTORS))
            self.offset[env_ids] = uniform((-self.cfg.offset_range, self.cfg.offset_range), (n, LIMBS))
            self.delay[env_ids] = torch.randint(0, self.cfg.max_delay_substeps + 1, (n, 1), device=self.device)
        start_limb_angles = self.bank_q[index, :LIMBS]
        for buffer in (self.actions, self.previous_actions, self.previous_actions2, self.torque, self.previous_joint_vel,
                       self.group_sums, self.standing_time, self.best_standing, self.max_height, self.self_contact_time,
                       self.self_contact_standing_time, self.body_self_contact_time, self.inherited, self.new_touch,
                       self.inherited_touch, self.new_contact_time, self.inherited_time, self.ready_time,
                       self.best_ready, self.peak_limb_speed, self.over_speed_time, self.saturated_time,
                       self.peak_torso_rate, self.previous_arm_error, self.previous_near, self.arm_ready_time,
                       self.best_arm_ready, *([self.term_sums] if self.term_sums is not None else [])):
            buffer[env_ids] = 0.
        self.targets[env_ids] = start_limb_angles
        self.previous_targets[env_ids] = start_limb_angles
        standing_start = (self.family[env_ids] == self.standing_family if self.standing_family is not None
                          else torch.zeros(n, dtype=torch.bool, device=self.device))
        self.unactuated_env[env_ids] = torch.where(standing_start, 0, self.unactuated_steps)
        self.first_standing_s[env_ids] = float('nan')
        self.history[env_ids] = 0.
        self.history[env_ids] = self.one_step_obs()[env_ids][:, None].expand(-1, HISTORY, -1)

    def record_episodes(self, env_ids):
        """Append a summary of every finished episode to ``self.completed``."""
        done = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(done) == 0:
            return
        rows = torch.stack((done.float(), self.family[done].float(), self.pose_index[done].float(),
                            self.episode_length_buf[done].float() * self.step_dt, self.best_standing[done],
                            self.standing_time[done], self.max_height[done], self.first_standing_s[done],
                            self.self_contact_time[done], self.self_contact_standing_time[done],
                            self.new_contact_time[done], self.inherited_time[done], self.best_ready[done],
                            self.peak_limb_speed[done], self.over_speed_time[done], self.saturated_time[done],
                            self.peak_torso_rate[done], self.best_arm_ready[done], self.arm_error_max()[done],
                            *self.group_sums[done].T), -1)
        # One transfer to the CPU for the episode rows and the per-body contact times.
        columns = rows.shape[1]
        packed = torch.cat((rows, self.body_self_contact_time[done]), -1).cpu().tolist()
        rows, body_times = [row[:columns] for row in packed], [row[columns:] for row in packed]
        names = self.contacts.body_names
        for row, per_body in zip(rows, body_times):
            (env, family, pose, seconds, best, final, height, first, touch, touch_standing, new_touch, inherited,
             ready, speed, over_speed, saturated, torso_rate, arm_ready, final_arm_error, *group_returns) = row
            self.completed.append(dict(
                env=int(env), family=self.families[int(family)], pose=int(pose), seconds=seconds,
                best_standing_s=best, final_standing_s=final, max_height_m=height,
                first_standing_s=None if math.isnan(first) else first, stood_2s=best >= 2., standing_at_end=final >= 1.,
                self_contact_s=touch, self_contact_standing_s=touch_standing, new_contact_s=new_touch,
                inherited_contact_s=inherited, best_ready_s=ready, peak_limb_speed=speed, over_speed_s=over_speed,
                saturated_s=saturated, peak_torso_rate=torso_rate, best_arm_ready_s=arm_ready,
                final_arm_error_rad=final_arm_error,
                self_contact_bodies={names[i]: time for i, time in enumerate(per_body) if time > 0},
                pull_force_n=self.pull_force_n, action_bound=self.action_bound,
                returns=dict(zip(REWARD_GROUPS, group_returns))))
        self.completed = self.completed[-4096:]
        if self.term_sums is not None:
            means = (self.term_sums[done].mean(0) / self.max_episode_length_s).cpu().tolist()
            self.extras['log'] = {f'Reward/{name}': mean for name, mean in zip(self.term_names, means)}
