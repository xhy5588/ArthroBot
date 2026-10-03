"""Hybrid standing environment for the wheel-legged humanoid (Isaac Lab direct env).

Episodes last 15 s and start from the nominal standing pose with a small random
tilt (+/-0.015 rad). An episode fails when the torso tilts more than 30 deg, drops
below 0.32 m, touches the floor with anything but the wheels, or drifts away.
Physics: 240 Hz, PGS 32/8, 60 Hz control. The scene must be built with
:func:`arthrobot_tasks.humanoid.assets.ensure_training_usd` first, and
``cfg.nominal_pose`` must be set (from the checkpoint settings or
:func:`arthrobot_assets.humanoid.nominal_pose.solve_pose`).
"""
import itertools
import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from arthrobot.motors import load_motor
from arthrobot_assets.humanoid import LIMB_JOINT_COUNT, MOTOR_JOINTS, WHEEL_BODIES
from arthrobot_assets.humanoid.training_model import load_training_model, output_dir
from arthrobot_tasks.humanoid.standing.control import (POSITION_GAIN_NM_PER_RAD, POSITION_SCALE_RAD,
                                                       TORQUE_LIMIT_NM, VELOCITY_GAIN_NM_S_PER_RAD, hybrid_torques)
from arthrobot_tasks.humanoid.standing.observation import OBSERVATION_SIZE, heading, standing_observation
from arthrobot_tasks.humanoid.standing.rewards import standing_reward_terms

MOTOR = load_motor('mg5010')
REWARD_TERMS = ('upright', 'survival', 'height', 'stationary', 'position', 'wheel_contact', 'arm_clearance',
                'posture', 'torque_cost', 'action_rate', 'body_motion', 'heading')
WHEEL_CONTACT_FORCE_N = 3.
FALL_TILT_DEG, FALL_HEIGHT_M = 30., .32
BODY_CONTACT_FORCE_N, BODY_CONTACT_GRACE_STEPS = 10., 18
MAX_DRIFT_FORWARD_M, MAX_DRIFT_SIDEWAYS_M = 12., 2.
FAILURE_REWARD = -10.
TERMINATION_REASONS = ('lane_or_invalid_state', 'non_wheel_floor_contact', 'torso_height', 'torso_tilt', 'timeout')


def floor_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(static_friction=.9, dynamic_friction=.8, restitution=0.,
                                          friction_combine_mode='average')


@configclass
class HumanoidStandingEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 15.
    action_space = len(MOTOR_JOINTS)
    observation_space = OBSERVATION_SIZE
    state_space = 0
    randomize_resets = True
    action_rate_weight = .10
    nominal_pose: dict | None = None
    position_scale = POSITION_SCALE_RAD
    position_kp = POSITION_GAIN_NM_PER_RAD
    position_kd = VELOCITY_GAIN_NM_S_PER_RAD
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 240, render_interval=decimation,
        physx=sim_utils.PhysxCfg(solver_type=0, gpu_max_rigid_contact_count=2**20, gpu_max_rigid_patch_count=2**17,
                                gpu_found_lost_pairs_capacity=2**20),
        physics_material=floor_material())
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=30., replicate_physics=True)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path='/World/envs/env_.*/Robot',
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(output_dir() / 'robot.usd'), activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=32, solver_velocity_iteration_count=8)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., .5), joint_pos={'.*': 0.}),
        # Implicit drives are off: torques are computed explicitly every physics step.
        actuators={'motors': ImplicitActuatorCfg(
            joint_names_expr=list(MOTOR_JOINTS), stiffness=0., damping=0., effort_limit_sim=TORQUE_LIMIT_NM,
            velocity_limit_sim=MOTOR.max_speed_rad_s, armature=MOTOR.reflected_inertia_kg_m2)},
        soft_joint_pos_limit_factor=1.)
    contacts: ContactSensorCfg = ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/.*', history_length=1,
                                                  update_period=1 / 240)

    def set_nominal_pose(self, pose: dict) -> None:
        """Start every episode from ``pose`` (output of ``solve_pose`` or a checkpoint's settings)."""
        self.nominal_pose = pose
        self.robot.init_state.pos = (0., 0., pose['initial_height_m'])
        self.robot.init_state.rot = tuple(pose['initial_quaternion_wxyz'])
        self.robot.init_state.joint_pos = {name: pose['joint_positions'].get(name, 0.) for name in MOTOR_JOINTS}


class HumanoidStandingEnv(DirectRLEnv):
    cfg: HumanoidStandingEnvCfg

    def __init__(self, cfg: HumanoidStandingEnvCfg, render_mode=None, **kwargs):
        if cfg.nominal_pose is None:
            raise ValueError('Set cfg.nominal_pose (cfg.set_nominal_pose) before creating the environment')
        super().__init__(cfg, render_mode, **kwargs)
        _, model = load_training_model()
        self.motor_ids, names = self.robot.find_joints(list(MOTOR_JOINTS), preserve_order=True)
        assert names == list(MOTOR_JOINTS) and len(set(self.motor_ids)) == self.robot.num_joints == 20
        self.limb_ids = self.motor_ids[:LIMB_JOINT_COUNT]
        self.wheel_sensor_ids, _ = self.contacts.find_bodies(list(WHEEL_BODIES), preserve_order=True)
        self.body_sensor_ids = [i for i, name in enumerate(self.contacts.body_names) if not name.endswith('wheel_body')]
        n, device = self.num_envs, self.device
        self.actions = torch.zeros(n, len(MOTOR_JOINTS), device=device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.last_torque = torch.zeros_like(self.actions)
        self.masses = self.robot.data.default_mass.to(device).unsqueeze(-1)
        self.start_com = torch.zeros(n, 3, device=device)
        # Arm collision boxes (8 corners each) for the arm-clearance reward.
        arm_bodies = list(model['arm_collision_bounds_m'])
        self.arm_body_ids, _ = self.robot.find_bodies(arm_bodies, preserve_order=True)
        corners = [[[high[axis] if bit[axis] else low[axis] for axis in range(3)]
                    for bit in itertools.product((0, 1), repeat=3)]
                   for low, high in (model['arm_collision_bounds_m'][name] for name in arm_bodies)]
        self.arm_corners = torch.tensor(corners, device=device, dtype=torch.float32)
        # Episode statistics.
        self.episode_sums = {name: torch.zeros(n, device=device) for name in REWARD_TERMS}
        self.speed_integral = torch.zeros(n, device=device)
        self.peak_speed = torch.zeros_like(self.speed_integral)
        self.max_tilt = torch.zeros_like(self.speed_integral)
        self.grounded_time = torch.zeros_like(self.speed_integral)
        self.horizontal_speed_integral = torch.zeros_like(self.speed_integral)
        self.minimum_height = torch.full_like(self.speed_integral, float('inf'))
        self.control_stats = torch.zeros(3, device=device)   # torque saturation, torque^2, action change^2
        self.control_stat_steps = 0
        self.total_resets = self.total_successes = 0
        self.completed_episode_records = []

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations['robot'] = self.robot
        sim_utils.spawn_ground_plane('/World/ground', sim_utils.GroundPlaneCfg(
            size=(200., 200.), color=(.18, .21, .25), physics_material=floor_material()))
        self.contacts = ContactSensor(self.cfg.contacts)
        self.scene.sensors['contacts'] = self.contacts
        self.scene.clone_environments(copy_from_source=False)
        if self.device == 'cpu':
            self.scene.filter_collisions(global_prim_paths=['/World/ground'])
        light = sim_utils.DomeLightCfg(intensity=900., color=(.9, .95, 1.))
        light.func('/World/Light', light)
        if self.sim.has_gui():
            self._add_floor_grid()
        # Contact filtering is one sensor body to many filter bodies, so each body gets its own
        # floor-only sensor; this separates floor support from self-contact.
        floor = next(str(prim.GetPath()) for prim in self.scene.stage.Traverse()
                     if str(prim.GetPath()).startswith('/World/ground/') and prim.GetTypeName() == 'Plane')
        self.floor_sensors = {}
        for name in load_training_model()[1]['body_masses']:
            sensor = ContactSensor(ContactSensorCfg(prim_path=f'/World/envs/env_.*/Robot/{name}',
                                                    filter_prim_paths_expr=[floor], update_period=1 / 240))
            self.scene.sensors['floor_' + name] = sensor
            self.floor_sensors[name] = sensor

    def _add_floor_grid(self):
        """Non-colliding 0.5 m grid lines, for reading motion in the viewer."""
        from pxr import Gf, UsdGeom
        for axis in range(2):
            for index in range(-8, 9):
                line = UsdGeom.Cube.Define(self.scene.stage, f'/World/Grid/line_{axis}_{index + 8}')
                line.CreateSizeAttr(1.)
                line.CreateDisplayColorAttr([(.32, .40, .47)])
                xform = UsdGeom.Xformable(line)
                xform.AddTranslateOp().Set(Gf.Vec3d(index * .5 if axis else 0., 0. if axis else index * .5, .001))
                xform.AddScaleOp().Set(Gf.Vec3f(.006 if axis else 8., 8. if axis else .006, .001))

    # ------------------------------------------------------------------ state helpers
    def floor_forces(self) -> torch.Tensor:
        """Floor contact force on every body, (envs, bodies, 3)."""
        return torch.stack([self.floor_sensors[name].data.force_matrix_w[:, 0].sum(1)
                            for name in self.contacts.body_names], 1)

    def com_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.robot.data
        total = self.masses.sum(1)
        return (data.body_com_pos_w * self.masses).sum(1) / total, (data.body_com_lin_vel_w * self.masses).sum(1) / total

    def heading(self) -> torch.Tensor:
        return heading(self.robot.data.root_quat_w)

    def torso_height(self) -> torch.Tensor:
        return self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]

    def arm_clearance(self) -> torch.Tensor:
        """Height of the lowest arm collision-box corner above the floor."""
        data = self.robot.data
        points = self.arm_corners.unsqueeze(0).expand(self.num_envs, -1, -1, -1)
        quat = data.body_link_quat_w[:, self.arm_body_ids, None, :].expand(-1, -1, 8, -1)
        points_w = quat_apply(quat.reshape(-1, 4), points.reshape(-1, 3)).reshape_as(points)
        heights = points_w[..., 2] + data.body_link_pos_w[:, self.arm_body_ids, None, 2]
        return heights.amin(dim=(1, 2)) - self.scene.env_origins[:, 2]

    def wheel_contacts(self) -> torch.Tensor:
        return self.floor_forces()[:, self.wheel_sensor_ids, 2] > WHEEL_CONTACT_FORCE_N

    # ------------------------------------------------------------------ control
    def commanded_torques(self, actions: torch.Tensor) -> torch.Tensor:
        data = self.robot.data
        return hybrid_torques(actions, data.joint_pos[:, self.motor_ids], data.joint_vel[:, self.motor_ids],
                              data.default_joint_pos[:, self.motor_ids], self.cfg.position_scale,
                              self.cfg.position_kp, self.cfg.position_kd)

    def _pre_physics_step(self, actions):
        self.previous_actions[:] = self.actions
        self.actions[:] = torch.nan_to_num(actions).clamp(-1., 1.)
        self.last_torque[:] = self.commanded_torques(self.actions)

    def _apply_action(self):
        # The PD feedback is recomputed every physics step; PhysX's implicit drives stay off.
        self.last_torque[:] = self.commanded_torques(self.actions)
        self.robot.set_joint_effort_target(self.last_torque, joint_ids=self.motor_ids)

    # ------------------------------------------------------------------ MDP
    def _get_observations(self):
        com, _ = self.com_state()
        observation = standing_observation(self.robot.data, self.motor_ids, self.actions,
                                           self.wheel_contacts().float(), self.torso_height(),
                                           com[:, :2] - self.start_com[:, :2], self.heading(), MOTOR.max_speed_rad_s)
        assert observation.shape == (self.num_envs, self.cfg.observation_space)
        return {'policy': observation}

    def _get_rewards(self):
        data = self.robot.data
        com, velocity = self.com_state()
        grounded = self.wheel_contacts()
        posture = data.joint_pos[:, self.limb_ids] - data.default_joint_pos[:, self.limb_ids]
        posture = torch.atan2(torch.sin(posture), torch.cos(posture))
        terms = standing_reward_terms(
            data.projected_gravity_b, velocity, data.root_ang_vel_b, self.actions, self.previous_actions,
            self.last_torque, grounded, com[:, :2] - self.start_com[:, :2], self.heading(), self.torso_height(),
            self.cfg.nominal_pose['initial_height_m'], self.arm_clearance(), posture, self.cfg.action_rate_weight)
        reward = sum(terms.values()) * self.step_dt + FAILURE_REWARD * self.reset_terminated.float()
        for name, value in terms.items():
            self.episode_sums[name] += value * self.step_dt
        self.speed_integral += velocity[:, 0] * self.step_dt
        self.peak_speed = torch.maximum(self.peak_speed, velocity[:, 0])
        tilt = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1., 1.))
        self.max_tilt = torch.maximum(self.max_tilt, tilt)
        self.grounded_time += grounded.all(-1).float() * self.step_dt
        self.horizontal_speed_integral += velocity[:, :2].norm(dim=-1) * self.step_dt
        self.minimum_height = torch.minimum(self.minimum_height, self.torso_height())
        self.control_stats += torch.stack(((self.last_torque.abs() >= TORQUE_LIMIT_NM - .01).float().mean(),
                                           self.last_torque.square().mean(),
                                           (self.actions - self.previous_actions).square().mean()))
        self.control_stat_steps += 1
        return reward

    def _get_dones(self):
        data = self.robot.data
        com, _ = self.com_state()
        # Upright means the torso Z axis is close to world Z.
        fallen = -data.projected_gravity_b[:, 2] < math.cos(math.radians(FALL_TILT_DEG))
        fallen |= self.torso_height() < FALL_HEIGHT_M
        body_contact = self.floor_forces()[:, self.body_sensor_ids].norm(dim=-1).amax(-1) > BODY_CONTACT_FORCE_N
        fallen |= body_contact & (self.episode_length_buf > BODY_CONTACT_GRACE_STEPS)
        escaped = ((com[:, 0] - self.start_com[:, 0]).abs() > MAX_DRIFT_FORWARD_M) | (
            (com[:, 1] - self.start_com[:, 1]).abs() > MAX_DRIFT_SIDEWAYS_M)
        invalid = ~torch.isfinite(data.root_state_w).all(-1) | ~torch.isfinite(data.joint_pos).all(-1)
        return fallen | escaped | invalid, self.episode_length_buf >= self.max_episode_length - 1

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if hasattr(self, 'start_com'):   # skip during DirectRLEnv construction
            self._record_finished_episodes(env_ids)
        super()._reset_idx(env_ids)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        if self.cfg.randomize_resets:
            roll_pitch_yaw = torch.tensor(self.cfg.nominal_pose['initial_rpy'], device=self.device) + (
                torch.rand(len(env_ids), 3, device=self.device) * 2 - 1) * .015
            root[:, 3:7] = quat_from_euler_xyz(*roll_pitch_yaw.T)
        joint_positions = self.robot.data.default_joint_pos[env_ids].clone()
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_positions, torch.zeros_like(joint_positions), env_ids=env_ids)
        self.robot.set_joint_effort_target(torch.zeros_like(joint_positions), env_ids=env_ids)
        if hasattr(self, 'start_com'):
            com_in_torso = torch.tensor(self.cfg.nominal_pose['center_of_mass_local_m'],
                                        device=self.device).expand(len(env_ids), -1)
            self.start_com[env_ids] = root[:, :3] + quat_apply(root[:, 3:7], com_in_torso)

    def _record_finished_episodes(self, env_ids):
        durations = self.episode_length_buf[env_ids].float() * self.step_dt
        finished = durations > 0
        survived = self.reset_time_outs[env_ids] & ~self.reset_terminated[env_ids]
        self.total_resets += int(finished.sum())
        self.total_successes += int((finished & survived).sum())
        if bool(finished.any()):
            com, _ = self.com_state()
            ids = env_ids[finished]
            duration = durations[finished]
            data = self.robot.data
            contact_forces = self.floor_forces()[ids][:, self.body_sensor_ids].norm(dim=-1)
            strongest_force, strongest_body = contact_forces.max(-1)
            reason = torch.zeros_like(strongest_force)
            reason[(strongest_force > BODY_CONTACT_FORCE_N) & (self.episode_length_buf[ids] > BODY_CONTACT_GRACE_STEPS)] = 1
            reason[data.root_pos_w[ids, 2] - self.scene.env_origins[ids, 2] < FALL_HEIGHT_M] = 2
            reason[-data.projected_gravity_b[ids, 2] < math.cos(math.radians(FALL_TILT_DEG))] = 3
            reason[survived[finished]] = 4
            rows = torch.stack((
                ids.float(), duration, survived[finished].float(), com[ids, 0] - self.start_com[ids, 0],
                self.speed_integral[ids] / duration, self.peak_speed[ids], self.max_tilt[ids] * 180 / math.pi,
                self.grounded_time[ids] / duration, reason, strongest_body.float(), strongest_force,
                self.horizontal_speed_integral[ids] / duration, self.minimum_height[ids],
                com[ids, 1] - self.start_com[ids, 1]), -1).cpu().tolist()
            for (env, seconds, success, forward, mean_speed, peak_speed, tilt, wheel_contact, reason_index,
                 body_index, force, horizontal_speed, height, lateral) in rows:
                self.completed_episode_records.append(dict(
                    env=int(env), seconds=seconds, survived=bool(success), forward_m=forward,
                    mean_speed_m_s=mean_speed, peak_speed_m_s=peak_speed, max_torso_tilt_deg=tilt,
                    both_wheels_contact_fraction=wheel_contact, termination_reason=TERMINATION_REASONS[int(reason_index)],
                    strongest_non_wheel_contact_body=self.contacts.body_names[self.body_sensor_ids[int(body_index)]],
                    strongest_non_wheel_contact_n=force, mean_horizontal_speed_m_s=horizontal_speed,
                    minimum_torso_height_m=height, lateral_m=lateral))
            self.completed_episode_records = self.completed_episode_records[-2048:]
        self.extras['log'] = {f'Episode_Reward/{name}': values[env_ids].mean() / self.max_episode_length_s
                              for name, values in self.episode_sums.items()}
        self.extras['log'].update({'Episode/seconds': durations.mean(),
                                   'Episode/survived_fraction': survived.float().mean(),
                                   'Episode/mean_speed_m_s': (self.speed_integral[env_ids]
                                                              / durations.clamp_min(self.step_dt)).mean()})
        for values in (*self.episode_sums.values(), self.actions, self.previous_actions, self.last_torque,
                       self.speed_integral, self.peak_speed, self.max_tilt, self.grounded_time,
                       self.horizontal_speed_integral):
            values[env_ids] = 0.
        self.minimum_height[env_ids] = float('inf')
