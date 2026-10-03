"""Arm reach environment: move the tool center point (TCP) to random 3D targets.

Position-only reaching with joint-position actions on the six arm joints; the
gripper keeps its validated drive and stays open. Targets are sampled in the
mount frame (see :mod:`.mount`): 0.15-0.40 m forward, +/-0.25 m sideways and
0.05-0.40 m above the base, all reachable by numerical IK.

Training-only changes to the simulation asset (applied when spawning):
+/-180 deg joint limits, one convex hull per collision mesh, and TGS with
8 position / 1 velocity iterations at 120 Hz.
"""
from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from arthrobot.gripper import DM4310_GRIPPER
from arthrobot.motors import POSITION_GAIN_NM_PER_RAD, VELOCITY_GAIN_NM_S_PER_RAD, load_motor
from arthrobot_assets.arm import ARM_JOINTS, JAW_JOINTS, PINION_JOINT, TOOL_BODY
from arthrobot_assets.arm.usd import ensure_arm_usd, output_dir
from arthrobot_tasks.arm_reach.mount import (BASE_ANCHOR_IN_BASE, BASE_POSITION, JOINT_LIMIT_DEG, MOUNT_QUAT_WXYZ,
                                             MOUNT_ROTATION, TCP_IN_TOOL)

MOTOR = load_motor('mg5010')
COLLIDER_COUNT = 535
COLLISION_APPROXIMATION = 'convexHull'
GROUND_Z = -.75
PEDESTAL_TOP = -.005    # just below the base, so the fixed base never rests on it
COMMAND = 'tcp_target'


# ------------------------------------------------------------------ robot
@clone
def spawn_training_arm(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Build/refresh the arm USD, reference it, and apply training-only overrides to this stage copy."""
    from pxr import Usd, UsdPhysics
    ensure_arm_usd()
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
    limited_joints = hulls = 0
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdPhysics.RevoluteJoint) and child.GetName() in ARM_JOINTS:
            joint = UsdPhysics.RevoluteJoint(child)
            joint.CreateLowerLimitAttr(-JOINT_LIMIT_DEG)
            joint.CreateUpperLimitAttr(JOINT_LIMIT_DEG)
            limited_joints += 1
        if child.HasAPI(UsdPhysics.CollisionAPI):
            # Convex decomposition matters for gripper contact, not for reaching, and costs far more.
            UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr(COLLISION_APPROXIMATION)
            hulls += 1
    assert (limited_joints, hulls) == (len(ARM_JOINTS), COLLIDER_COUNT), (limited_joints, hulls)
    return prim


ARM_CFG = ArticulationCfg(
    prim_path='{ENV_REGEX_NS}/Robot',
    spawn=sim_utils.UsdFileCfg(
        func=spawn_training_arm,
        usd_path=str(output_dir() / 'arm.usd'),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=BASE_POSITION, rot=MOUNT_QUAT_WXYZ, joint_pos={'.*': 0.}),
    soft_joint_pos_limit_factor=.95,
    actuators={
        'arm': ImplicitActuatorCfg(
            joint_names_expr=list(ARM_JOINTS), stiffness=POSITION_GAIN_NM_PER_RAD, damping=VELOCITY_GAIN_NM_S_PER_RAD,
            effort_limit_sim=MOTOR.rated_torque_nm, velocity_limit_sim=MOTOR.max_speed_rad_s,
            armature=MOTOR.reflected_inertia_kg_m2),
        # One equivalent rack drive; jaw 1 and the pinion follow through mimic joints.
        'gripper': ImplicitActuatorCfg(
            joint_names_expr=[JAW_JOINTS[0]], stiffness=DM4310_GRIPPER.linear_stiffness_n_per_m,
            damping=DM4310_GRIPPER.linear_damping_n_s_per_m, effort_limit_sim=DM4310_GRIPPER.jaw_force_limit_n,
            armature=0.),
        'gripper_passive': ImplicitActuatorCfg(
            joint_names_expr=[JAW_JOINTS[1], PINION_JOINT], stiffness=0., damping=0., effort_limit_sim=0.,
            armature=0.),
    },
)


@configclass
class ArmReachSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path='/World/ground', spawn=sim_utils.GroundPlaneCfg(),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., GROUND_Z)))
    pedestal = AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/Pedestal',
        spawn=sim_utils.CylinderCfg(radius=.05, height=PEDESTAL_TOP - GROUND_Z,
                                    collision_props=sim_utils.CollisionPropertiesCfg(),
                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.35, .35, .38))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., (PEDESTAL_TOP + GROUND_Z) / 2)))
    robot: ArticulationCfg = ARM_CFG
    light = AssetBaseCfg(prim_path='/World/light', spawn=sim_utils.DomeLightCfg(color=(.75, .75, .75), intensity=2500.))


# ------------------------------------------------------------------ command
class TcpTargetCommand(CommandTerm):
    """Uniform TCP position targets in the mount frame, resampled on a timer."""

    cfg: TcpTargetCommandCfg

    def __init__(self, cfg: TcpTargetCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.tool_body_id = self.robot.find_bodies(cfg.body_name)[0][0]
        self.tcp_in_tool = torch.tensor(cfg.tcp_offset, device=self.device).repeat(self.num_envs, 1)
        self.anchor = torch.tensor(BASE_ANCHOR_IN_BASE, device=self.device)
        self.rotation = torch.tensor(MOUNT_ROTATION, dtype=torch.float, device=self.device)
        self.target_in_mount = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics['position_error'] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target_in_mount

    # Mount coordinates: p_mount = R (p_base - anchor). Row vectors below.
    def mount_to_world(self, points: torch.Tensor) -> torch.Tensor:
        in_base = self.anchor + points @ self.rotation
        return self.robot.data.root_pos_w + quat_apply(self.robot.data.root_quat_w, in_base)

    def world_to_mount(self, points: torch.Tensor) -> torch.Tensor:
        in_base = quat_apply_inverse(self.robot.data.root_quat_w, points - self.robot.data.root_pos_w)
        return (in_base - self.anchor) @ self.rotation.T

    def tcp_pos_w(self) -> torch.Tensor:
        tool_quat = self.robot.data.body_quat_w[:, self.tool_body_id]
        return self.robot.data.body_pos_w[:, self.tool_body_id] + quat_apply(tool_quat, self.tcp_in_tool)

    def target_pos_w(self) -> torch.Tensor:
        return self.mount_to_world(self.target_in_mount)

    def _update_metrics(self):
        self.metrics['position_error'] = torch.norm(self.tcp_pos_w() - self.target_pos_w(), dim=-1)

    def _resample_command(self, env_ids):
        ranges = self.cfg.ranges
        for axis, (low, high) in enumerate((ranges.pos_x, ranges.pos_y, ranges.pos_z)):
            self.target_in_mount[env_ids, axis] = torch.empty(len(env_ids), device=self.device).uniform_(low, high)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis and not hasattr(self, 'target_markers'):
            self.target_markers = VisualizationMarkers(self.cfg.target_marker_cfg)
            self.tcp_markers = VisualizationMarkers(self.cfg.tcp_marker_cfg)
        if hasattr(self, 'target_markers'):
            self.target_markers.set_visibility(debug_vis)
            self.tcp_markers.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if self.robot.is_initialized:
            self.target_markers.visualize(translations=self.target_pos_w())
            self.tcp_markers.visualize(translations=self.tcp_pos_w())


def _sphere_markers(path: str, radius: float, color: tuple) -> VisualizationMarkersCfg:
    return VisualizationMarkersCfg(prim_path=path, markers={'sphere': sim_utils.SphereCfg(
        radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))})


@configclass
class TcpTargetCommandCfg(CommandTermCfg):
    class_type: type = TcpTargetCommand
    asset_name: str = 'robot'
    body_name: str = TOOL_BODY
    tcp_offset: tuple[float, float, float] = TCP_IN_TOOL

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = (.15, .40)
        pos_y: tuple[float, float] = (-.25, .25)
        pos_z: tuple[float, float] = (.05, .40)

    ranges: Ranges = Ranges()
    target_marker_cfg: VisualizationMarkersCfg = _sphere_markers('/Visuals/Command/tcp_target', .015, (.1, .9, .1))
    tcp_marker_cfg: VisualizationMarkersCfg = _sphere_markers('/Visuals/Command/tcp_current', .01, (.95, .45, .1))


# ------------------------------------------------------------------ MDP terms
def tcp_position(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """TCP position in the mount frame."""
    command: TcpTargetCommand = env.command_manager.get_term(command_name)
    return command.world_to_mount(command.tcp_pos_w())


def tcp_distance(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: TcpTargetCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.tcp_pos_w() - command.target_pos_w(), dim=-1)


def tcp_distance_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """1 at the target, falling off over ``std`` meters."""
    return 1 - torch.tanh(tcp_distance(env, command_name) / std)


def arm_joints() -> SceneEntityCfg:
    return SceneEntityCfg('robot', joint_names=list(ARM_JOINTS), preserve_order=True)


@configclass
class CommandsCfg:
    tcp_target = TcpTargetCommandCfg(resampling_time_range=(4., 4.), debug_vis=True)


@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(asset_name='robot', joint_names=list(ARM_JOINTS), scale=.5,
                                            use_default_offset=True, preserve_order=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={'asset_cfg': arm_joints()},
                            noise=Unoise(n_min=-.01, n_max=.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={'asset_cfg': arm_joints()},
                            noise=Unoise(n_min=-.01, n_max=.01))
        tcp_position = ObsTerm(func=tcp_position, params={'command_name': COMMAND})
        target_position = ObsTerm(func=mdp.generated_commands, params={'command_name': COMMAND})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_arm = EventTerm(func=mdp.reset_joints_by_offset, mode='reset',
                          params={'asset_cfg': arm_joints(), 'position_range': (-.25, .25), 'velocity_range': (0., 0.)})


@configclass
class RewardsCfg:
    tcp_distance = RewTerm(func=tcp_distance, weight=-.2, params={'command_name': COMMAND})
    tcp_distance_fine = RewTerm(func=tcp_distance_tanh, weight=.1, params={'std': .05, 'command_name': COMMAND})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={'asset_cfg': arm_joints()})
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1., params={'asset_cfg': arm_joints()})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    # Smoothness penalties switch to full weight after 4,500 policy steps (about iteration 188).
    action_rate = CurrTerm(func=mdp.modify_reward_weight,
                           params={'term_name': 'action_rate', 'weight': -.005, 'num_steps': 4500})
    joint_vel = CurrTerm(func=mdp.modify_reward_weight,
                         params={'term_name': 'joint_vel', 'weight': -.001, 'num_steps': 4500})


@configclass
class ArmReachEnvCfg(ManagerBasedRLEnvCfg):
    scene: ArmReachSceneCfg = ArmReachSceneCfg(num_envs=1024, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4               # 120 Hz physics, 30 Hz policy
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.
        self.viewer.eye = (1.1, 1.1, .7)
        self.viewer.lookat = (.2, 0., .1)


@configclass
class ArmReachEnvCfg_PLAY(ArmReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        # Frame the first arm rather than the grid center.
        self.viewer.origin_type = 'env'
        self.viewer.env_index = 0
        self.viewer.eye = (.85, .7, .5)
        self.viewer.lookat = (.2, 0., .12)
