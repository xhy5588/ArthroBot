"""Scene with the current (rigid) arm and a loose arm side by side in every environment.

Both arms are mounted with joint 1 vertical, as in the reach task; the loose
arm sits LOOSE_OFFSET_Y to the left of the rigid one. Each arm has its own
mount frame (base bottom on the joint 1 axis, x forward, z up), and targets are
given in that frame, so both arms get the same relative targets.
"""
from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse

from arthrobot_assets.arm import ARM_JOINTS, TOOL_BODY
from arthrobot_tasks.arm_planning.loose import GEAR_JOINTS, loose_arm_cfg
from arthrobot_tasks.arm_reach.env_cfg import ARM_CFG, GROUND_Z, PEDESTAL_TOP
from arthrobot_tasks.arm_reach.mount import BASE_ANCHOR_IN_BASE, BASE_POSITION, MOUNT_ROTATION, TCP_IN_TOOL

LOOSE_OFFSET_Y = 0.7
ENV_SPACING = 2.0
# MG5010E-i36: 18-bit encoder on the motor, 14-bit on the reducer output.
OUTPUT_ENCODER_STEP = 2 * math.pi / 2**14


def make_pedestal(y: float) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/Pedestal' + ('Loose' if y else ''),
        spawn=sim_utils.CylinderCfg(
            radius=0.05, height=PEDESTAL_TOP - GROUND_Z, collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, y, (PEDESTAL_TOP + GROUND_Z) / 2)))


@configclass
class PairSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path='/World/ground', spawn=sim_utils.GroundPlaneCfg(),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, GROUND_Z)))
    pedestal = make_pedestal(0.0)
    pedestal_loose = make_pedestal(LOOSE_OFFSET_Y)
    robot = ARM_CFG
    robot_loose = loose_arm_cfg('{ENV_REGEX_NS}/RobotLoose', (BASE_POSITION[0], BASE_POSITION[1] + LOOSE_OFFSET_Y, BASE_POSITION[2]))
    light = AssetBaseCfg(prim_path='/World/light',
                         spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0))


def pair_scene_cfg(num_envs: int, offset_y: float = LOOSE_OFFSET_Y, env_spacing: float = ENV_SPACING) -> PairSceneCfg:
    """PairSceneCfg with the loose arm offset_y to the left of the rigid one."""
    cfg = PairSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    cfg.pedestal_loose.init_state.pos = (0.0, offset_y, cfg.pedestal_loose.init_state.pos[2])
    cfg.robot_loose.init_state.pos = (BASE_POSITION[0], BASE_POSITION[1] + offset_y, BASE_POSITION[2])
    return cfg


class MountFrame:
    """Mount-frame helpers for one arm (as TcpTargetCommand in the reach task)."""

    def __init__(self, robot: Articulation):
        self.robot = robot
        device = robot.device
        self.servo_ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
        # The loose arm has gearbox backlash joints between the motor and the output encoder.
        gear = [name for name in GEAR_JOINTS if name in robot.joint_names]
        self.gear_ids = robot.find_joints(gear, preserve_order=True)[0] if gear else None
        self.tool_id = robot.find_bodies(TOOL_BODY)[0][0]
        self.offset = torch.tensor(TCP_IN_TOOL, device=device).repeat(robot.num_instances, 1)
        self.anchor = torch.tensor(BASE_ANCHOR_IN_BASE, device=device)
        self.rotation = torch.tensor(MOUNT_ROTATION, dtype=torch.float, device=device)

    def to_world(self, p_m: torch.Tensor) -> torch.Tensor:
        data = self.robot.data
        return data.root_pos_w + quat_apply(data.root_quat_w, self.anchor + p_m @ self.rotation)

    def to_mount(self, p_w: torch.Tensor) -> torch.Tensor:
        data = self.robot.data
        return (quat_apply_inverse(data.root_quat_w, p_w - data.root_pos_w) - self.anchor) @ self.rotation.T

    def tcp_world(self) -> torch.Tensor:
        data = self.robot.data
        quat = data.body_quat_w[:, self.tool_id]
        return data.body_pos_w[:, self.tool_id] + quat_apply(quat, self.offset)

    def tool_rotation(self) -> torch.Tensor:
        """True rotation of tool_output in the mount frame [N, 3, 3]."""
        data = self.robot.data
        world_from_root = matrix_from_quat(data.root_quat_w)
        world_from_tool = matrix_from_quat(data.body_quat_w[:, self.tool_id])
        return self.rotation @ world_from_root.transpose(1, 2) @ world_from_tool

    def tcp(self) -> torch.Tensor:
        """True TCP position in the mount frame (what a perfect camera would measure)."""
        return self.to_mount(self.tcp_world())

    def encoders(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Servo (motor-side) joint angles and speeds: what the motor encoders read."""
        data = self.robot.data
        return data.joint_pos[:, self.servo_ids], data.joint_vel[:, self.servo_ids]

    def output_encoders(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Angles after the gearbox, quantized to the 14-bit output encoder, and speeds."""
        q, qd = self.encoders()
        if self.gear_ids is not None:
            q = q + self.robot.data.joint_pos[:, self.gear_ids]
            qd = qd + self.robot.data.joint_vel[:, self.gear_ids]
        return torch.round(q / OUTPUT_ENCODER_STEP) * OUTPUT_ENCODER_STEP, qd
