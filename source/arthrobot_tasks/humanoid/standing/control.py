"""Hybrid motor command: joint PD around the nominal pose, plus direct wheel torques.

Every motor shares the same 13 N m cap. Joint errors are wrapped to [-pi, pi]
because the joints are continuous.
"""
import torch

LIMB_JOINTS = 18
POSITION_SCALE_RAD = .25        # joint target = nominal + 0.25 rad x action
POSITION_GAIN_NM_PER_RAD = 80.
VELOCITY_GAIN_NM_S_PER_RAD = 4.
TORQUE_LIMIT_NM = 13.


def hybrid_torques(actions: torch.Tensor, joint_positions: torch.Tensor, joint_velocities: torch.Tensor,
                   nominal_positions: torch.Tensor, scale: float = POSITION_SCALE_RAD,
                   kp: float = POSITION_GAIN_NM_PER_RAD, kd: float = VELOCITY_GAIN_NM_S_PER_RAD,
                   limit: float = TORQUE_LIMIT_NM) -> torch.Tensor:
    """Motor torques for 20 actions in ``MOTOR_JOINTS`` order (18 limb targets, then 2 wheels)."""
    actions = torch.nan_to_num(actions).clamp(-1., 1.)
    target = nominal_positions[..., :LIMB_JOINTS] + scale * actions[..., :LIMB_JOINTS]
    error = target - joint_positions[..., :LIMB_JOINTS]
    error = torch.atan2(torch.sin(error), torch.cos(error))
    limb_torques = kp * error - kd * joint_velocities[..., :LIMB_JOINTS]
    return torch.cat((limb_torques, limit * actions[..., LIMB_JOINTS:]), -1).clamp(-limit, limit)
