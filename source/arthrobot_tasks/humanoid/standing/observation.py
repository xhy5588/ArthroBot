"""The standing policy's 96-number observation.

Used by the standing environment and, unchanged, by the get-up hand-over and the
get-up physics check, so the trained policy always sees the same inputs.

    root linear velocity (3), root angular velocity (3), projected gravity (3),
    sin and cos of the 20 motor angles (40), motor speeds / max speed (20),
    previous actions (20), wheel floor contacts (2), torso height (1),
    COM sideways offset (1), sin and cos of the heading (2), COM forward offset (1)

Offsets and heading are relative to the start of standing, in the start heading frame.
"""
import torch

OBSERVATION_SIZE = 96
CLIP = 20.


def standing_observation(robot_data, motor_ids, previous_actions: torch.Tensor, wheel_contacts: torch.Tensor,
                         torso_height: torch.Tensor, com_offset_xy: torch.Tensor, relative_heading: torch.Tensor,
                         max_motor_speed: float) -> torch.Tensor:
    joint_positions = robot_data.joint_pos[:, motor_ids]
    observation = torch.cat((
        robot_data.root_lin_vel_b, robot_data.root_ang_vel_b, robot_data.projected_gravity_b,
        torch.sin(joint_positions), torch.cos(joint_positions),
        robot_data.joint_vel[:, motor_ids] / max_motor_speed, previous_actions, wheel_contacts,
        torso_height[:, None], com_offset_xy[:, 1:2],
        torch.stack((torch.sin(relative_heading), torch.cos(relative_heading)), -1), com_offset_xy[:, 0:1]), -1)
    return observation.clamp(-CLIP, CLIP)


def heading(root_quat_w: torch.Tensor) -> torch.Tensor:
    """Yaw angle of a (w, x, y, z) quaternion."""
    w, x, y, z = root_quat_w.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
