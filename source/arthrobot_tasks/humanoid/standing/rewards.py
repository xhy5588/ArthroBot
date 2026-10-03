"""Dense standing reward terms (each is later multiplied by the control time step).

A valid, still pose earns positive reward; errors are penalized with Huber costs
so that large errors still give a useful gradient. Velocities are those of the
whole-robot center of mass, in m/s.
"""
import torch

ARM_JOINTS = 12


def huber(error: torch.Tensor) -> torch.Tensor:
    """Unit-threshold Huber cost: quadratic near zero, linear far away."""
    magnitude = error.abs()
    return torch.where(magnitude <= 1., .5 * error.square(), magnitude - .5)


def standing_reward_terms(gravity_b, com_velocity, angular_velocity_b, actions, previous_actions, torques,
                          wheel_contacts, com_offset_xy, heading, torso_height, target_height, arm_clearance,
                          posture_error, action_rate_weight=.10) -> dict[str, torch.Tensor]:
    """``posture_error`` lists the 12 arm joints, then the 6 leg joints (wrapped angles)."""
    upright = torch.exp(-12. * gravity_b[:, :2].square().sum(-1))
    return dict(
        upright=4. * upright,
        survival=2. * torch.ones_like(upright),
        height=6. - 4. * huber((torso_height - target_height) / .10),
        stationary=-2. * huber(com_velocity[:, :2] / .30).sum(-1),
        position=-.5 * huber(com_offset_xy / .25).sum(-1),
        wheel_contact=2. * wheel_contacts.all(-1).float(),
        arm_clearance=1.5 * (arm_clearance / .06).clamp(0., 1.),
        posture=-.25 * (huber(posture_error[:, :ARM_JOINTS] / .35).mean(-1)
                        + huber(posture_error[:, ARM_JOINTS:] / .35).mean(-1)),
        torque_cost=-.001 * torques.square().sum(-1),
        action_rate=-action_rate_weight * (actions - previous_actions).square().sum(-1),
        body_motion=-.10 * angular_velocity_b.square().sum(-1) - .5 * com_velocity[:, 2].square(),
        heading=-.5 * (1. - torch.cos(heading)))
