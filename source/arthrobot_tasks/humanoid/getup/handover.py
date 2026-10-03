"""Run the standing policy inside the get-up environment (hand-over and physics checks).

The standing policy was trained from upright starts at a yaw of about 0. Here its
heading and center-of-mass inputs are measured relative to the moment it takes
over, in the heading frame of that moment, so it sees the inputs it was trained
on from any start. Robots in standing mode receive the standing policy's hybrid
torques (nominal pose + 0.25 rad x action with PD 80/4, wheel torque 13 N m x action)
instead of the get-up PD control.
"""
import torch

from arthrobot_tasks.humanoid.getup.env import MAX_SPEED, WHEEL_CONTACT_FORCE_N, GetupEnv
from arthrobot_tasks.humanoid.standing.control import hybrid_torques
from arthrobot_tasks.humanoid.standing.observation import heading, standing_observation

# A robot "stayed standing" when, over the measured period, its tilt stayed below 20 deg,
# its torso above 0.46 m, and both wheels were on the floor at least 80% of the time.
MAX_TILT_DEG = 20.
MIN_TORSO_HEIGHT_M = .46
MIN_WHEEL_SUPPORT_FRACTION = .8


class HandoverEnv(GetupEnv):
    """Get-up environment in which selected robots are driven by the standing policy."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n = self.num_envs
        self.standing_mode = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.handover_com = torch.zeros(n, 3, device=self.device)
        self.handover_heading = torch.zeros(n, device=self.device)
        self.masses = self.robot.data.default_mass.to(self.device).unsqueeze(-1)

    def center_of_mass(self) -> torch.Tensor:
        return (self.robot.data.body_com_pos_w * self.masses).sum(1) / self.masses.sum(1)

    def heading(self) -> torch.Tensor:
        return heading(self.robot.data.root_quat_w)

    def hand_over(self, env_ids: torch.Tensor) -> None:
        """Switch robots to the standing policy, which observes its own previous action (zero at first)."""
        self.handover_com[env_ids] = self.center_of_mass()[env_ids]
        self.handover_heading[env_ids] = self.heading()[env_ids]
        self.standing_mode[env_ids] = True
        self.actions[env_ids] = 0.

    def standing_policy_observation(self) -> torch.Tensor:
        offset = self.center_of_mass()[:, :2] - self.handover_com[:, :2]
        cos, sin = torch.cos(self.handover_heading), torch.sin(self.handover_heading)
        forward = cos * offset[:, 0] + sin * offset[:, 1]
        sideways = -sin * offset[:, 0] + cos * offset[:, 1]
        return standing_observation(self.robot.data, self.motor_ids, self.actions, self.wheels_down().float(),
                                    self.torso_height(), torch.stack((forward, sideways), -1),
                                    self.heading() - self.handover_heading, MAX_SPEED)

    def wheels_down(self) -> torch.Tensor:
        """[N, 2] both wheels pressing on the floor."""
        return self.floor_forces()[:, self.wheel_sensor_ids, 2] > WHEEL_CONTACT_FORCE_N

    def tilt_deg(self) -> torch.Tensor:
        return torch.rad2deg(torch.acos((-self.robot.data.projected_gravity_b[:, 2]).clamp(-1., 1.)))

    def _apply_action(self):
        super()._apply_action()
        data = self.robot.data
        standing_torques = hybrid_torques(self.actions, data.joint_pos[:, self.motor_ids],
                                          data.joint_vel[:, self.motor_ids], self.nominal)
        self.torque[:] = torch.where(self.standing_mode[:, None], standing_torques, self.torque)
        self.robot.set_joint_effort_target(self.torque, joint_ids=self.motor_ids)


class StandingTracker:
    """Per robot, over the steps in which it is tracked: peak tilt, lowest torso height and wheel support."""

    def __init__(self, env: HandoverEnv):
        n, device = env.num_envs, env.device
        self.env = env
        self.steps = torch.zeros(n, device=device)
        self.max_tilt_deg = torch.zeros(n, device=device)
        self.min_height_m = torch.full((n,), float('inf'), device=device)
        self.wheel_support_steps = torch.zeros(n, device=device)

    def update(self, tracked: torch.Tensor) -> None:
        env = self.env
        self.steps += tracked.float()
        self.max_tilt_deg = torch.where(tracked, torch.maximum(self.max_tilt_deg, env.tilt_deg()), self.max_tilt_deg)
        self.min_height_m = torch.where(tracked, torch.minimum(self.min_height_m, env.torso_height()), self.min_height_m)
        self.wheel_support_steps += env.wheels_down().all(-1).float() * tracked

    def wheel_support_fraction(self) -> torch.Tensor:
        return self.wheel_support_steps / self.steps.clamp(min=1)

    def stayed_standing(self) -> torch.Tensor:
        return ((self.steps > 0) & (self.max_tilt_deg < MAX_TILT_DEG) & (self.min_height_m > MIN_TORSO_HEIGHT_M)
                & (self.wheel_support_fraction() >= MIN_WHEEL_SUPPORT_FRACTION))
