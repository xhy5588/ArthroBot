"""IK controller for the arm: what a controller on the real robot can run. Torch only; no Isaac imports.

It works on a batch of arms with the inputs a real robot has: the six joint angles
from the motor encoders or the 14-bit output encoders and, for the camera variant,
the true TCP pose. Commands are servo position targets.

IK on the rigid model (:mod:`.kinematics`), for the TCP position (spare joints kept
near a rest posture) or the full gripper pose; a minimum-jerk joint path to the
solution; and gravity compensation (each target offset by gravity torque / servo
stiffness). It knows nothing about obstacles: for collision-free motion see
:mod:`.curobo_robot` and ``scripts/arm/plan_trajectories.py``. Optional corrections
once a move ends:

- camera: integrate the measured TCP error (and, for pose targets, the gripper's
  rotation error) into the IK goal;
- output encoders: integrate each gravity-loaded joint's error, read after the
  gearbox, into its servo target.
"""
import math

import torch

from arthrobot.motors import POSITION_GAIN_NM_PER_RAD
from arthrobot_tasks.arm_planning.kinematics import ArmModel, rotation_exp, rotation_log

SERVO_STIFFNESS = POSITION_GAIN_NM_PER_RAD  # N m/rad, the MG5010 drive in the simulation model.
IK_PEAK_SPEED = 1.5  # rad/s, peak joint speed of the minimum-jerk path.
IK_MIN_DURATION = 1.0  # s
CAMERA_GAIN = 2.0  # 1/s, integral gain of the TCP correction.
OUTPUT_GAIN = 2.0  # 1/s, integral gain of the joint correction from the output encoders.
OUTPUT_DEADBAND = 1.5 * 2 * math.pi / 2**14  # rad: ignore errors within 1.5 counts of the 14-bit encoder.
# Correcting every joint made unloaded ones hunt (loose, play in gearbox: shake 7.8 mm instead of 2.0);
# loaded joints only kept most of the gain (error 5.2 mm instead of 4.6, plain IK 9.3) at 2.3 mm shake.
OUTPUT_MIN_GRAVITY_TORQUE = 0.3  # N m
# IK secondary goal: keep joints 4-6 near zero, shoulder and elbow loosely near zero, base yaw free.
# Without it the wrist folded the gripper into the forearm.
IK_REST_WEIGHTS = (0., 0.2, 0.2, 1., 1., 1.)


def rotation_angle(a, b):
    """Angle [N] between rotations a and b [N, 3, 3]."""
    cosine = ((a * b).sum((-2, -1)) - 1) / 2
    return torch.acos(cosine.clamp(-1., 1.))


def minimum_jerk(s):
    s = s.clamp(0., 1.)
    return s**3 * (10 - 15 * s + 6 * s**2)


class IKController:
    def __init__(self, model: ArmModel, num_arms, device, camera_mask, output_mask):
        self.model = model
        self.camera = camera_mask  # [num_arms] bool: these arms also use the true TCP.
        self.output = output_mask  # [num_arms] bool: these arms correct joints with the output encoders.
        z = lambda *shape: torch.zeros(num_arms, *shape, dtype=torch.float64, device=device)
        self.q_from, self.q_goal, self.q_desired = z(6), z(6), z(6)
        self.elapsed, self.duration = z(), torch.ones(num_arms, dtype=torch.float64, device=device)
        self.bias, self.rotation_bias, self.target, self.joint_bias = z(3), z(3), z(3), z(6)
        self.ik_error, self.ik_rotation_error = z(), z()
        self.ik_valid = torch.ones(num_arms, dtype=torch.bool, device=device)
        self.target_rot, self.approach_only = None, False
        self.rest = z(6)
        self.rest_weights = torch.tensor(IK_REST_WEIGHTS, dtype=torch.float64, device=device)

    def solve(self, target, q0, iterations=100):
        return self.model.ik(target, q0, iterations=iterations, rest=self.rest, rest_weights=self.rest_weights)

    def new_target(self, target, target_rot=None, approach_only=False):
        """target [N, 3] TCP position; target_rot [N, 3, 3] gripper axes, or None for position only."""
        self.target = target.to(torch.float64)
        self.bias.zero_()
        self.rotation_bias.zero_()
        self.joint_bias.zero_()
        self.q_from = self.q_desired.clone()
        if target_rot is None:
            self.target_rot = None
            self.q_goal, self.ik_error = self.solve(self.target, self.q_from)
        else:
            self.approach_only = approach_only
            self.q_goal, self.ik_error, self.ik_rotation_error, self.ik_valid = self.model.ik_pose(
                self.target, target_rot, self.q_from, approach_only=approach_only, rest=self.rest,
                rest_weights=self.rest_weights)
            # The IK may have used the 180-deg-turned grasp; keep whichever one it reached.
            _, reached = self.model.pose(self.q_goal)
            target_rot = target_rot.to(torch.float64)
            flipped = target_rot @ torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.float64, device=target.device))
            use_flip = rotation_angle(flipped, reached) < rotation_angle(target_rot, reached)
            self.target_rot = torch.where(use_flip[:, None, None], flipped, target_rot)
        travel = (self.q_goal - self.q_from).abs().max(-1).values
        # Minimum-jerk peak speed is 1.875 x the average speed.
        self.duration = (1.875 * travel / IK_PEAK_SPEED).clamp(min=IK_MIN_DURATION)
        self.elapsed.zero_()

    def command(self, true_tcp, q_output, dt, true_rot=None):
        """true_tcp and true_rot (gripper axes) are used only by the camera arms."""
        self.elapsed += dt
        arrived = self.elapsed >= self.duration
        correcting = self.camera & arrived
        if bool(correcting.any()):
            error = self.target - true_tcp.to(torch.float64)
            self.bias[correcting] += CAMERA_GAIN * dt * error[correcting]
            goal = self.target[correcting] + self.bias[correcting]
            if self.target_rot is None:
                q, _ = self.model.ik(goal, self.q_goal[correcting], iterations=3, rest=self.rest[correcting],
                                     rest_weights=self.rest_weights)
            else:
                rotation_error = rotation_log(self.target_rot @ true_rot.to(torch.float64).transpose(1, 2))
                self.rotation_bias[correcting] += CAMERA_GAIN * dt * rotation_error[correcting]
                goal_rot = rotation_exp(self.rotation_bias[correcting]) @ self.target_rot[correcting]
                q = self.model.ik_pose(goal, goal_rot, self.q_goal[correcting],
                                       approach_only=self.approach_only, use_seeds=False, symmetric=False,
                                       iterations=3)[0]
            self.q_goal[correcting] = q
        blend = minimum_jerk(self.elapsed / self.duration)[:, None]
        self.q_desired = self.q_from + blend * (self.q_goal - self.q_from)
        correcting = self.output & arrived
        gravity = self.model.gravity_torque(self.q_desired)
        if bool(correcting.any()):
            error = self.q_desired - q_output.to(torch.float64)
            # Only joints that gravity holds against one side of their backlash: an unloaded joint
            # floats inside its gap, and correcting it makes the joint hunt back and forth.
            keep = (error.abs() > OUTPUT_DEADBAND) & (gravity.abs() > OUTPUT_MIN_GRAVITY_TORQUE)
            error = torch.where(keep, error, torch.zeros_like(error))
            self.joint_bias[correcting] += OUTPUT_GAIN * dt * error[correcting]
        # PD equilibrium: SERVO_STIFFNESS * (command - q) + gravity torque = 0.
        return (self.q_desired + self.joint_bias - gravity / SERVO_STIFFNESS).float()
