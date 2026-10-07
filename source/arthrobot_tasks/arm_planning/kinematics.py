"""Rigid kinematic and mass model of the arm from its URDF, batched in torch.

This is what a controller on the real robot can compute from its joint encoders:
forward kinematics, the TCP Jacobian, gravity torques and numerical IK, for the
TCP position alone or for the full gripper pose. It knows nothing about play or
flex. Everything is in the training mount frame (origin at the base bottom on
the joint 1 axis, x forward, z up; see :mod:`arthrobot_tasks.arm_reach.mount`).
No Isaac Sim imports.

Gripper frame (at the TCP): z = approach, the direction the gripper points
(tool x); y = the jaw closing axis (gripper_jaw_1's slide axis); x = y x z. The
jaws are symmetric, so a grasp is the same with the gripper turned 180 deg about z.

The joints are not a textbook arm: J1 base yaw; J2 shoulder pitch; J3 rolls the
upper arm about its length; J4 and J5 are parallel (elbow and wrist bend in one
plane); J6 rolls the gripper about its approach axis. There is no closed-form IK,
so :meth:`ArmModel.ik_pose` solves numerically from several start poses.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from arthrobot import paths
from arthrobot.urdf import origin_matrix
from arthrobot_assets.arm import ARM_JOINTS, TOOL_BODY
from arthrobot_tasks.arm_reach.mount import BASE_ANCHOR_IN_BASE, MOUNT_ROTATION, TCP_IN_TOOL

GRAVITY = 9.81
# Joint 5 beyond about 70 deg folds the gripper into the forearm (seen in simulation); with joint 4
# bent too, contact started at 64 deg in a top-down grasp.
WRIST_LIMIT = np.radians(60.)
# Start poses for ik_pose, besides the caller's: elbow and upper-arm roll both ways.
IK_SEEDS_DEG = ((0, 20, 45, -30, -40, 0), (0, 20, -45, 30, 40, 0), (0, 45, 60, -60, -30, 0),
                (0, 45, -60, 60, 30, 0), (0, 0, 0, 0, 0, 0), (0, 30, 90, -45, -45, 90))


def arm_urdf() -> Path:
    """``build/arm/arm.urdf``, built from the CAD export if missing."""
    path = paths.BUILD_DIR / 'arm' / 'arm.urdf'
    if not path.exists():
        from arthrobot_assets.arm.build import write
        path, _ = write()
    return path


class ArmModel:
    """Batched FK, Jacobian, gravity torque and IK for the six arm joints.

    The gripper is treated as fixed to tool_output in its open state (all gripper
    joints at zero), as in the reach task.
    """

    def __init__(self, device='cpu', urdf: Path | None = None):
        self.device = device
        root = ET.parse(urdf or arm_urdf()).getroot()
        joints = {j.get('name'): j for j in root.findall('joint')}
        links = {l.get('name'): l for l in root.findall('link')}
        f64 = dict(dtype=torch.float64, device=device)
        self.origins, self.axes, self.chain = [], [], []
        for name in ARM_JOINTS:
            joint = joints[name]
            self.origins.append(torch.tensor(origin_matrix(joint.find('origin')), **f64))
            self.axes.append(torch.tensor(np.fromstring(joint.find('axis').get('xyz'), sep=' '), **f64))
            self.chain.append(joint.find('child').get('link'))
        assert self.chain[-1] == TOOL_BODY
        # Bodies carried by tool_output with the gripper open: fixed transforms.
        self.fixed = {j.find('child').get('link'): torch.tensor(origin_matrix(j.find('origin')), **f64)
                      for j in joints.values() if j.find('parent').get('link') == TOOL_BODY}
        # Mass properties (URDF inertial origins have zero rpy: inertia in the link frame at the COM).
        self.mass, self.com, self.inertia = {}, {}, {}
        for name in self.chain + list(self.fixed):
            inertial = links[name].find('inertial')
            self.mass[name] = float(inertial.find('mass').get('value'))
            self.com[name] = torch.tensor(np.fromstring(inertial.find('origin').get('xyz'), sep=' '), **f64)
            i = {k: float(v) for k, v in inertial.find('inertia').attrib.items()}
            self.inertia[name] = torch.tensor([[i['ixx'], i['ixy'], i['ixz']], [i['ixy'], i['iyy'], i['iyz']],
                                               [i['ixz'], i['iyz'], i['izz']]], **f64)
        self.rotation = torch.tensor(MOUNT_ROTATION, **f64)
        self.anchor = torch.tensor(BASE_ANCHOR_IN_BASE, **f64)
        self.tcp_offset = torch.tensor(TCP_IN_TOOL, **f64)
        # Gripper frame in the tool_output frame: z along the TCP offset, y along the jaw slide axis.
        jaw = joints['gripper_jaw_1']
        slide = origin_matrix(jaw.find('origin'))[:3, :3] @ np.fromstring(jaw.find('axis').get('xyz'), sep=' ')
        approach = np.asarray(TCP_IN_TOOL) / np.linalg.norm(TCP_IN_TOOL)
        closing = slide - approach * (slide @ approach)
        closing /= np.linalg.norm(closing)
        self.gripper_in_tool = torch.tensor(np.column_stack([np.cross(closing, approach), closing, approach]), **f64)
        self.gravity = torch.tensor([0., 0., -GRAVITY], **f64)

    def _to_mount(self, transform):
        """Base-frame (CAD) transforms [..., 4, 4] -> (positions, rotations) in the mount frame."""
        position = (transform[..., :3, 3] - self.anchor) @ self.rotation.T
        return position, self.rotation @ transform[..., :3, :3]

    def frames(self, q):
        """q [N, 6] -> dict with joint positions/axes [N, 6, 3] and body poses, all in the mount frame."""
        q = q.to(torch.float64)
        transform = torch.eye(4, dtype=torch.float64, device=self.device).expand(q.shape[0], 4, 4)
        joint_pos, joint_axis, bodies = [], [], {}
        for i, (origin, axis) in enumerate(zip(self.origins, self.axes)):
            transform = transform @ origin
            position, rotation = self._to_mount(transform)
            joint_pos.append(position)
            joint_axis.append(rotation @ axis)
            transform = transform @ axis_rotation(axis, q[:, i])
            bodies[self.chain[i]] = transform
        for name, origin in self.fixed.items():
            bodies[name] = bodies[TOOL_BODY] @ origin
        tool = bodies[TOOL_BODY]
        tcp = ((tool[:, :3, :3] @ self.tcp_offset + tool[:, :3, 3]) - self.anchor) @ self.rotation.T
        mount_bodies = {name: self._to_mount(t) for name, t in bodies.items()}
        return {'joint_pos': torch.stack(joint_pos, 1), 'joint_axis': torch.stack(joint_axis, 1),
                'bodies': mount_bodies, 'tcp': tcp, 'gripper_rot': mount_bodies[TOOL_BODY][1] @ self.gripper_in_tool}

    def tcp(self, q):
        return self.frames(q)['tcp']

    def jacobian(self, q, frames=None):
        """TCP position Jacobian [N, 3, 6] in the mount frame."""
        f = frames or self.frames(q)
        return torch.linalg.cross(f['joint_axis'], f['tcp'][:, None] - f['joint_pos'], dim=-1).transpose(1, 2)

    def pose(self, q):
        """TCP position [N, 3] and gripper rotation [N, 3, 3] (columns x, y, z) in the mount frame."""
        f = self.frames(q)
        return f['tcp'], f['gripper_rot']

    def jacobian6(self, q, frames=None):
        """[N, 6, 6]: TCP linear velocity (rows 0-2) and gripper angular velocity (rows 3-5), mount frame."""
        f = frames or self.frames(q)
        return torch.cat([self.jacobian(q, f), f['joint_axis'].transpose(1, 2)], 1)

    def outboard_bodies(self, i):
        return self.chain[i:] + list(self.fixed)

    def gravity_torque(self, q):
        """Generalized gravity force on each arm joint [N, 6], N m (the torque gravity applies)."""
        f = self.frames(q)
        torque = torch.zeros(q.shape[0], 6, dtype=torch.float64, device=self.device)
        for i in range(6):
            for name in self.outboard_bodies(i):
                position, rotation = f['bodies'][name]
                com = position + rotation @ self.com[name]
                moment = torch.linalg.cross(com - f['joint_pos'][:, i], self.mass[name] * self.gravity.expand_as(com), dim=-1)
                torque[:, i] += (moment * f['joint_axis'][:, i]).sum(-1)
        return torque

    def axis_inertia(self, q):
        """Inertia of everything outboard of each joint about that joint's axis [N, 6], kg m^2."""
        f = self.frames(q)
        inertia = torch.zeros(q.shape[0], 6, dtype=torch.float64, device=self.device)
        for i in range(6):
            axis, origin = f['joint_axis'][:, i], f['joint_pos'][:, i]
            for name in self.outboard_bodies(i):
                position, rotation = f['bodies'][name]
                com = position + rotation @ self.com[name]
                body_axis = (rotation.transpose(1, 2) @ axis[..., None])[..., 0]
                spin = (body_axis * (self.inertia[name] @ body_axis[..., None])[..., 0]).sum(-1)
                lever = torch.linalg.cross(com - origin, axis, dim=-1).square().sum(-1)
                inertia[:, i] += spin + self.mass[name] * lever
        return inertia

    def ik(self, target, q0, iterations=100, damping=0.02, max_step=0.3, tolerance=2e-4, rest=None,
           rest_weights=None, rest_gain=0.3):
        """Position-only damped least squares from q0. Returns (q, remaining error in m).

        Six joints position the TCP with three to spare. Without a secondary goal the
        spare motion drifts from target to target; on this arm the wrist (joint 5)
        then folds the gripper into the forearm. With `rest`, each step also pulls
        the joints toward that posture (weighted per joint) in the Jacobian's null
        space, which leaves the TCP where it is. The damped pseudo-inverse lets that
        pull disturb the TCP slightly, so a few plain steps finish the solve.
        """
        q = self._dls(q0.to(torch.float64).clone(), target.to(torch.float64), iterations, damping, max_step,
                      tolerance, rest, rest_weights, rest_gain)
        if rest is not None:
            q = self._dls(q, target.to(torch.float64), 20, damping, max_step, tolerance, None, None, 0.)
        return q, (target.to(torch.float64) - self.tcp(q)).norm(dim=-1)

    def _dls(self, q, target, iterations, damping, max_step, tolerance, rest, rest_weights, rest_gain):
        eye3 = torch.eye(3, dtype=torch.float64, device=self.device)
        eye6 = torch.eye(6, dtype=torch.float64, device=self.device)
        for _ in range(iterations):
            f = self.frames(q)
            error = target - f['tcp']
            jac = self.jacobian(q, f)
            pinv = jac.transpose(1, 2) @ torch.linalg.inv(jac @ jac.transpose(1, 2) + damping**2 * eye3)
            step = (pinv @ error[..., None])[..., 0]
            if rest is not None:
                pull = rest_gain * rest_weights * (rest - q)
                step = step + ((eye6 - pinv @ jac) @ pull[..., None])[..., 0]
            done = error.norm(dim=-1) < tolerance
            if rest is not None:
                done &= step.norm(dim=-1) < 1e-4
            if bool(done.all()):
                break
            scale = (max_step / step.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.)
            q = q + step * scale
        return q

    def ik_pose(self, target_pos, target_rot, q0, approach_only=False, use_seeds=True, symmetric=True,
                iterations=200, damping=0.02, max_step=0.3, rotation_scale=0.1, rest=None, rest_weights=None,
                rest_gain=0.3, position_tolerance=5e-4, rotation_tolerance=np.radians(0.5)):
        """Gripper pose IK: TCP position plus gripper rotation, or plus only its approach axis.

        target_rot [N, 3, 3] has the gripper's x, y, z (approach) axes as columns, in the mount
        frame. Starts from q0 and, with use_seeds, from IK_SEEDS_DEG; with symmetric, also tries
        the target turned 180 deg about its approach axis (the jaws are symmetric). Of the
        solutions within tolerance and with joint 5 inside WRIST_LIMIT, returns the one with the
        smallest joint travel from q0. rotation_scale weighs rotation error against position
        (0.1 m per rad: 1 deg counts like 1.7 mm). With approach_only one joint is spare, and
        `rest` pulls it as in ik().
        Returns (q [N, 6], position error [N] m, rotation error [N] rad, valid [N] bool).
        """
        f64 = dict(dtype=torch.float64, device=self.device)
        n = target_pos.shape[0]
        q0, target_pos, target_rot = (x.to(torch.float64) for x in (q0, target_pos, target_rot))
        starts = [q0]
        if use_seeds:
            starts += [torch.tensor(np.radians(seed), **f64).expand(n, 6) for seed in IK_SEEDS_DEG]
        rotations = [target_rot]
        if symmetric:
            rotations.append(target_rot @ torch.diag(torch.tensor([-1., -1., 1.], **f64)))
        combos = [(start, rot) for rot in rotations for start in starts]
        count = len(combos)
        q = torch.cat([start for start, _ in combos])
        goal_pos = target_pos.repeat(count, 1)
        goal_rot = torch.cat([rot for _, rot in combos])
        rest_all = None if rest is None else rest.to(torch.float64).repeat(count, 1)
        q = self._pose_dls(q, goal_pos, goal_rot, approach_only, iterations, damping, max_step, rotation_scale,
                           rest_all, rest_weights, rest_gain)
        if rest_all is not None and approach_only:
            # As in ik(): the posture pull leaks through the damped pseudo-inverse; finish without it.
            q = self._pose_dls(q, goal_pos, goal_rot, approach_only, 30, damping, max_step, rotation_scale,
                               None, None, 0.)
        q = torch.atan2(torch.sin(q), torch.cos(q))
        position_error, rotation_error = self.pose_error(q, goal_pos, goal_rot, approach_only)
        valid = (position_error < position_tolerance) & (rotation_error < rotation_tolerance) & (
            q[:, 4].abs() <= WRIST_LIMIT)
        # Pick per target: valid first, then the least joint travel from q0. Solutions are kept within
        # +-180 deg (the training limits, until the real cable limits are known), so travel is not
        # wrapped: going from +170 to -170 deg takes 340 deg of motion, not 20.
        q, position_error, rotation_error, valid = (x.view(count, n, *x.shape[1:]) for x in
                                                    (q, position_error, rotation_error, valid))
        travel = (q - q0).abs().max(-1).values
        score = torch.where(valid, travel, 1e3 + position_error / position_tolerance + rotation_error)
        best = score.argmin(0)
        pick = lambda x: x.gather(0, best.view(1, n, *([1] * (x.dim() - 2))).expand(1, n, *x.shape[2:]))[0]
        return pick(q), pick(position_error), pick(rotation_error), pick(valid)

    def pose_error(self, q, target_pos, target_rot, approach_only=False):
        """Position error [N] m and rotation error [N] rad (angle between approach axes if approach_only)."""
        tcp, rot = self.pose(q)
        position_error = (target_pos.to(torch.float64) - tcp).norm(dim=-1)
        if approach_only:
            cosine = (rot[:, :, 2] * target_rot[:, :, 2].to(torch.float64)).sum(-1).clamp(-1., 1.)
            return position_error, torch.acos(cosine)
        return position_error, rotation_log(target_rot.to(torch.float64) @ rot.transpose(1, 2)).norm(dim=-1)

    def _pose_dls(self, q, target_pos, target_rot, approach_only, iterations, damping, max_step, rotation_scale,
                  rest, rest_weights, rest_gain):
        eye3 = torch.eye(3, dtype=torch.float64, device=self.device)
        eye6 = torch.eye(6, dtype=torch.float64, device=self.device)
        for _ in range(iterations):
            f = self.frames(q)
            jac = self.jacobian6(q, f)
            rot = f['gripper_rot']
            if approach_only:
                # Align the approach axes; rotation about the approach axis is left free.
                approach, wanted = rot[:, :, 2], target_rot[:, :, 2]
                axis = torch.linalg.cross(approach, wanted, dim=-1)
                angle = torch.atan2(axis.norm(dim=-1), (approach * wanted).sum(-1))
                rotation_error = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12) * angle[:, None]
                jac = torch.cat([jac[:, :3], (eye3 - approach[..., None] * approach[:, None]) @ jac[:, 3:]], 1)
            else:
                rotation_error = rotation_log(target_rot @ rot.transpose(1, 2))
            error = torch.cat([target_pos - f['tcp'], rotation_scale * rotation_error], -1)
            jac = torch.cat([jac[:, :3], rotation_scale * jac[:, 3:]], 1)
            pinv = jac.transpose(1, 2) @ torch.linalg.inv(jac @ jac.transpose(1, 2) + damping**2 * eye6)
            step = (pinv @ error[..., None])[..., 0]
            if rest is not None and approach_only:
                pull = rest_gain * rest_weights * (rest - q)
                step = step + ((eye6 - pinv @ jac) @ pull[..., None])[..., 0]
            scale = (max_step / step.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.)
            q = q + step * scale
        return q


def rotation_log(rotation):
    """Rotation matrices [N, 3, 3] -> rotation vectors [N, 3] (axis times angle, angle in [0, pi])."""
    m = rotation
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    # Quaternion (w, x, y, z), choosing the largest component for accuracy (Shepperd).
    candidates = torch.stack([1 + trace, 1 + 2 * m[:, 0, 0] - trace, 1 + 2 * m[:, 1, 1] - trace,
                              1 + 2 * m[:, 2, 2] - trace], -1)
    best = candidates.argmax(-1)
    root = candidates.gather(-1, best[:, None])[:, 0].clamp_min(1e-12).sqrt()
    w = torch.stack([root, (m[:, 2, 1] - m[:, 1, 2]) / root, (m[:, 0, 2] - m[:, 2, 0]) / root,
                     (m[:, 1, 0] - m[:, 0, 1]) / root], -1)
    x = torch.stack([(m[:, 2, 1] - m[:, 1, 2]) / root, root, (m[:, 0, 1] + m[:, 1, 0]) / root,
                     (m[:, 0, 2] + m[:, 2, 0]) / root], -1)
    y = torch.stack([(m[:, 0, 2] - m[:, 2, 0]) / root, (m[:, 0, 1] + m[:, 1, 0]) / root, root,
                     (m[:, 1, 2] + m[:, 2, 1]) / root], -1)
    z = torch.stack([(m[:, 1, 0] - m[:, 0, 1]) / root, (m[:, 0, 2] + m[:, 2, 0]) / root,
                     (m[:, 1, 2] + m[:, 2, 1]) / root, root], -1)
    quat = torch.stack([w, x, y, z], 1).gather(1, best[:, None, None].expand(-1, 1, 4))[:, 0] / 2
    quat = torch.where(quat[:, :1] < 0, -quat, quat)
    vector = quat[:, 1:]
    sin_half = vector.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sin_half, quat[:, :1])
    return torch.where(sin_half > 1e-12, vector / sin_half.clamp_min(1e-12) * angle, 2 * vector)


def rotation_exp(vector):
    """Rotation vectors [N, 3] -> rotation matrices [N, 3, 3] (Rodrigues)."""
    angle = vector.norm(dim=-1, keepdim=True)[..., None]
    x, y, z = (vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)).unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).view(-1, 3, 3)
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return eye + torch.sin(angle) * k + (1 - torch.cos(angle)) * (k @ k)


def axis_rotation(axis, angle):
    """Homogeneous rotations [N, 4, 4] about a fixed unit axis (Rodrigues)."""
    x, y, z = axis
    k = torch.tensor([[0., -z, y], [z, 0., -x], [-y, x, 0.]], dtype=torch.float64, device=angle.device)
    s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
    rotation = torch.eye(3, dtype=torch.float64, device=angle.device) + s * k + (1 - c) * (k @ k)
    out = torch.eye(4, dtype=torch.float64, device=angle.device).repeat(angle.shape[0], 1, 1)
    out[:, :3, :3] = rotation
    return out
