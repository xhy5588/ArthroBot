"""Gripper pose targets for planning tests, in the mount frame. Torch only; no Isaac imports."""
from __future__ import annotations

import math

import numpy as np
import torch

from arthrobot_tasks.arm_planning.kinematics import ArmModel
from arthrobot_tasks.arm_planning.obstacles import HOME_TCP, TARGET_ABS_Y, TARGET_X, TARGET_Z

# Grasp poses for the collision-model check: table height up to 20 cm, at least 15 cm from the base
# axis (closer, the jaws hit the base or shoulder), pointing down or tilted up to 45 deg.
GRASP_BOX = ((0.10, 0.40), (-0.25, 0.25), (0.0, 0.20))
GRASP_MAX_TILT_DEG = 45.
GRASP_MIN_RADIUS = 0.15
FLIP = torch.diag(torch.tensor([-1., -1., 1.], dtype=torch.float64))  # Same grasp, turned 180 deg about z.


def top_down(yaw: torch.Tensor) -> torch.Tensor:
    """Gripper axes [N, 3, 3] pointing straight down with the jaw axis at yaw (rad) about z."""
    yaw = torch.as_tensor(yaw, dtype=torch.float64)
    closing = torch.stack([yaw.cos(), yaw.sin(), torch.zeros_like(yaw)], -1)
    approach = torch.tensor([0., 0., -1.], dtype=torch.float64).expand_as(closing)
    return torch.stack([torch.linalg.cross(closing, approach, dim=-1), closing, approach], -1)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """[N, 3, 3] -> [N, 4] (w, x, y, z)."""
    from scipy.spatial.transform import Rotation
    xyzw = Rotation.from_matrix(matrix.cpu().numpy()).as_quat()
    return torch.tensor(xyzw[:, [3, 0, 1, 2]], dtype=torch.float32)


def pick_targets(model: ArmModel, sequences: int, targets: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reachable top-down grasps [S, K, 3] and [S, K, 3, 3], alternating sides of the wall (obstacles.py)."""
    rng = np.random.default_rng(seed)
    positions, rotations = [], []
    while len(positions) < sequences * targets:
        k = len(positions) % targets
        side = 1. if (k + len(positions) // targets) % 2 == 0 else -1.
        p = torch.tensor([[rng.uniform(*TARGET_X), side * rng.uniform(*TARGET_ABS_Y), rng.uniform(*TARGET_Z)]],
                         dtype=torch.float64)
        r = top_down(torch.tensor([rng.uniform(0., math.pi)]))
        _, _, _, valid = model.ik_pose(p, r, torch.zeros(1, 6, dtype=torch.float64))
        if bool(valid[0]):
            positions.append(p[0])
            rotations.append(r[0])
    return torch.stack(positions).view(sequences, targets, 3), torch.stack(rotations).view(sequences, targets, 3, 3)


def home_pose(model: ArmModel) -> torch.Tensor:
    """Pointing down above the wall: the highest reachable point over HOME_TCP's x, y (pointing-down reach
    fades above about 0.2 m), so moves between the two sides can pass over the wall."""
    x, y, top = HOME_TCP
    for z in np.arange(top, 0.17, -0.01):
        for yaw in (0., math.pi / 2):
            q, _, _, valid = model.ik_pose(torch.tensor([[x, y, z]], dtype=torch.float64),
                                           top_down(torch.tensor([yaw])), torch.zeros(1, 6, dtype=torch.float64))
            if bool(valid[0]):
                return q[0]
    raise RuntimeError('no reachable home pose above the wall')


def sample_grasps(model: ArmModel, sequences: int, targets: int, seed: int, device, mode: str = 'tilted'):
    """Grasp poses [S, K, 3] and [S, K, 3, 3] in GRASP_BOX that the rigid-model IK reaches from its start poses
    with the wrist inside its limit ('down': pointing straight down; 'tilted': up to GRASP_MAX_TILT_DEG off),
    with a random rotation about the approach axis. Also returns the fraction of candidates kept."""
    generator = torch.Generator().manual_seed(seed)
    n = 8 * sequences * targets
    uniform = lambda low, high: torch.rand(n, generator=generator, dtype=torch.float64) * (high - low) + low
    position = torch.stack([uniform(*bounds) for bounds in GRASP_BOX], -1)
    tilt = uniform(0., math.radians(GRASP_MAX_TILT_DEG)) if mode == 'tilted' else torch.zeros(n, dtype=torch.float64)
    azimuth, yaw = uniform(-math.pi, math.pi), uniform(0., math.pi)
    approach = torch.stack([tilt.sin() * azimuth.cos(), tilt.sin() * azimuth.sin(), -tilt.cos()], -1)
    closing = torch.stack([yaw.cos(), yaw.sin(), torch.zeros_like(yaw)], -1)
    closing = closing - (closing * approach).sum(-1, keepdim=True) * approach
    closing = closing / closing.norm(dim=-1, keepdim=True)
    rotation = torch.stack([torch.linalg.cross(closing, approach, dim=-1), closing, approach], -1)
    position, rotation = position.to(device), rotation.to(device)
    _, _, _, valid = model.ik_pose(position, rotation, torch.zeros(n, 6, dtype=torch.float64, device=device))
    valid &= position[:, :2].norm(dim=-1) >= GRASP_MIN_RADIUS
    keep = torch.nonzero(valid)[:, 0]
    assert len(keep) >= sequences * targets, f'only {len(keep)} reachable grasp poses'
    keep = keep[:sequences * targets]
    return (position[keep].float().view(sequences, targets, 3), rotation[keep].view(sequences, targets, 3, 3),
            float(valid.float().mean()))
