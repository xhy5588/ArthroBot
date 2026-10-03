"""Offline banks of start poses for the get-up task.

Lying families (base frame: X forward, Y left, Z up):

- ``back`` / ``front`` / ``left`` / ``right``: the torso lies on that side, with a random
  yaw and +-25 deg of roll/pitch jitter;
- ``random``: a uniformly random orientation.

Joints are the standing pose plus uniform noise (arms +-0.6 rad, legs +-0.4 rad),
clipped to the self-collision limits. Each pose is placed with its lowest
collision hull 2 cm above the floor, and poses in which unjointed hulls intersect
are rejected (the convex-hull test of :mod:`.joint_limits`). Every episode starts
with 0.6 s unactuated, so the robot settles into a resting pose first.

The ``standing`` family holds upright poses resting on the wheels with the center
of mass over the axle; the get-up policy is fine-tuned on them so that it also
holds and corrects an upright robot.

Run ``scripts/humanoid/make_pose_banks.py`` to make new banks.
"""
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from arthrobot import paths
from arthrobot.urdf import forward_kinematics, read_inertial
from arthrobot_assets.humanoid import MOTOR_JOINTS, WHEEL_BODIES
from arthrobot_assets.humanoid.training_model import load_training_model, output_dir
from arthrobot_tasks.humanoid.getup.joint_limits import (JOINT_LIMITS_FILE, first_self_collision, jointed_pairs,
                                                          load_joint_limits, load_link_hulls, place_hulls)
from arthrobot_tasks.humanoid.standing.policy import CHECKPOINT as STANDING_CHECKPOINT

POSE_BANK_DIR = Path(__file__).resolve().parent / 'data/pose_banks'
STANDING_SETTINGS = STANDING_CHECKPOINT.parent / 'settings.json'
LYING_FAMILIES = ('back', 'front', 'left', 'right', 'random')
# Rotation that takes the upright torso to "this side down".
SIDE_DOWN_ROTATION = {'back': ('y', -90.), 'front': ('y', 90.), 'left': ('x', -90.), 'right': ('x', 90.)}
FLOOR_CLEARANCE_M = .02
JOINT_NOISE_RAD = np.array([.6] * 12 + [.4] * 6 + [0., 0.])
MAX_STANDING_PITCH_DEG = 20.

_worker = {}   # robot model shared by the pool workers, loaded once per process


def _load_worker_model():
    robot, _ = load_training_model()
    _worker.update(robot=robot, hulls=load_link_hulls(robot), jointed=jointed_pairs(robot),
                   inertials={link.get('name'): read_inertial(link) for link in robot.findall('link')})


def _rotated_world_hulls(joint_positions, rotation: np.ndarray):
    frames = forward_kinematics(_worker['robot'], dict(zip(MOTOR_JOINTS, joint_positions)))
    world = {}
    for name, pieces in _worker['hulls'].items():
        frame = frames[name].copy()
        frame[:3, :3] = rotation @ frame[:3, :3]
        frame[:3, 3] = rotation @ frame[:3, 3]
        world[name] = place_hulls(pieces, frame)
    return frames, world


def _floor_height_if_valid(candidate) -> float | None:
    """Torso height that puts the lowest hull FLOOR_CLEARANCE_M above the floor; None if the pose self-collides."""
    joint_positions, quaternion_xyzw = candidate
    _, world = _rotated_world_hulls(joint_positions, Rotation.from_quat(quaternion_xyzw).as_matrix())
    if first_self_collision(world, _worker['jointed']) is not None:
        return None
    lowest = min(hull.lower[2] for hulls in world.values() for hull in hulls)
    return FLOOR_CLEARANCE_M - lowest


def _upright_root_pose(candidate) -> list[float] | None:
    """Root pose [x, y, z, qw, qx, qy, qz] resting on the wheels with the COM over the axle, or None."""
    joint_positions, yaw = candidate
    frames = forward_kinematics(_worker['robot'], dict(zip(MOTOR_JOINTS, joint_positions)))
    inertials = _worker['inertials']
    mass = sum(properties.mass for properties in inertials.values())
    center_of_mass = sum(inertials[name].mass * (frame[:3, :3] @ inertials[name].center + frame[:3, 3])
                         for name, frame in frames.items()) / mass
    axle = (frames[WHEEL_BODIES[0]][:3, 3] + frames[WHEEL_BODIES[1]][:3, 3]) / 2
    pitch = np.arctan2(center_of_mass[0] - axle[0], center_of_mass[2] - axle[2])
    if abs(pitch) > np.radians(MAX_STANDING_PITCH_DEG):
        return None
    rotation = (Rotation.from_euler('z', yaw) * Rotation.from_euler('y', -pitch)).as_matrix()
    _, world = _rotated_world_hulls(joint_positions, rotation)
    if first_self_collision(world, _worker['jointed']) is not None:
        return None
    lowest = {name: min(hull.lower[2] for hull in hulls) for name, hulls in world.items()}
    wheels = min(lowest[name] for name in WHEEL_BODIES)
    if min(height for name, height in lowest.items() if name not in WHEEL_BODIES) < wheels + .03:
        return None   # an arm or leg part would touch the floor before the wheels
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    return [0., 0., float(.01 - wheels), w, x, y, z]


def _standing_pose_and_limits():
    nominal = json.loads(STANDING_SETTINGS.read_text())['nominal_pose']['joint_positions']
    lower, upper = load_joint_limits(MOTOR_JOINTS)
    return np.array([nominal.get(name, 0.) for name in MOTOR_JOINTS]), lower, upper


def make_lying_bank(seed: int, per_family: int, workers: int = 16, log=print) -> dict:
    rng = np.random.default_rng(seed)
    standing, lower, upper = _standing_pose_and_limits()
    poses = []
    with Pool(workers, initializer=_load_worker_model) as pool:
        for family in LYING_FAMILIES:
            accepted = []
            while len(accepted) < per_family:
                candidates = []
                for _ in range(2 * (per_family - len(accepted))):
                    joint_positions = np.clip(standing + rng.uniform(-1, 1, len(MOTOR_JOINTS)) * JOINT_NOISE_RAD,
                                              lower, upper)
                    yaw = Rotation.from_euler('z', rng.uniform(-np.pi, np.pi))
                    if family == 'random':
                        orientation = Rotation.random(random_state=rng)
                    else:
                        axis, angle = SIDE_DOWN_ROTATION[family]
                        jitter = Rotation.from_euler('xy', rng.uniform(-25, 25, 2), degrees=True)
                        orientation = yaw * jitter * Rotation.from_euler(axis, angle, degrees=True)
                    candidates.append((joint_positions, orientation.as_quat()))
                for (joint_positions, quaternion), height in zip(candidates, pool.map(_floor_height_if_valid, candidates)):
                    if height is not None and len(accepted) < per_family:
                        x, y, z, w = quaternion
                        accepted.append(dict(family=family, joints=joint_positions.tolist(),
                                             root_pose=[0., 0., float(height), w, x, y, z]))
            poses += accepted
            log(f'GETUP_POSES: {family} {len(accepted)}')
    urdf = output_dir() / 'robot.urdf'
    return dict(seed=seed, joint_order=list(MOTOR_JOINTS), families=list(LYING_FAMILIES),
                floor_clearance_m=FLOOR_CLEARANCE_M, urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
                limits_sha256=hashlib.sha256(JOINT_LIMITS_FILE.read_bytes()).hexdigest(),
                nominal_source=str(STANDING_SETTINGS.relative_to(paths.REPO_ROOT)),
                note='Floor-placed and hull self-collision free; dynamic recoverability is not certified.',
                poses=poses)


def make_standing_bank(seed: int, count: int, end_states: Path | None = None, workers: int = 16, log=print) -> dict:
    """Upright starts (motors on from the first step) with arms and hips away from the standing pose.

    Half come from recorded get-up end states plus small noise, half are random:
    shoulder (arm_1) anywhere within its limits, the other arm joints standing pose
    +-1 rad, hip pitch and knees +-0.3 rad, hip abduction +-0.1 rad.
    """
    rng = np.random.default_rng(seed)
    standing, lower, upper = _standing_pose_and_limits()
    recorded = json.loads(Path(end_states).read_text())['poses'] if end_states else []
    poses = []
    with Pool(workers, initializer=_load_worker_model) as pool:
        while len(poses) < count:
            candidates = []
            for index in range(2 * (count - len(poses))):
                if recorded and index % 2 == 0:
                    joint_positions = np.array(recorded[rng.integers(len(recorded))])
                    joint_positions[:12] += rng.uniform(-.1, .1, 12)
                    joint_positions[12:18] += rng.uniform(-.05, .05, 6)
                else:
                    joint_positions = standing.copy()
                    for shoulder in (0, 6):
                        joint_positions[shoulder] = rng.uniform(lower[shoulder], upper[shoulder])
                        joint_positions[shoulder + 1:shoulder + 6] += rng.uniform(-1., 1., 5)
                    joint_positions[[12, 14, 15, 17]] += rng.uniform(-.3, .3, 4)
                    joint_positions[[13, 16]] += rng.uniform(-.1, .1, 2)
                joint_positions[18:] = 0.
                candidates.append((np.clip(joint_positions, lower, upper), rng.uniform(-np.pi, np.pi)))
            for (joint_positions, _), root_pose in zip(candidates, pool.map(_upright_root_pose, candidates)):
                if root_pose is not None and len(poses) < count:
                    poses.append(dict(family='standing', joints=joint_positions.tolist(), root_pose=root_pose))
    log(f'GETUP_POSES: standing {len(poses)}')
    return dict(seed=seed, joint_order=list(MOTOR_JOINTS), families=['standing'], end_states=str(end_states),
                poses=poses,
                note='Upright on the wheels, COM over the axle; hull self-collision free. Not certified balanced.')


def merge_banks(*banks: dict) -> dict:
    """One bank holding every pose of ``banks`` (families in order of first appearance)."""
    assert all(bank['joint_order'] == banks[0]['joint_order'] for bank in banks)
    families = list(dict.fromkeys(family for bank in banks for family in bank['families']))
    return dict(joint_order=banks[0]['joint_order'], families=families,
                poses=[pose for bank in banks for pose in bank['poses']])
