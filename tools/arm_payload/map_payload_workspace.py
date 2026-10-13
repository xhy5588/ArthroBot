"""Sample ARM_REBOT's static payload workspace using its existing self-collision model.

Needs cuRobo, PyTorch and CUDA. Self-collisions use ArthroBot's cuRobo sphere model
(scripts/arm/build_curobo_robot.py); the included map was sampled with the original
ARM_REBOT sphere model. This is a best-found-pose map, not a motion plan
or a certified mechanical rating. Coordinates are relative to the upright mount.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc
import torch

from estimate_payload import ArmGravity
from payload_analysis.page_data import write_map_csv
from payload_analysis.statics import payload_intervals as intervals
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState

HERE = Path(__file__).resolve().parent
from payload_analysis.paths import REPO_DIR, arm_urdf
LEVELS = (0., .5, 1., 1.5, 2.)
MODES = ('any_orientation', 'downward_15deg')


class VoxelMap:
    def __init__(self, width):
        self.width = width
        self.n = int(round(2./width))
        assert abs(self.n*width-2.) < 1e-8
        self.capacity = np.full(self.n**3, -np.inf)
        self.q = np.zeros((self.n**3, 6), np.float64)
        self.position = np.zeros((self.n**3, 3), np.float32)
        self.count = np.zeros(self.n**3, np.int64)
        self.extremes = {str(m): {'radius_m': -1.} for m in LEVELS}

    def add(self, q, position, capacity):
        if not len(q):
            return
        cells = np.floor((position+1.)/self.width).astype(np.int32)
        assert (cells >= 0).all() and (cells < self.n).all()
        ids = np.ravel_multi_index(cells.T, (self.n,)*3)
        self.count += np.bincount(ids, minlength=len(self.count))
        # The last entry of each sorted cell has the largest payload capacity.
        order = np.lexsort((capacity, ids))
        last = np.r_[ids[order][1:] != ids[order][:-1], True]
        pick = order[last]
        pick = pick[capacity[pick] > self.capacity[ids[pick]]]
        target = ids[pick]
        self.capacity[target], self.q[target], self.position[target] = capacity[pick], q[pick], position[pick]
        radius = np.linalg.norm(position[:, :2], axis=1)
        for mass in LEVELS:
            available = np.flatnonzero(capacity >= mass)
            if len(available):
                i = available[np.argmax(radius[available])]
                if radius[i] > self.extremes[str(mass)]['radius_m']:
                    self.extremes[str(mass)] = dict(radius_m=float(radius[i]), position_m=position[i].tolist(),
                                                  angles_deg=np.rad2deg(q[i]).tolist(), capacity_kg=float(capacity[i]))

    def summary(self):
        return dict(occupied_cells=int((self.count > 0).sum()),
                    payload_cells={str(m): int((self.capacity >= m).sum()) for m in LEVELS},
                    farthest_sampled_horizontal_reach=self.extremes)

    def save(self, path):
        ids = np.flatnonzero(self.count)
        center = (np.stack(np.unravel_index(ids, (self.n,)*3), 1)+.5)*self.width-1.
        data = dict(voxel_center_m=center, sampled_tcp_m=self.position[ids], max_static_payload_kg=self.capacity[ids],
                    joint_angles_rad=self.q[ids], sample_count=self.count[ids])
        np.savez_compressed(path.with_suffix('.npz'), **data)
        write_map_csv(path.with_suffix('.csv'), data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--power', type=int, default=24, help='Number of Sobol joint poses is 2**power.')
    parser.add_argument('--batch', type=int, default=65536)
    parser.add_argument('--voxel', type=float, default=.025)
    parser.add_argument('--output', type=Path, default=HERE/'workspace_payload')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from arthrobot_tasks.arm_planning.curobo_robot import build_config, config_path, planning_urdf_path
    urdf = arm_urdf()
    config = config_path() if config_path().exists() else build_config()
    gravity = ArmGravity(urdf)
    gravity.check_energy_gradient()
    checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=str(config), scene_model=None, collision_activation_distance=0.))
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(str(config)))
    assert list(kin.joint_names) == gravity.names

    def collides(q):
        _, distance = checker.get_scene_self_collision_distance_from_joints(q[:, None].contiguous())
        return (distance.view(len(q), -1).amax(-1) > 0).cpu().numpy()

    known = np.radians([[0]*6, [-34, 12, -91, 71, -64, -37], [-44, 126, -91, -95, -12, -144],
                        [-39, 123, -91, -73, -31, 16], [-11, 4, 59, -44, -147, 0]])
    # Known free / colliding poses, checked with the original ARM_REBOT sphere model.
    flags = collides(torch.tensor(known, dtype=torch.float32, device='cuda'))
    assert flags.tolist() == [False, True, True, True, True], flags
    maps = {mode: VoxelMap(args.voxel) for mode in MODES}
    report = dict(samples=2**args.power, seed=20261007, voxel_size_m=args.voxel,
                  joint_limits_deg=[-180, 180], torque_limits_nm=gravity.caps.tolist(),
                  moving_mass_kg=float(gravity.masses.sum()), fixed_base_mass_kg=gravity.base_mass,
                  coordinates='Upright mount frame: origin at base bottom, +Z up; horizontal radius from base axis.',
                  payload='Point mass at the existing between-jaw TCP; gripper geometry held open.',
                  method='Scrambled Sobol joint sampling; best static payload among sampled poses per Cartesian cell.',
                  # Robot files relative to the repository (page_data checks the model numerically,
                  # against model_fingerprint.json), this tool's files relative to the tool folder.
                  source_sha256={**{str(p.relative_to(REPO_DIR)) if p.is_relative_to(REPO_DIR) else str(p):
                                    hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [urdf, config, planning_urdf_path(),
                                              REPO_DIR/'source/arthrobot_tasks/arm_reach/mount.py']},
                                 **{str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [HERE/'estimate_payload.py', HERE/'payload_analysis/model.py',
                                              HERE/'payload_analysis/statics.py', Path(__file__).resolve()]}},
                  limitations=['No external obstacles, floor or table; negative Z requires an elevated mount.',
                               'Approximate sphere self-collision model with a 5 mm sphere buffer; not exact mesh collision.',
                               'Best found at a sampled point inside each cell, not a guaranteed optimum at the cell center.',
                               'A pose existing does not prove a continuous collision-free path from the current arm pose.',
                               'Static gravity only: no acceleration, structural strength, grip-slip, thermal or cable-limit certification.',
                               'MG5010 torque/mass settings with existing MG4010 placeholder CAD geometry.',
                               'Object geometry is unspecified and is not included in collision checks.'],
                  validation={'energy_gradient': 'passed', 'known_collision_cases': flags.tolist()},
                  progress=[])
    sobol = qmc.Sobol(d=6, scramble=True, seed=report['seed'])
    started = time.monotonic()
    rejected, free_count, down_count = 0, 0, 0
    fk_error = 0.
    best_examples = []
    for start in range(0, report['samples'], args.batch):
        q = (sobol.random(min(args.batch, report['samples']-start))*2.-1.)*np.pi
        tq = torch.tensor(q, dtype=torch.float32, device='cuda')
        bad = collides(tq)
        rejected += int(bad.sum())
        state = kin.compute_kinematics(JointState.from_position(tq, joint_names=kin.joint_names))
        pose = state.tool_poses.get_link_pose('tcp')
        kp = pose.position.view(-1, 3).cpu().numpy()
        quat = pose.quaternion.view(-1, 4).cpu().numpy()
        approach_z = 1.-2.*(quat[:, 1]**2+quat[:, 2]**2)
        down = approach_z <= -math.cos(math.radians(15.))
        own, unit, _, position, *_ = gravity.evaluate(q)
        fk_error = max(fk_error, float(np.linalg.norm(position-kp, axis=1).max()))
        lower, upper, valid = intervals(own, unit, gravity.caps)
        # Layer membership must mean every load from 0 through the reported maximum is feasible.
        valid &= (lower == 0.) & np.isfinite(upper)
        free = ~bad & valid
        free_count += int(free.sum())
        down_count += int((free & down).sum())
        maps['any_orientation'].add(q[free], position[free], upper[free])
        maps['downward_15deg'].add(q[free & down], position[free & down], upper[free & down])
        done = start+len(q)
        if done % max(args.batch, report['samples']//8) == 0 or done == report['samples']:
            event = dict(samples=done, seconds=round(time.monotonic()-started, 2),
                         cells={mode: item.summary()['payload_cells'] for mode, item in maps.items()})
            report['progress'].append(event)
            print(json.dumps(event), flush=True)
    assert fk_error < 2e-6, fk_error
    report['self_collision_rejected'] = rejected
    report['valid_samples'] = dict(any_orientation=free_count, downward_15deg=down_count)
    report['validation']['max_fk_difference_m'] = fk_error
    report['maps'] = {mode: item.summary() for mode, item in maps.items()}
    # Recompute every retained witness using the stored joint angles (not just the original samples).
    for mode, item in maps.items():
        ids = np.flatnonzero(item.count)
        max_ratio, max_position_error, collision_count = 0., 0., 0
        for begin in range(0, len(ids), args.batch):
            batch_ids = ids[begin:begin+args.batch]
            q = item.q[batch_ids].astype(np.float64)
            own, unit, _, p, *_ = gravity.evaluate(q)
            total = own+item.capacity[batch_ids, None]*unit
            max_ratio = max(max_ratio, float((np.abs(total)/gravity.caps).max()))
            max_position_error = max(max_position_error, float(np.linalg.norm(p-item.position[batch_ids], axis=1).max()))
            collision_count += int(collides(torch.tensor(q, dtype=torch.float32, device='cuda')).sum())
        print('Witness validation', mode, max_ratio, max_position_error, collision_count, flush=True)
        assert max_ratio < 1.00000001 and max_position_error < 2e-6 and collision_count == 0, (mode, max_ratio, max_position_error, collision_count)
        report['validation'][mode] = dict(retained_poses=len(ids), max_torque_limit_ratio=max_ratio,
                                         max_stored_position_error_m=max_position_error,
                                         collision_flags=collision_count)
        item.save(args.output/mode)
    report['elapsed_seconds'] = time.monotonic()-started
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Finished:', args.output, 'in', round(report['elapsed_seconds'], 1), 'seconds', flush=True)


if __name__ == '__main__':
    main()
