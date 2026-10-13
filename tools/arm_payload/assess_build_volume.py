"""Geometry-only screening of rod lengths for a 1 m cube beside a fixed arm.

Does not modify the arm model. Candidate masses, collisions, torque capability,
and paths are NOT validated by this reach-only screen.
"""
import itertools
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
from payload_analysis.paths import arm_urdf
from estimate_payload import ArmGravity, origin
from arthrobot_tasks.arm_planning.kinematics import ArmModel


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    source = arm_urdf()
    gravity = ArmGravity(source)
    root = ET.parse(source).getroot()
    directions = {}
    for body, joint in [('upper_arm_output', 2), ('forearm_output', 4)]:
        visual = next(v for v in root.find(f"link[@name='{body}']").findall('visual')
                      if v.find('geometry/mesh').get('filename').endswith('_150.stl'))
        axis = origin(visual.find('origin'))[:3, 2]
        directions[joint] = axis*np.sign(axis @ gravity.origins[joint, :3, 3])
    _, _, _, _, pivots, _ = gravity.evaluate(np.zeros(6))
    length_bound = float(np.linalg.norm(np.diff(pivots[0], axis=0), axis=1).sum()+np.linalg.norm(gravity.grasp))
    # Joint 1 is level with the middle of the cube. The cube starts 10 cm from its axis.
    joint1_z = float(pivots[0, 0, 2])
    corners = np.array(list(itertools.product([.1, 1.1], [-.5, .5], [joint1_z-.5, joint1_z+.5])))
    seeds = np.r_[np.zeros((1, 6)), (qmc.Sobol(6, scramble=True, seed=62).random_base2(5)*2-1)*np.pi]
    report = dict(assumption='A 1 m cube beside one upright fixed-base arm; joint 1 at cube mid-height; 10 cm horizontal stand-off.',
                  current_rods_mm=[150, 150], current_rod_mass_kg=[.008, .008],
                  current_joint1_to_tcp_chain_upper_bound_m=length_bound,
                  minimum_radius_containing_1m_cube_m=float(np.sqrt(3)/2),
                  farthest_corner_from_face_center_m=float(np.sqrt(1.5)),
                  farthest_corner_at_10cm_standoff_m=float(np.sqrt(1.1**2+.5**2+.5**2)),
                  current_geometry_cannot_enclose_cube=bool(length_bound < np.sqrt(3)/2),
                  target_corner_positions_in_mount_frame_m=corners.tolist(),
                  seeds_per_corner=len(seeds), candidates=[],
                  interpretation=['Passing eight corners is necessary but not sufficient for whole-volume coverage.',
                                  'IK failure means no solution found by this numerical search; not a proof of impossibility.',
                                  'Collision, module shape/COM, new mass distribution, motor torques, dynamics and assembly access are NOT checked.',
                                  'Only the two straight rods are extended; all other joint/tool offsets stay fixed.',
                                  'Candidate lengths are for design screening, not fabrication recommendations.'])
    for lengths_mm in ((150,150), (500,500), (600,600), (650,650), (700,700), (650,750), (750,650), (600,800), (800,600), (800,800)):
        model = ArmModel(device='cpu')
        for (joint, direction), length_mm in zip(directions.items(), lengths_mm):
            model.origins[joint][:3, 3] += torch.tensor(direction*(length_mm/1000-.15))
        q0 = torch.tensor(np.tile(seeds, (8, 1)), dtype=torch.float64)
        targets = torch.tensor(np.repeat(corners, len(seeds), axis=0))
        q, error = model.ik(targets, q0, iterations=220, tolerance=2e-5)
        position_errors = error.numpy().reshape(8, len(seeds)).min(axis=1)
        goals, rotations = [], []
        for corner in corners:
            radial = corner-pivots[0, 0]
            inward = -radial/np.linalg.norm(radial)
            for approach in [*np.eye(3), *(-np.eye(3)), inward]:
                reference = np.array([0., 0., 1.]) if abs(approach[2]) < .9 else np.array([0., 1., 0.])
                closing = reference-approach*(reference@approach)
                closing /= np.linalg.norm(closing)
                base = np.column_stack([np.cross(closing, approach), closing, approach])
                for roll in (np.eye(3), np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])):
                    goals.append(corner)
                    rotations.append(base@roll)
        pose_q0 = torch.tensor(np.tile(seeds, (len(goals), 1)), dtype=torch.float64)
        pose_targets = torch.tensor(np.repeat(goals, len(seeds), axis=0))
        pose_rotations = torch.tensor(np.repeat(rotations, len(seeds), axis=0))
        solution = model._pose_dls(pose_q0, pose_targets, pose_rotations, False, 280, .02, .3, .15, None, None, 0.)
        pe, re = model.pose_error(solution, pose_targets, pose_rotations, approach_only=False)
        pe, re = pe.numpy().reshape(len(goals), len(seeds)), re.numpy().reshape(len(goals), len(seeds))
        valid = (pe < .002) & (re < np.radians(1.))
        score = pe+.15*re
        picks = score.argmin(axis=1)
        item = dict(rod_lengths_mm=list(lengths_mm),
                    joint1_to_tcp_chain_upper_bound_m=float(sum(torch.linalg.norm(t[:3, 3]) for t in model.origins[1:])+np.linalg.norm(gravity.grasp)),
                    free_orientation_corners_reached=int((position_errors < .002).sum()),
                    free_orientation_best_corner_error_m=position_errors.tolist(),
                    required_orientation_cases=len(goals),
                    required_orientation_cases_reached=int(valid.any(axis=1).sum()),
                    all_14_tested_orientations_reached_per_corner=valid.any(axis=1).reshape(8,14).all(axis=1).tolist(),
                    required_orientation_best_error_m=pe[np.arange(len(goals)), picks].tolist(),
                    required_orientation_best_angle_error_deg=np.rad2deg(re[np.arange(len(goals)), picks]).tolist())
        report['candidates'].append(item)
        print(json.dumps({k:v for k,v in item.items() if 'best' not in k}), flush=True)
    report['arbitrary_orientation_wrist_distance_necessary_m'] = float(np.linalg.norm(corners[-1]-pivots[0,0])+np.linalg.norm(gravity.grasp))
    wrist_bound = length_bound-float(np.linalg.norm(gravity.grasp))
    report['equal_rod_length_optimistic_lower_bound_for_inward_orientation_m'] = .15+(report['arbitrary_orientation_wrist_distance_necessary_m']-wrist_bound)/2
    report['tested_orientations'] = 'Approach along each of +/-X, +/-Y, +/-Z and toward joint 1; rolls 0 and 90 degrees. A finite necessary screen, not all orientations.'
    report['elapsed_seconds'] = time.monotonic()-started
    output = HERE/'workspace_payload/build_volume_feasibility.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print('Saved', output, 'in', round(report['elapsed_seconds'], 1), 's', flush=True)


if __name__ == '__main__':
    main()
