"""Screen constant payloads throughout a cube, optimizing IK separately per load.

Uses the explicit ideal fixed-base redesign, not the existing short-rod CAD.
Intersect signed affine torque intervals instead of scaling a 2 kg result.
"""
import json
from pathlib import Path

import numpy as np

import assess_redesigns as designs
# Preserve the existing helper names while sharing the signed interval solver.
from payload_analysis import common_maximum, payload_intervals as intervals

HERE = Path(__file__).resolve().parent


def screen(height, length):
    designs.CUBE_MIN = np.array([.1, -.5, -height])
    designs.CUBE_MAX = np.array([1.1, .5, 1.-height])
    positions, rotations, labels, position_count, orientation_count = designs.goal_set(9, 64)
    result = dict(shoulder_height_above_cube_bottom_m=height,
                  cube_min=designs.CUBE_MIN.tolist(), cube_max=designs.CUBE_MAX.tolist(),
                  equal_joint_span_m=length, position_count=position_count,
                  orientations_per_position=orientation_count, tests=len(positions), options=[])
    for geared in (False, True):
        model = designs.SerialConcept(length, geared=geared)
        model.check_energy_gradient()
        reachable, q = model.ik_branches(positions, rotations)
        assert reachable.all(), 'The selected length must first cover the cube geometrically.'
        own, unit, _, tcp, *_ = model.evaluate(q.reshape(-1, 6))
        own, unit = own.reshape(4, -1, 6), unit.reshape(4, -1, 6)
        direct_p, direct_r = model.direct_fk(q.reshape(-1, 6))
        np.testing.assert_allclose(tcp, direct_p, atol=1e-12)
        designs.error_metrics(direct_p, direct_r, np.tile(positions, (4, 1)),
                              np.tile(rotations, (4, 1, 1)), np.tile(reachable, 4))
        options = [('existing_13nm', np.full(6, 13.))] if not geared else [
            ('extra_4_to_1_shoulder_2_to_1_elbow', np.array([13., 44.2, 22.1, 13., 13., 13.])),
            ('extra_8_to_1_shoulder_4_to_1_elbow', np.array([13., 88.4, 44.2, 13., 13., 13.])),
        ]
        for name, caps in options:
            lo, hi, feasible = intervals(own, unit, caps)
            maximum = common_maximum(lo, hi, feasible)
            rows = []
            for mass in [0., .25, .5, .75, 1., 1.5, 2.]:
                absolute = np.abs(own+mass*unit)
                best = (absolute/caps).max(axis=2).argmin(axis=0)
                selected = absolute[best, np.arange(len(positions))]
                passes = (selected <= caps+1e-9).all(axis=1)
                via_intervals = (feasible & (lo <= mass+1e-9) & (hi >= mass-1e-9)).any(axis=0)
                np.testing.assert_array_equal(passes, via_intervals)
                rows.append(dict(payload_kg=mass, feasible_poses=int(passes.sum()),
                                 pose_percent=float(100*passes.mean()),
                                 positions_passing_all_sampled_orientations=int(passes.reshape(position_count, orientation_count).all(axis=1).sum()),
                                 peak_selected_torques_nm=selected.max(axis=0).tolist()))
            # A concrete witness, with the best empty-arm positioning branch.
            util = (np.abs(own)/caps).max(axis=2)
            branches = util.argmin(axis=0)
            target = int(util.min(axis=0).argmax())
            branch = int(branches[target])
            witness = dict(position_m=positions[target].tolist(), orientation=labels[target],
                           orientation_matrix=rotations[target].tolist(),
                           q_deg=np.rad2deg(q[branch, target]).tolist(),
                           own_torques_nm=own[branch, target].tolist(),
                           torques_per_kg_nm=unit[branch, target].tolist(),
                           all_branch_own_torques_nm=own[:, target].tolist(),
                           all_branch_torques_per_kg_nm=unit[:, target].tolist())
            if maximum is not None:
                assert (np.abs(own+maximum*unit) <= caps+1e-7).all(axis=2).any(axis=0).all()
                assert not (np.abs(own+(maximum+1e-5)*unit) <= caps+1e-9).all(axis=2).any(axis=0).all()
            option = dict(name=name, caps_nm=caps.tolist(), total_arm_mass_kg=float(model.base_mass+model.masses.sum()),
                          common_sampled_static_payload_max_kg=maximum,
                          poses_with_no_feasible_nonnegative_payload=int((~feasible.any(axis=0)).sum()),
                          sweep=rows, worst_empty_arm_witness=witness)
            result['options'].append(option)
            print(height, length, name, 'common max:', maximum,
                  'empty feasible:', rows[0]['feasible_poses'], '/', len(positions), flush=True)
    return result


def main():
    report = dict(
        interpretation='Maximum constant payload that can be statically held at every sampled position and required orientation.',
        current_short_arm='Original 150 + 150 mm rods cannot reach the full cube regardless of payload.',
        assumptions=['Fixed base 100 mm outside the cube, centered across its width.',
                     'Ideal yaw/pitch/pitch plus three intersecting wrist axes; existing hardware masses retained.',
                     'Object COM at the middle grasp; TCP offset approximately 167 mm.',
                     'Existing joints capped at 13 N m. Extra gearing uses 0.85 assumed torque derating.',
                     'Both geared options include 750 g shoulder and 500 g elbow hardware allowances.',
                     'Four positioning IK branches optimized separately for each payload; interval unions account for counterbalancing.',
                     'Static sampled limits only; no collisions, joint stops, dynamics, thermal or strength validation.'],
        scenarios=[screen(.5, .75), screen(.05, .875)],
        validation='Potential-energy gradient, independent FK, affine intervals versus direct torque, and common-limit boundary checks passed.')
    output = HERE/'workspace_payload/cube_payload_limits.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Saved', output)


if __name__ == '__main__':
    main()
