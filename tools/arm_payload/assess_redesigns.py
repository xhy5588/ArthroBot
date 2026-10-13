"""Reproducible ideal-kinematic and static-torque screens of arm redesign concepts.

These are explicit concept models, not modified CAD or a validated physical robot.
Only stationary-base serial arms are considered. No collision/path,
printed-part strength, stiffness, dynamics, joint stops, or cable tests are claimed.
"""
import argparse
import itertools
import json
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from estimate_payload import ArmGravity

HERE = Path(__file__).resolve().parent
from payload_analysis.paths import arm_urdf
CUBE_MIN = np.array([.1, -.5, -.5])
CUBE_MAX = np.array([1.1, .5, .5])
TCP = float(np.linalg.norm(ArmGravity(arm_urdf()).grasp))
PAYLOAD = 2.
CAP = 13.


def rz(a):
    a = np.atleast_1d(a)
    return Rotation.from_rotvec(a[:, None]*np.array([0., 0., 1.])).as_matrix()


def ry(a):
    a = np.atleast_1d(a)
    return Rotation.from_rotvec(a[:, None]*np.array([0., 1., 0.])).as_matrix()


def euler_zyz(rotations):
    with warnings.catch_warnings():
        # Euler singularity still has an exact pose solution; paths are not claimed.
        warnings.filterwarnings('ignore', message='Gimbal lock detected')
        return Rotation.from_matrix(rotations).as_euler('ZYZ')


class SerialConcept(ArmGravity):
    def __init__(self, length, geared=False):
        self.length = length
        self.mount = np.eye(4)
        self.origins = np.tile(np.eye(4), (6, 1, 1))
        self.origins[2, 0, 3] = length
        self.origins[3, 0, 3] = length
        self.axes = np.array([[0,0,1], [0,1,0], [0,1,0], [0,0,1], [0,1,0], [0,0,1]], float)
        self.names = ['base yaw', 'shoulder pitch', 'elbow pitch', 'wrist yaw', 'wrist pitch', 'tool roll']
        self.caps = np.full(6, CAP)
        self.grasp = np.array([0., 0., TCP])
        self.tool_direction = np.array([0., 0., 1.])
        tube_mass = .008*length/.15
        self.base_mass = .54
        self.masses = np.array([.56, .56+tube_mass, .56+tube_mass, .54, .54, .509])
        self.centers = np.zeros((6, 3))
        self.centers[1:3, 0] = (.56*length+tube_mass*length/2)/(.56+tube_mass)
        self.centers[5, 2] = TCP/2
        self.geared = geared
        if geared:
            # Additional output reduction after the existing MG5010 gearbox.
            # 0.85 is a design derating assumption, not measured static efficiency.
            self.caps[1:3] = CAP*np.array([8., 4.])*.85
            self.masses[0] += .75  # shoulder hardware centered on its pitch pivot
            old_mass = self.masses[1]
            self.masses[1] += .50  # elbow hardware carried at upper-link tip
            self.centers[1] = (old_mass*self.centers[1]+np.array([.50*length,0,0]))/self.masses[1]

    def ik_branches(self, positions, rotations):
        wrist = positions-TCP*rotations[:, :, 2]
        radius = np.linalg.norm(wrist, axis=1)
        reachable = radius <= 2*self.length+1e-12
        cosine = np.clip((radius**2-2*self.length**2)/(2*self.length**2), -1, 1)
        phi = np.arctan2(wrist[:, 1], wrist[:, 0])
        radial = np.linalg.norm(wrist[:, :2], axis=1)
        candidates = []
        for reversed_yaw in [False, True]:
            yaw = phi+(np.pi if reversed_yaw else 0.)
            r = -radial if reversed_yaw else radial
            for sign in [1, -1]:
                elbow = sign*np.arccos(cosine)
                shoulder = np.arctan2(wrist[:, 2], r)-np.arctan2(
                    self.length*np.sin(elbow), self.length+self.length*np.cos(elbow))
                pre = rz(yaw) @ ry(-shoulder-elbow)
                angles = euler_zyz(np.swapaxes(pre, 1, 2) @ rotations)
                candidates.append(np.column_stack([yaw, -shoulder, -elbow, angles]))
        return reachable, np.stack(candidates)

    def ik(self, positions, rotations, payload_kg=PAYLOAD):
        reachable, candidates = self.ik_branches(positions, rotations)
        own, unit, *_ = self.evaluate(candidates.reshape(-1, 6))
        torques = (own+payload_kg*unit).reshape(4, len(positions), 6)
        choice = (np.abs(torques)/self.caps).max(axis=2).argmin(axis=0)
        ids = np.arange(len(positions))
        q = candidates[choice, ids]
        return reachable, q, torques[choice, ids]

    def direct_fk(self, q):
        q = np.atleast_2d(q)
        pre = rz(q[:, 0]) @ ry(q[:, 1]+q[:, 2])
        rotation = pre @ Rotation.from_euler('ZYZ', q[:, 3:]).as_matrix()
        x = self.length*(np.cos(q[:, 1])+np.cos(q[:, 1]+q[:, 2]))
        z = -self.length*(np.sin(q[:, 1])+np.sin(q[:, 1]+q[:, 2]))
        wrist = np.column_stack([x*np.cos(q[:, 0]), x*np.sin(q[:, 0]), z])
        return wrist+TCP*rotation[:, :, 2], rotation


def orientation_set():
    # Six cardinal approach directions with four gripper rolls each.
    rotations, labels = [], []
    for direction, axis in zip(['+X','+Y','+Z','−X','−Y','−Z'], np.r_[np.eye(3), -np.eye(3)]):
        reference = np.array([0.,0.,1.]) if abs(axis[2]) < .9 else np.array([0.,1.,0.])
        y = reference-axis*(reference@axis)
        y /= np.linalg.norm(y)
        frame = np.column_stack([np.cross(y, axis), y, axis])
        for roll in [0, 90, 180, 270]:
            rotations.append(frame @ rz(np.radians(roll))[0])
            labels.append(f'Approach {direction} · roll {roll}°')
    return np.array(rotations), labels


def goal_set(grid_size, random_orientations=0):
    positions = np.array(list(itertools.product(*[np.linspace(a,b,grid_size) for a,b in zip(CUBE_MIN,CUBE_MAX)])))
    rotations, labels = orientation_set()
    if random_orientations:
        rotations = np.r_[rotations, Rotation.random(random_orientations, random_state=62026).as_matrix()]
        labels += [f'Random orientation {i+1}' for i in range(random_orientations)]
    goals, frames, names = [], [], []
    for position in positions:
        local = list(rotations)
        axis = -position/np.linalg.norm(position)
        reference = np.array([0.,0.,1.]) if abs(axis[2]) < .9 else np.array([0.,1.,0.])
        y = reference-axis*(reference@axis); y /= np.linalg.norm(y)
        inward = np.column_stack([np.cross(y, axis), y, axis])
        local += [inward, inward @ rz(np.pi/2)[0]]
        local_labels = labels+['Approach toward base · roll 0°', 'Approach toward base · roll 90°']
        frames.extend(local); goals.extend([position]*len(local)); names.extend(local_labels)
    return np.array(goals), np.array(frames), names, len(positions), len(labels)+2


def error_metrics(actual_p, actual_r, targets, target_r, valid):
    pe = np.linalg.norm(actual_p-targets, axis=1)
    re = Rotation.from_matrix(np.swapaxes(actual_r, 1, 2) @ target_r).magnitude()
    if valid.any():
        assert pe[valid].max() < 1e-9
        assert re[valid].max() < 1e-9
    return pe, re


def records_serial(model, goals):
    positions, rotations, labels, *_ = goals
    reachable, q, torques = model.ik(positions, rotations)
    _, _, _, tcp, pivots, _ = model.evaluate(q)
    actual_p, actual_r = model.direct_fk(q)
    np.testing.assert_allclose(tcp, actual_p, atol=1e-12)
    pe, re = error_metrics(tcp, actual_r, positions, rotations, reachable)
    records = []
    for i in range(len(positions)):
        ok = bool(reachable[i])
        # For impossible targets, show a reachable zero-angle reference rather than
        # connecting the arm to a target that its kinematics cannot reach.
        reference = model.evaluate(np.zeros(6)) if not ok else None
        chain = np.r_[pivots[i], tcp[i:i+1]] if ok else np.r_[reference[4][0], reference[3]]
        records.append(dict(position=positions[i].tolist(), orientation_label=labels[i],
                            orientation_matrix=rotations[i].tolist(), reachable=ok,
                            torque_pass=bool(ok and (np.abs(torques[i])<=model.caps).all()),
                            q=q[i].tolist() if ok else [0.]*6, points=chain.tolist(),
                            torques_nm=torques[i].tolist() if ok else [],
                            position_error_m=float(pe[i]) if ok else None,
                            orientation_error_rad=float(re[i]) if ok else None,
                            wrist_target_m=(positions[i]-TCP*rotations[i,:,2]).tolist(),
                            unavailable_reason=None if ok else 'Wrist target exceeds the two-link span; arm shown at its zero-angle reference.'))
    return records


def screen_serial(length, dense, display, geared=False):
    model = SerialConcept(length, geared=geared)
    model.check_energy_gradient()
    positions, rotations, _, _, _ = dense
    reachable, q, torques = model.ik(positions, rotations)
    tcp, actual_r = model.direct_fk(q)
    pe, re = error_metrics(tcp, actual_r, positions, rotations, reachable)
    peaks = np.abs(torques[reachable]).max(axis=0)
    passes = reachable & (np.abs(torques)<=model.caps).all(axis=1)
    required_wrist_radius = np.linalg.norm(CUBE_MAX)+TCP
    long = length > .5
    mass_notes = [
        'Concept-only mass model: six 460 g motors, 460 g gripper, 589 g total brackets/fittings, tubes at 8 g per 150 mm.',
        'Yaw motor plus 80 g hardware are fixed; shoulder/elbow motor centers lie on the corresponding pivot; three wrist axes intersect ideally.',
        'Motor housings and bearings still need a physically realizable wrist design. Mass allocation is a stated concept assumption, not new CAD.',
        'Torque maxima describe the best of four positioning IK branches for the tested poses. They are not certified global maxima or a complete motor sizing specification.',
    ]
    result = dict(
        id='serial_long' if long else 'serial_short',
        name='750 + 750 mm spherical-wrist arm' if long else '350 + 350 mm spherical-wrist arm',
        subtitle='Fixed arm base beside the cube',
        description='Base yaw, two parallel pitch joints for position, then three intersecting wrist axes for orientation. Lengths are joint-center spans, not carbon-tube cut lengths.',
        changes=['Replace the current distributed roll/bend arrangement with shoulder pitch + elbow pitch.',
                 'Place the three wrist axes at one center; keep the existing 167 mm center-to-grasp offset.',
                 'Retain the 13 N·m motor limits for this test. No stronger motor is silently assumed.'],
        dimensions=[dict(label='Upper pivot span', value=f'{length*1000:.0f} mm'),
                    dict(label='Forearm pivot span', value=f'{length*1000:.0f} mm'),
                    dict(label='Wrist to grasp', value=f'{TCP*1000:.1f} mm'),
                    dict(label='Concept arm mass', value=f'{model.masses.sum()+model.base_mass:.3f} kg'),
                    dict(label='Largest sampled shoulder / elbow torque', value=f'{peaks[1]:.1f} / {peaks[2]:.1f} N·m')],
        verdict='Ideal pose geometry passes; current motors fail' if long else 'Too short for the full cube',
        geometry=dict(passed=int(reachable.sum()), total=len(positions), analytic_pass=bool(2*length>=required_wrist_radius),
                      analytic_claim=f'All wrist targets are inside radius {required_wrist_radius:.6f} m; equal links reach radii 0 to {2*length:.3f} m with unrestricted ideal joints.',
                      required_wrist_radius_m=float(required_wrist_radius), available_radius_m=2*length,
                      maximum_position_error_m=float(pe[reachable].max()), maximum_orientation_error_rad=float(re[reachable].max())),
        statics=dict(passed=int(passes.sum()), total=len(positions), reachable_total=int(reachable.sum()),
                     caps_nm=model.caps.tolist(), peak_torques_nm=peaks.tolist(), joint_labels=model.names,
                     passes_entire_suite=bool(passes.all()), mass_model=mass_notes,
                     concept_masses_kg=model.masses.tolist(), concept_centers_m=model.centers.tolist(),
                     note='Gravity only, including the arm and 2 kg load. Four positioning branches tested per target; axes are ideal, with no joint stops.'),
        limitations=['Collisions, finite motor packaging, cable limits, singular motion and assembly paths are not checked.',
                     'A spherical wrist changes orientation control; it cannot remove the distance from the fixed base to the far cube corner.',
                     '2 kg hardware lifting is not validated. New actuator or counterbalance sizing requires real component masses and motion speeds.'],
        cases=records_serial(model, display),
        validation='Independent direct FK versus homogeneous-transform FK, and gravity versus negative potential-energy gradient, passed.')
    if geared:
        result['id'] = 'serial_geared'
        result['name'] = '750 + 750 mm arm with extra reduction'
        result['subtitle'] = 'Stationary base · proposed shoulder / elbow transmission'
        result['verdict'] = ('Ideal reach + conditional static budget pass' if passes.all()
                             else 'Extra reduction still fails some static targets')
        result['changes'][-1] = 'Add 8:1 output reduction at the shoulder and 4:1 at the elbow, after the existing MG5010 gearbox; wrist motors retain the 13 N·m cap.'
        result['changes'].append('Use separate output bearings and a housing sized for the amplified joint torque. Existing PLA holders are not rated for this concept.')
        result['dimensions'] += [dict(label='Additional shoulder / elbow reduction', value='8:1 / 4:1'),
                                 dict(label='Assumed output torque budgets', value='88.4 / 44.2 N·m'),
                                 dict(label='Rated-speed references after extra reduction', value='6.75 / 13.5 rpm')]
        result['statics']['transmission'] = dict(extra_ratios=[8.,4.], torque_derating=.85,
                                               shoulder_mass_budget_kg=.75, elbow_mass_budget_kg=.50,
                                               rated_speed_reference_rpm=[6.75,13.5])
        result['statics']['mass_model'].append('Additional transmission mass budgets: 750 g at the shoulder pivot and 500 g at the elbow pivot. Real reducers, bearings and housings must fit these budgets or be recalculated.')
        result['statics']['mass_model'].append('Output torque budget = 13 N·m × extra ratio × 0.85. The 0.85 factor is an assumed derating, not measured holding efficiency. No existing 36:1 reduction is multiplied a second time.')
        result['statics']['note'] += ' Shoulder / elbow budgets are 88.4 / 44.2 N·m after assumed additional reduction; this is a proposed transmission, not installed hardware. Other joints remain at 13 N·m.'
        result['limitations'].insert(0,'Passing results are conditional on the new transmission torque and mass budgets. Reducer fit, stiffness, backlash, reflected inertia, output bearings and thermal capability remain unvalidated.')
    return result



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'workspace_payload/redesign_validation.json')
    args = parser.parse_args()
    start = time.monotonic()
    dense = goal_set(9, 64)
    display = goal_set(3)
    report = dict(
        assumptions=dict(cube_min=CUBE_MIN.tolist(), cube_max=CUBE_MAX.tolist(), cube_size_m=1., payload_kg=PAYLOAD,
                         tcp_offset_m=TCP, module_length_m=.5, module_com='Balanced at the central grasp; point payload for kinematics/torque.',
                         base_location='Stationary shoulder 10 cm beside the cube near face, level with its center.',
                         position_count=dense[3], orientation_count=dense[4], tests_per_concept=len(dense[0]),
                         display_position_count=display[3], display_orientation_count=display[4],
                         orientation_sampling='24 cardinal direction/roll rotations, 64 seeded random rotations, and 2 inward-facing rotations at each position.',
                         motor_cap_nm=CAP, continuous_ideal_joint_angles=True, fixed_base_required=True,
                         shoulder_height_confirmed=False,
                         note='Concept validation only. No CAD, Isaac Sim dynamics, geometry collision, structural or physical lifting tests.'),
        bounds=dict(farthest_cube_corner_m=float(np.linalg.norm(CUBE_MAX)),
                    wrist_radius_needed_for_all_orientations_m=float(np.linalg.norm(CUBE_MAX)+TCP),
                    minimum_equal_pivot_span_m=float((np.linalg.norm(CUBE_MAX)+TCP)/2)),
        concepts=[], sources=[
            dict(title='MG5010E-i36-V3 supplier specifications: 13 N·m rated; 460 g', url='https://aifitlab.com/products/lkmtech-mg5010e-i36-v3-motor'),
            dict(title='Fixed-base mounting requirement', url='', note='User confirmed: one stationary base beside the work volume; no rack, rail or gantry.')])
    for length, geared in [(.35, False), (.75, False), (.75, True)]:
        result = screen_serial(length, dense, display, geared=geared)
        report['concepts'].append(result)
        print(result['id'], result['geometry']['passed'], '/', result['geometry']['total'],
              'static pass', result['statics']['passed'], 'peak Nm', result['statics']['peak_torques_nm'], flush=True)
    report['mounting_scenarios'] = [
        dict(name='Shoulder at cube middle height (current comparison)',
             shoulder_height_above_cube_bottom_m=.5,
             farthest_tcp_m=float(np.sqrt(1.1**2+.5**2+.5**2)),
             required_wrist_reach_m=float(np.sqrt(1.1**2+.5**2+.5**2)+TCP)),
        dict(name='Shoulder 50 mm above cube bottom (not this comparison)',
             shoulder_height_above_cube_bottom_m=.05,
             farthest_tcp_m=float(np.sqrt(1.1**2+.5**2+.95**2)),
             required_wrist_reach_m=float(np.sqrt(1.1**2+.5**2+.95**2)+TCP)),
    ]
    report['elapsed_seconds'] = time.monotonic()-start
    report['calculation_checks_passed'] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
