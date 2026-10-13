"""Static rod-length design screen; never changes the simulation or CAD files.

Extends both existing carbon-fiber tubes along their axes. Rod mass scales with
length, rod COM moves half the extension, and distal connector/rotor COMs move
the full extension. This is a motor-torque screen, not a structural assessment.
"""
import copy
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import differential_evolution, minimize
import trimesh

HERE = Path(__file__).resolve().parent
from payload_analysis.paths import arm_report, arm_urdf
from estimate_payload import ArmGravity, origin


class RodVariants:
    def __init__(self):
        self.source = arm_urdf()
        self.base = ArmGravity(self.source)
        tree = ET.parse(self.source).getroot()
        report = json.loads(arm_report().read_text())
        self.parts = {}
        meshes = {}
        for record in report['visual_manifest']:
            visual = tree.find(f"link[@name='{record['body']}']/visual[@name='{record['name']}']")
            geom = visual.find('geometry/mesh')
            key = (geom.get('filename'), geom.get('scale', '1 1 1'))
            if key not in meshes:
                mesh = trimesh.load_mesh(key[0])
                mesh.apply_scale(np.fromstring(key[1], sep=' '))
                meshes[key] = mesh
            mesh = meshes[key]
            transform = origin(visual.find('origin'))
            self.parts[record['source']] = dict(
                body=record['body'], allocation=record['allocation'],
                volume=mesh.volume, center=transform[:3, :3] @ mesh.center_mass + transform[:3, 3],
                transform=transform)
        for allocation in report['allocations']:
            parts = [p for p in self.parts.values() if p['allocation'] == allocation['component']]
            volume = sum(p['volume'] for p in parts)
            for part in parts:
                part['mass'] = allocation['mass_kg'] * part['volume'] / volume
        # Reconstruct every body's original mass and first moment independently.
        for body, expected in report['bodies'].items():
            parts = [p for p in self.parts.values() if p['body'] == body]
            mass = sum(p['mass'] for p in parts)
            center = sum(p['mass'] * p['center'] for p in parts) / mass
            np.testing.assert_allclose(mass, expected['mass_kg'], atol=1e-12)
            np.testing.assert_allclose(center, expected['center_of_mass_m'], atol=1e-12)
        self.rods = []
        # These explicit groups follow the CAD component positions, not link names alone.
        for body, joint, tube, distal in [
            ('upper_arm_output', 2, '150:0', ['joint_3_motor', 'rod_90deg_1']),
            ('forearm_output', 4, '150_1:0', ['joint_5_motor', 'center_connecter']),
        ]:
            rod = self.parts[tube]
            direction = rod['transform'][:3, 2].copy()
            direction *= np.sign(direction @ self.base.origins[joint, :3, 3])
            body_parts = [p for p in self.parts.values() if p['body'] == body]
            for part in body_parts:
                if part is not rod:
                    assert ((part['center'] - rod['center']) @ direction > 0) == (part['allocation'] in distal)
            self.rods.append(dict(body=body, joint=joint, tube=tube, distal=distal, direction=direction))

    def model(self, lengths_m):
        model = copy.deepcopy(self.base)
        for spec, length in zip(self.rods, lengths_m):
            delta = length - .15
            direction = spec['direction']
            mass, moment = 0., np.zeros(3)
            for key, part in self.parts.items():
                if part['body'] != spec['body']:
                    continue
                m, c = part['mass'], part['center'].copy()
                if key == spec['tube']:
                    m *= length / .15
                    c += .5 * delta * direction
                elif part['allocation'] in spec['distal']:
                    c += delta * direction
                mass += m
                moment += m * c
            i = spec['joint'] - 1
            model.masses[i] = mass
            model.centers[i] = moment / mass
            model.origins[spec['joint'], :3, 3] += delta * direction
        return model


def torque_record(model, q):
    own, unit, _, tcp, pivots, _ = model.evaluate(q)
    total = own[0] + 2 * unit[0]
    return dict(joint_angles_deg=np.rad2deg(q).tolist(), tcp_m=tcp[0].tolist(),
                horizontal_radius_m=float(np.linalg.norm((tcp[0] - pivots[0, 0])[:2])),
                own_torques_nm=own[0].tolist(), payload_2kg_torques_nm=(2 * unit[0]).tolist(),
                total_2kg_torques_nm=total.tolist(), peak_joint_torque_nm=float(np.abs(total).max()),
                limiting_joint=int(np.argmax(np.abs(total))) + 1,
                max_static_payload_kg=float(model.capacity(q)[0]),
                empty_arm_within_13nm=bool((np.abs(own[0]) <= 13).all()))


def cube_screen(variants, report):
    from arthrobot_tasks.arm_planning.kinematics import ArmModel
    from scipy.stats import qmc
    import torch

    torch.set_num_threads(2)
    previous = json.loads((HERE / 'workspace_payload/build_volume_feasibility.json').read_text())
    corners = np.array(previous['target_corner_positions_in_mount_frame_m'])
    seeds = np.r_[np.zeros((1, 6)), (qmc.Sobol(6, scramble=True, seed=62).random_base2(6) * 2 - 1) * np.pi]
    gravity = variants.model([.65, .65])
    model = ArmModel(device='cpu')
    model.origins = [torch.tensor(t, dtype=torch.float64) for t in gravity.origins]
    q0 = torch.tensor(np.tile(seeds, (len(corners), 1)), dtype=torch.float64)
    targets = torch.tensor(np.repeat(corners, len(seeds), axis=0))
    solution, _ = model.ik(targets, q0, iterations=300, tolerance=1e-6)
    q = solution.numpy()
    own, unit, _, tcp, _, _ = gravity.evaluate(q)
    np.testing.assert_allclose(model.pose(solution)[0].numpy(), tcp, atol=1e-10)
    error = np.linalg.norm(tcp - targets.numpy(), axis=1)
    peaks = np.abs(own + 2 * unit).max(axis=1)
    cases = []
    for i, corner in enumerate(corners):
        ids = np.arange(i * len(seeds), (i + 1) * len(seeds))
        valid = ids[error[ids] < .002]
        pick = valid[np.argmin(peaks[valid])] if len(valid) else None
        cases.append(dict(target_m=corner.tolist(), solutions_found=int(len(valid)),
                          solutions_with_2kg_under_13nm=int((peaks[valid] <= 13).sum()),
                          best_found=torque_record(gravity, q[pick]) if pick is not None else None))
    report['cube_650mm_2kg_free_orientation'] = dict(
        assumptions=previous['assumption'], starts_per_corner=len(seeds),
        interpretation='Best among numerical IK solutions found; not a global torque minimum or collision check.',
        corners=cases)
    for case in cases:
        if case['best_found'] is None:
            continue
        target = np.array(case['target_m'])
        initial = np.r_[np.deg2rad(case['best_found']['joint_angles_deg']),
                        case['best_found']['peak_joint_torque_nm']]
        cache = {}
        def evaluate(x):
            key = x[:6].tobytes()
            if key not in cache:
                own, unit, _, tcp, *_ = gravity.evaluate(x[:6])
                cache[key] = own[0] + 2 * unit[0], tcp[0]
            return cache[key]
        def equality(x):
            return evaluate(x)[1] - target
        def inequalities(x):
            tau = evaluate(x)[0]
            return np.r_[x[6] - tau, x[6] + tau]
        fit = minimize(lambda x: x[6], initial, method='SLSQP',
                       bounds=[(None, None)] * 6 + [(0, None)],
                       constraints=[{'type': 'eq', 'fun': equality},
                                    {'type': 'ineq', 'fun': inequalities}],
                       options={'maxiter': 300, 'ftol': 1e-9})
        case['torque_optimized_solution'] = dict(
            optimizer_success=bool(fit.success), message=str(fit.message),
            position_error_m=float(np.linalg.norm(equality(fit.x))),
            record=torque_record(gravity, fit.x[:6]))
    report['cube_650mm_2kg_free_orientation']['refinement_note'] = (
        'Constrained SLSQP locally minimizes maximum absolute joint torque at each corner, '
        'free orientation, initialized from the best sampled IK solution. No collision check; '
        'not a global optimum proof.')
    print('cube_650mm', [(x['solutions_with_2kg_under_13nm'], round(x['best_found']['peak_joint_torque_nm'], 3)) for x in cases], flush=True)


def main():
    started = time.monotonic()
    variants = RodVariants()
    report = dict(
        urdf_sha256=hashlib.sha256(variants.source.read_bytes()).hexdigest(),
        motor_cap_nm=13., payload_kg=2., payload_com='At existing between-jaw TCP; balanced central grasp assumed.',
        rod_mass_model='8 g per 150 mm; same cross section and density; mass scales with length.',
        candidates=[],
        limitations=[
            'Static torque only: no acceleration, thermal, grip, rod stiffness, strength, or printed-part assessment.',
            'No collisions or cable limits checked for changed geometries; short rods may not fit the existing fittings.',
            'Search over all six continuous joint angles; demanding poses found are not certified global extrema.',
            'Zero-angle pose changes its TCP position when rod lengths change; it is not a workspace-wide payload rating.',
            'MG4010 geometry remains the existing placeholder for MG5010 mass distributions.',
        ])
    for length_mm in (30, 50, 75, 100, 125, 150, 200, 250, 350, 500, 650):
        model = variants.model([length_mm / 1000] * 2)
        model.check_energy_gradient()
        def objective(q):
            own, unit, *_ = model.evaluate(q.T)
            return -np.abs(own + 2 * unit).max(axis=1)
        fits = [differential_evolution(objective, [(-np.pi, np.pi)] * 6,
                                      vectorized=True, updating='deferred', popsize=18,
                                      maxiter=200, seed=seed, tol=1e-8, polish=False)
                for seed in (17, 43)]
        fit = min(fits, key=lambda f: f.fun)
        item = dict(rod_lengths_mm=[length_mm] * 2,
                    total_arm_mass_kg=float(model.base_mass + model.masses.sum()),
                    zero_angle_pose=torque_record(model, np.zeros(6)),
                    demanding_2kg_pose_found=torque_record(model, fit.x),
                    independent_search_peaks_nm=[float(-f.fun) for f in fits])
        report['candidates'].append(item)
        print(length_mm, 'zero', round(item['zero_angle_pose']['max_static_payload_kg'], 3),
              '2kg_peak', round(-fit.fun, 3), 'seeds', item['independent_search_peaks_nm'], flush=True)
    report['validation'] = 'All body masses/COMs reconstructed from CAD meshes and allocations; all candidate gravity torques match potential-energy gradients.'
    cube_screen(variants, report)
    report['elapsed_s'] = time.monotonic() - started
    path = HERE / 'workspace_payload/rod_length_payload.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('Saved', path, flush=True)


if __name__ == '__main__':
    main()
