"""Independently check the browser pose solver against URDF/SciPy calculations.

The tests use exact nearby reachable poses, not a claim that local inverse
kinematics can solve every target. Changed rods remain a geometry/gravity preview;
these checks do not validate collisions, dynamics, or physical hardware.

Run from the repository root:
    /home/eric/env_isaaclab/bin/python projects/arthro_bot/payload_tool/payload_viewer/check_pose_solver.py

Paths are resolved relative to this project, so another working directory also
works when the script path is absolute. Node.js executes the pure browser module without requiring Chrome or Isaac Sim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))
from payload_analysis.paths import PROJECT_DIR, arm_urdf
# Import the standalone arm modules before any humanoid/arm_test path injection.
from estimate_payload import ArmGravity
from assess_rod_payload import RodVariants


NODE_RUNNER = r"""
'use strict';
const fs = require('fs');
const solver = require(process.argv[1]);
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
for (const method of ['forward', 'solve', 'capacity', 'rotationLog', 'axisRotation', 'multiply3', 'transpose3']) {
  if (typeof solver[method] !== 'function') throw new Error('Missing solver API: ' + method);
}
const output = {forward: [], ik: [], capacity: [], rotation: [], matrix: []};
for (const fixture of data.forward) {
  const actual = solver.forward(fixture.q, data.models[fixture.model]);
  output.forward.push(actual);
  output.capacity.push(solver.capacity(actual.own, actual.unit, data.models[fixture.model].caps));
}
for (const fixture of data.ik) {
  output.ik.push(solver.solve(fixture.position, fixture.rotation, fixture.seed,
    data.models[fixture.model], fixture.options));
}
for (const fixture of data.capacity) output.capacity.push(solver.capacity(fixture.own, fixture.unit, fixture.caps));
for (const fixture of data.rotation) output.rotation.push(solver.rotationLog(fixture));
for (const fixture of data.matrix) {
  output.matrix.push({product: solver.multiply3(fixture.a, fixture.b),
    transpose: solver.transpose3(fixture.a), axisRotation: solver.axisRotation(fixture.axis, fixture.angle)});
}
process.stdout.write(JSON.stringify(output, function (key, value) {
  return typeof value === 'number' && !Number.isFinite(value) ? String(value) : value;
}));
"""


def urdf_chain(path: Path):
    """Parse geometry directly, independently of ArmGravity's geometry parser."""
    root = ET.parse(path).getroot()
    origins, axes = [], []
    for number in range(1, 7):
        joint = root.find(f"joint[@name='joint_{number}']")
        raw = joint.find('origin')
        translation = np.array([float(x) for x in raw.get('xyz', '0 0 0').split()])
        angles = np.array([float(x) for x in raw.get('rpy', '0 0 0').split()])
        transform = np.eye(4)
        transform[:3, 3] = translation
        transform[:3, :3] = Rotation.from_euler('xyz', angles).as_matrix()
        axis = np.array([float(x) for x in joint.find('axis').get('xyz').split()])
        origins.append(transform)
        axes.append(axis / np.linalg.norm(axis))
    return np.array(origins), np.array(axes)


def independent_forward(q, model, origins=None, axes=None):
    """Column-vector homogeneous FK with SciPy rotations, including world axes."""
    origins = model.origins if origins is None else origins
    local_axes = model.axes if axes is None else axes
    transform = model.mount.copy()
    pivots, world_axes, centers = [], [], []
    for index, (origin, axis, angle) in enumerate(zip(origins, local_axes, q)):
        transform = transform @ origin
        pivots.append(transform[:3, 3].copy())
        world_axes.append(transform[:3, :3] @ axis)
        motion = np.eye(4)
        motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        transform = transform @ motion
        centers.append(transform[:3, :3] @ model.centers[index] + transform[:3, 3])
    tcp = transform[:3, :3] @ model.grasp + transform[:3, 3]
    return dict(tcp=tcp, R=transform[:3, :3].copy(), pivots=np.array(pivots),
                axes=np.array(world_axes), centers=np.array(centers), transform=transform)


def pack_model(model):
    return {key: np.asarray(getattr(model, key)).tolist() for key in
            ('mount', 'origins', 'axes', 'masses', 'centers', 'grasp', 'caps')}


def signed_interval(own, unit, caps):
    own, unit, caps = np.asarray(own), np.asarray(unit), np.asarray(caps)
    changing = np.abs(unit) > 1e-10
    safe_denominator = np.where(changing, unit, 1.)
    endpoints = np.array([(-caps-own)/safe_denominator, (caps-own)/safe_denominator])
    lower_per_joint = np.where(changing, endpoints.min(axis=0), -np.inf)
    upper_per_joint = np.where(changing, endpoints.max(axis=0), np.inf)
    lower = max(0., float(lower_per_joint.max()))
    upper = float(upper_per_joint.min())
    feasible = bool((np.abs(own[~changing]) <= caps[~changing]).all() and upper >= lower)
    limiting = int(upper_per_joint.argmin()) if np.isfinite(upper) else None
    return dict(lower=lower, upper=upper, feasible=feasible, limitingJoint=limiting)


def finite_float(value):
    return float(value)  # Also decodes the runner's "Infinity" / "-Infinity".


def assert_interval(actual, expected, description):
    assert actual['feasible'] == expected['feasible'], (description, actual, expected)
    for key in ('lower', 'upper'):
        np.testing.assert_allclose(finite_float(actual[key]), expected[key], atol=1e-8, rtol=1e-10,
                                   err_msg=f'{description}: signed {key} payload bound differs')
    if expected['feasible'] and np.isfinite(expected['upper']):
        assert actual['limitingJoint'] == expected['limitingJoint'], (description, actual, expected)


def build_fixtures():
    rng = np.random.default_rng(20261012)
    variants = RodVariants()
    lengths = ((150, 150), (100, 100), (350, 650), (750, 750))
    models = [variants.model(np.array(pair)/1000) for pair in lengths]
    direct_origins, direct_axes = urdf_chain(arm_urdf())
    np.testing.assert_allclose(direct_origins, models[0].origins, atol=1e-12)
    np.testing.assert_allclose(direct_axes, models[0].axes, atol=1e-12)
    data = dict(models=[pack_model(model) for model in models], forward=[], ik=[],
                capacity=[], rotation=[], matrix=[])
    expected_forward, expected_intervals = [], []
    regular_options = dict(maxIterations=250, positionTolerance=1e-6,
                           orientationTolerance=1e-6, rotationWeight=.15,
                           damping=.003, maxStep=.3, maxTimeMs=0, restarts=0)

    for model_index, model in enumerate(models):
        # Include exact zero, isolated rotations, general poses, and wrapped angles.
        angles = np.r_[np.zeros((1, 6)), np.eye(6)*1.3, -np.eye(6)*2.1,
                       rng.uniform(-np.pi, np.pi, (20, 6)),
                       rng.uniform(-np.pi, np.pi, (3, 6)) + 4*np.pi]
        for q in angles:
            reference = independent_forward(q, model,
                direct_origins if model_index == 0 else None,
                direct_axes if model_index == 0 else None)
            own, unit, _, tcp, pivots, centers = model.evaluate(q)
            # Independent geometry must agree before it becomes a JS fixture.
            np.testing.assert_allclose(reference['tcp'], tcp[0], atol=1e-12)
            np.testing.assert_allclose(reference['pivots'], pivots[0], atol=1e-12)
            np.testing.assert_allclose(reference['centers'], centers[0], atol=1e-12)
            reference.update(own=own[0], unit=unit[0])
            expected_forward.append(reference)
            expected_intervals.append(signed_interval(own[0], unit[0], model.caps))
            data['forward'].append(dict(model=model_index, q=q.tolist()))

        # Avoid using a global-IK benchmark as a correctness test. These exact
        # targets deliberately have a nonsingular local inverse near their seed.
        regular = []
        for q in rng.uniform(-2.5, 2.5, (1000, 6)):
            fk = independent_forward(q, model)
            jacobian = np.r_[np.cross(fk['axes'], fk['tcp']-fk['pivots']).T,
                             .15*fk['axes'].T]
            if np.linalg.svd(jacobian, compute_uv=False)[-1] > .015:
                regular.append((q, fk))
            if len(regular) == 8:
                break
        assert len(regular) == 8, 'Could not find sufficient nonsingular local IK fixtures.'
        for q, fk in regular:
            data['ik'].append(dict(kind='nearby_reachable', model=model_index,
                position=fk['tcp'].tolist(), rotation=fk['R'].tolist(),
                seed=(q+rng.normal(0, .025, 6)).tolist(), known_solution=q.tolist(),
                expected_reachable=True, options=regular_options))

        # A UI rotation request changes orientation while retaining the TCP.
        # Establish feasibility independently with SciPy before testing JS.
        for axis_index, (q, fk) in enumerate(regular[:3]):
            target_r = Rotation.from_rotvec(np.eye(3)[axis_index]*np.deg2rad(6)).as_matrix() @ fk['R']
            target_p = fk['tcp'].copy()
            def residual(candidate):
                pose = independent_forward(candidate, model)
                orientation = Rotation.from_matrix(target_r @ pose['R'].T).as_rotvec()
                return np.r_[pose['tcp']-target_p, .15*orientation]
            solved = least_squares(residual, q, xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=250)
            assert np.linalg.norm(residual(solved.x)) < 1e-9, 'Synthetic fixed-TCP rotation is not established feasible.'
            assert np.max(np.abs(solved.x-q)) < .8, 'Fixed-TCP fixture left the local IK neighborhood.'
            data['ik'].append(dict(kind='rotate_at_fixed_tcp', model=model_index,
                position=target_p.tolist(), rotation=target_r.tolist(), seed=q.tolist(),
                known_solution=solved.x.tolist(), expected_reachable=True, options=regular_options))

        zero_fk = independent_forward(np.zeros(6), model)
        data['ik'].append(dict(kind='unreachable_far', model=model_index,
            position=[8., -7., 5.], rotation=zero_fk['R'].tolist(), seed=[0.]*6,
            expected_reachable=False, options=dict(regular_options, maxIterations=70)))

    # Signed intersections must handle payload counterbalancing an overloaded
    # empty arm; clipping own and payload torques to magnitudes would fail these.
    data['capacity'] = [
        dict(own=[20., 0.], unit=[-5., 2.], caps=[13., 10.]),
        dict(own=[-20.], unit=[5.], caps=[13.]),
        dict(own=[20.], unit=[0.], caps=[13.]),
        dict(own=[-20.], unit=[-5.], caps=[13.]),
        dict(own=[0.], unit=[0.], caps=[13.]),
        dict(own=[13., -13.], unit=[1., -1.], caps=[13., 13.]),
    ]
    expected_intervals.extend(signed_interval(c['own'], c['unit'], c['caps']) for c in data['capacity'])

    axes = [np.array([1., 0., 0.]), np.array([0., -1., 0.]),
            np.array([.2, -.7, .5]), np.array([-.8, .1, -.3])]
    for axis in axes:
        axis = axis/np.linalg.norm(axis)
        for angle in (0., 1e-9, .2, np.pi-1e-3, np.pi-1e-7, np.pi):
            data['rotation'].append(Rotation.from_rotvec(axis*angle).as_matrix().tolist())
    for _ in range(10):
        a, b = Rotation.random(2, random_state=rng).as_matrix()
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        data['matrix'].append(dict(a=a.tolist(), b=b.tolist(), axis=axis.tolist(), angle=float(rng.uniform(-4, 4))))
    return data, models, lengths, expected_forward, expected_intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', type=Path, default=HERE/'pose_solver.js')
    parser.add_argument('--output', type=Path, help='Optional JSON report path; otherwise only prints the report.')
    args = parser.parse_args()
    node = shutil.which('node')
    if node is None:
        raise RuntimeError('Node.js is required for pure browser solver validation.')
    solver = args.solver.resolve()
    if not solver.is_file():
        raise FileNotFoundError(f'Pose solver is not available yet: {solver}')
    fixtures, models, lengths, expected_forward, expected_intervals = build_fixtures()
    process = subprocess.run([node, '-e', NODE_RUNNER, str(solver)], input=json.dumps(fixtures),
                             text=True, capture_output=True, timeout=120, check=True)
    actual = json.loads(process.stdout)
    errors = {key: 0. for key in ('tcp', 'R', 'pivots', 'axes', 'centers', 'transform', 'own', 'unit')}
    assert len(actual['forward']) == len(expected_forward)
    for index, (result, reference) in enumerate(zip(actual['forward'], expected_forward)):
        for key in errors:
            values = np.asarray(result[key], dtype=float)
            if key == 'transform':
                values = values.reshape(4, 4)  # Public API uses flat row-major transforms.
            assert np.isfinite(values).all(), (index, key, 'nonfinite FK result')
            error = float(np.max(np.abs(values-reference[key])))
            errors[key] = max(errors[key], error)
            tolerance = 2e-9 if key in ('own', 'unit') else 2e-11
            np.testing.assert_allclose(values, reference[key], atol=tolerance, rtol=0,
                                       err_msg=f'Forward fixture {index}: {key} differs from Python')
    assert len(actual['capacity']) == len(expected_intervals)
    for index, (result, expected) in enumerate(zip(actual['capacity'], expected_intervals)):
        assert_interval(result, expected, f'Capacity fixture {index}')

    ik_errors = dict(max_position_error_m=0., max_orientation_error_rad=0.,
                     max_reported_position_error_difference_m=0., max_reported_orientation_error_difference_rad=0.)
    counts = dict(nearby_reachable=0, rotate_at_fixed_tcp=0, unreachable_far=0)
    assert len(actual['ik']) == len(fixtures['ik'])
    for index, (result, fixture) in enumerate(zip(actual['ik'], fixtures['ik'])):
        q = np.asarray(result['q'], dtype=float)
        assert q.shape == (6,) and np.isfinite(q).all(), (index, 'Invalid returned angles', result)
        pose = independent_forward(q, models[fixture['model']])
        position_error = float(np.linalg.norm(pose['tcp']-fixture['position']))
        orientation_error = float(Rotation.from_matrix(np.asarray(fixture['rotation']) @ pose['R'].T).magnitude())
        reported_position = float(result['positionError'])
        reported_orientation = float(result['orientationError'])
        assert np.isfinite([reported_position, reported_orientation]).all(), (index, 'Invalid errors', result)
        np.testing.assert_allclose(reported_position, position_error, atol=2e-9, rtol=0,
                                   err_msg=f'IK {index}: reported position error is incorrect')
        np.testing.assert_allclose(reported_orientation, orientation_error, atol=2e-8, rtol=0,
                                   err_msg=f'IK {index}: reported orientation error is incorrect')
        ik_errors['max_reported_position_error_difference_m'] = max(
            ik_errors['max_reported_position_error_difference_m'], abs(reported_position-position_error))
        ik_errors['max_reported_orientation_error_difference_rad'] = max(
            ik_errors['max_reported_orientation_error_difference_rad'], abs(reported_orientation-orientation_error))
        if fixture['expected_reachable']:
            assert result['converged'], (index, fixture['kind'], 'Nearby known-reachable target failed', result)
            assert position_error <= 1.01e-6 and orientation_error <= 1.01e-6, (index, position_error, orientation_error)
            ik_errors['max_position_error_m'] = max(ik_errors['max_position_error_m'], position_error)
            ik_errors['max_orientation_error_rad'] = max(ik_errors['max_orientation_error_rad'], orientation_error)
        else:
            assert not result['converged'] and position_error > 5., (index, 'Unreachable target reported as solved', result)
        counts[fixture['kind']] += 1

    max_log_reconstruction_error = 0.
    for index, (matrix, vector) in enumerate(zip(fixtures['rotation'], actual['rotation'])):
        vector = np.asarray(vector, dtype=float)
        assert vector.shape == (3,) and np.isfinite(vector).all(), (index, 'Invalid SO(3) logarithm')
        reconstructed = Rotation.from_rotvec(vector).as_matrix()
        error = float(np.max(np.abs(reconstructed-np.asarray(matrix))))
        max_log_reconstruction_error = max(max_log_reconstruction_error, error)
        assert error < 2e-7, (index, 'SO(3) logarithm fails near pi or zero', error)
        assert np.linalg.norm(vector) <= np.pi+1e-7, (index, 'Rotation log is not the shortest rotation')
    for fixture, result in zip(fixtures['matrix'], actual['matrix']):
        np.testing.assert_allclose(result['product'], np.asarray(fixture['a']) @ fixture['b'], atol=1e-12)
        np.testing.assert_allclose(result['transpose'], np.asarray(fixture['a']).T, atol=1e-12)
        np.testing.assert_allclose(result['axisRotation'], Rotation.from_rotvec(np.asarray(fixture['axis'])*fixture['angle']).as_matrix(), atol=1e-12)

    report = dict(passed=True, solver=str(solver), rod_lengths_mm=lengths,
                  forward_pose_checks=len(expected_forward), forward_max_absolute_errors=errors,
                  signed_capacity_checks=len(expected_intervals), inverse_pose_checks=counts,
                  inverse_pose_errors=ik_errors, rotation_log_checks=len(fixtures['rotation']),
                  maximum_rotation_log_reconstruction_error=max_log_reconstruction_error,
                  independent_reference='Direct URDF parsing, SciPy homogeneous forward kinematics, ArmGravity torque model, and independently solved fixed-TCP rotation fixtures.',
                  limitations='Local inverse-kinematics correctness checks only. No guarantee for all reachable targets, joint limits, collisions, continuous paths, dynamics, or hardware.')
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
