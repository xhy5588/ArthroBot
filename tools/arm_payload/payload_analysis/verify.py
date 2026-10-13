"""Focused numerical verification; does not regenerate sampled workspaces.

Run from the repository root:
    /home/eric/metamachine_next/environments/isaaclab/.venv/bin/python projects/arthro_bot/payload_tool/payload_analysis/verify.py

Checks signed inequality edge cases, target-wise IK interval unions, independent
energy gradients, frozen original-arm results, and legacy import compatibility.
No Isaac Sim, CUDA, collision engine, or 67-million-pose search is needed.
"""
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))

from payload_analysis.paths import PROJECT_DIR, arm_urdf

from payload_analysis import (
    ArmGravity, common_maximum, gravity_torques, origin,
    payload_intervals, torque_for_payload,
)

URDF = arm_urdf()


class SignedStaticsTests(unittest.TestCase):
    def test_gravity_cross_product_sign_and_payload_superposition(self):
        # A mass on +X pulls around +Y under downward gravity.
        own, unit = gravity_torques([[0., 0., 0.]], [[0., 1., 0.]],
                                   [[.5, 0., 0.]], [2.], [1., 0., 0.])
        np.testing.assert_allclose(own, [9.81], atol=1e-14)
        np.testing.assert_allclose(unit, [9.81], atol=1e-14)
        np.testing.assert_allclose(torque_for_payload(own, unit, 2.), [29.43], atol=1e-14)
        # A load on -X counterbalances rather than reinforcing own gravity.
        _, opposite = gravity_torques([[0., 0., 0.]], [[0., 1., 0.]],
                                      [[.5, 0., 0.]], [2.], [-1., 0., 0.])
        np.testing.assert_allclose(torque_for_payload(own, opposite, 1.), [0.], atol=1e-14)

    def test_signed_scalar_counterexamples(self):
        cases = [
            # own, per kilogram, cap, expected lower, upper, feasible
            (5., 2., 13., 0., 4., True),
            (-5., -2., 13., 0., 4., True),
            (14., -2., 13., .5, 13.5, True),  # requires counterbalancing mass
            (14., 2., 13., 0., -.5, False),   # load only worsens overload
            (13., 1., 13., 0., 0., True),     # feasible zero, not impossible
            (14., 0., 13., 0., np.inf, False),
            (5., 0., 13., 0., np.inf, True),  # gravity gives no finite bound
            (5., 1e-11, 13., 0., np.inf, True),
        ]
        for own, unit, cap, lower, upper, feasible in cases:
            with self.subTest(own=own, unit=unit):
                interval = payload_intervals(own, unit, cap)
                self.assertEqual(float(interval.lower_kg), lower)
                self.assertEqual(float(interval.upper_kg), upper)
                self.assertEqual(bool(interval.feasible), feasible)

    def test_intersection_across_joints_and_batched_mass(self):
        point = payload_intervals([14., 0.], [-2., 2.], [13., 1.])
        self.assertTrue(point.feasible)
        self.assertEqual(float(point.lower_kg), .5)
        self.assertEqual(float(point.upper_kg), .5)
        impossible = payload_intervals([14., 0.], [-2., 2.], [13., .5])
        self.assertFalse(impossible.feasible)
        own = np.array([[5., 0.], [14., 0.]])
        unit = np.array([[2., 2.], [-2., 2.]])
        interval = payload_intervals(own, unit, [13., 1.])
        np.testing.assert_allclose(interval.lower_kg, [0., .5])
        np.testing.assert_allclose(interval.upper_kg, [.5, .5])
        np.testing.assert_array_equal(interval.feasible, [True, True])
        np.testing.assert_allclose(torque_for_payload(own, unit, [0., .5]),
                                   [[5., 0.], [13., 1.]])

    def test_common_mass_preserves_union_gaps_and_lower_bounds(self):
        # Target A: [0,1] or [4,5]; target B: [2,3]. No common mass.
        self.assertIsNone(common_maximum([[0., 2.], [4., 0.]],
                                         [[1., 3.], [5., 0.]],
                                         [[True, True], [True, False]]))
        self.assertEqual(common_maximum([[0., 1.], [4., 5.]],
                                        [[2., 3.], [6., 5.5]],
                                        np.ones((2, 2), bool)), 5.5)
        self.assertIsNone(common_maximum([[4., 1.]], [[6., 3.]], [[True, True]]))
        self.assertEqual(common_maximum([[0., 0.]], [[0., 0.]], [[True, True]]), 0.)
        self.assertEqual(common_maximum([[.5, 1.]], [[3., 2.]], [[True, True]]), 2.)
        self.assertIsNone(common_maximum([[0., 0.]], [[3., 3.]], [[False, True]]))
        self.assertEqual(common_maximum([[0., 1.]], [[np.inf, np.inf]], [[True, True]]),
                         np.inf)

    def test_common_maximum_matches_exhaustive_endpoint_candidates(self):
        # Any finite greatest point of closed interval unions is an upper
        # endpoint. Enumerating all endpoints independently checks the
        # descending-ceiling implementation, including disjoint unions.
        rng = np.random.default_rng(4010)
        for _ in range(40):
            lo = rng.uniform(0., 5., (4, 9))
            hi = lo+rng.uniform(0., 3., lo.shape)
            valid = rng.random(lo.shape) > .1
            candidates = sorted(set(hi[valid]), reverse=True)
            expected = next((float(m) for m in candidates
                             if (valid & (lo <= m) & (hi >= m)).any(axis=0).all()), None)
            self.assertEqual(common_maximum(lo, hi, valid), expected)


class OriginalArmRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((PROJECT_DIR/'payload_static_current.json').read_text())
        # The frozen reference was computed with the original ARM_REBOT files; ArthroBot's
        # arm must be the same model, parameter by parameter (model_fingerprint.json).
        from payload_analysis.page_data import verify_map_assets
        verify_map_assets(ArmGravity(URDF), URDF)

    def test_legacy_imports_and_six_array_evaluate_contract(self):
        from estimate_payload import ArmGravity as LegacyArmGravity, origin as legacy_origin
        self.assertIs(LegacyArmGravity, ArmGravity)
        self.assertIs(legacy_origin, origin)
        model = LegacyArmGravity(URDF)
        result = model.evaluate(np.zeros(6))
        self.assertEqual(len(result), 6)
        self.assertEqual([a.shape for a in result],
                         [(1, 6), (1, 6), (1,), (1, 3), (1, 6, 3), (1, 6, 3)])
        transform = origin(ET.fromstring('<origin xyz="1 2 3" rpy="0 0 0"/>'))
        np.testing.assert_allclose(transform[:3, 3], [1., 2., 3.])
        self.assertAlmostEqual(float(model.masses.sum()+model.base_mass), 3.825, places=12)

    def test_frozen_pose_reports_both_mounts_offsets_and_closed_jaws(self):
        for mounting, reference in self.report['mounts'].items():
            for name in ('cad_pose', 'most_demanding_found', 'same_pose_80_percent_torque',
                         'same_pose_object_10cm_further', 'same_pose_jaws_closed'):
                with self.subTest(mounting=mounting, pose=name):
                    expected = reference[name]
                    model = ArmGravity(URDF, mounting, .05 if name == 'same_pose_jaws_closed' else 0.)
                    q = np.deg2rad(expected['joint_angles_deg'])
                    actual = model.describe(q, expected['torque_fraction'], expected['payload_offset_m'])
                    for field in ('payload_kg', 'own_gravity_torque_nm', 'payload_torque_per_kg_nm',
                                  'total_gravity_torque_nm', 'payload_center_world_m',
                                  'joint_origins_world_m'):
                        np.testing.assert_allclose(actual[field], expected[field], rtol=2e-12, atol=2e-12)
                    # Same joint; ArthroBot names it joint_N where the reference says revolute_N.
                    self.assertEqual(actual['limiting_joint'].rsplit('_', 1)[1],
                                     expected['limiting_joint'].rsplit('_', 1)[1])
                    self.assertEqual(actual['empty_arm_feasible'], expected['empty_arm_feasible'])
        current = ArmGravity(URDF)
        self.assertAlmostEqual(float(current.capacity(np.zeros(6))[0]), 1.745462050412507, places=12)
        worst = self.report['mounts']['vertical']['most_demanding_found']
        self.assertAlmostEqual(float(current.capacity(np.deg2rad(worst['joint_angles_deg']))[0]),
                               .4872573066791623, places=12)

    def test_gravity_matches_independent_potential_energy_gradients(self):
        for mounting in ('vertical', 'cad_viewer'):
            ArmGravity(URDF, mounting).check_energy_gradient()

    def test_old_subclasses_and_impossible_versus_zero_capacity(self):
        from assess_redesigns import SerialConcept
        from assess_cube_payload import intervals, common_maximum as old_common
        self.assertIs(intervals, payload_intervals)
        self.assertIs(old_common, common_maximum)
        model = SerialConcept(.75)
        p = np.array([[1.1, .5, 0.]])
        approach = -p[0]/np.linalg.norm(p[0])
        y = np.array([0., 0., 1.])
        rotation = np.column_stack([np.cross(y, approach), y, approach])[None]
        reachable, branches = model.ik_branches(p, rotation)
        self.assertTrue(reachable.all())
        q = branches.reshape(-1, 6)
        own, unit, *_ = model.evaluate(q)
        np.testing.assert_allclose(np.abs(own[:, 1]), 32.89403437233334, atol=1e-11)
        np.testing.assert_allclose(np.abs(unit[:, 1]), 11.853468100096277, atol=1e-11)
        self.assertFalse(model.payload_interval(q).feasible.any())
        np.testing.assert_array_equal(model.capacity(q), np.zeros(4))


if __name__ == '__main__':
    unittest.main(verbosity=2)
