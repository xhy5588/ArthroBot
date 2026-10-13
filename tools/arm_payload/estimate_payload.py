"""Static payload screening for the standalone ARM_REBOT and its current mount.

Uses the training mount (joint 1 vertical) and existing between-jaw TCP.
Does not certify collision-free motion, PLA strength, grip or dynamic lifting.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

HERE = Path(__file__).resolve().parent
# Compatibility re-exports: existing scripts can keep importing this module.
# The reusable model and signed interval math now live in payload_analysis.
if __package__:
    from .payload_analysis.paths import arm_urdf
    from .payload_analysis import ArmGravity, origin
    from .payload_analysis.model import MOUNT_ROTATION, ROOT_POS, TCP_OFFSET
else:
    from payload_analysis.paths import arm_urdf
    from payload_analysis import ArmGravity, origin
    from payload_analysis.model import MOUNT_ROTATION, ROOT_POS, TCP_OFFSET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'payload_static_current.json')
    args = parser.parse_args()
    urdf = arm_urdf()
    report = dict(model=str(urdf), model_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
                  calculation='Gravity torque only, fixed base, all gripper masses included once',
                  payload_location='Reach-task TCP between the jaw clips (arthrobot_tasks/arm_reach/mount.py TCP_IN_TOOL)',
                  motor_torque_limit_nm=13., gravity_m_s2=9.81, mounts={},
                  limitations=['No acceleration, grip-slip, structural strength, thermal or balance rating.',
                               'Continuous-angle search includes unverified self-colliding poses; numerical minima are not certified global bounds.',
                               'Geometry-estimated mass distribution and MG4010 placeholder geometry retained for MG5010 motors.',
                               'Additional object COM offset is along the nominal tool-forward direction.'])
    for mounting in ('vertical', 'cad_viewer'):
        model = ArmGravity(urdf, mounting)
        model.check_energy_gradient()
        searches = []
        for seed in (17, 43, 91):
            fit = differential_evolution(lambda q: model.capacity(q.T), [(-np.pi, np.pi)]*6,
                                         seed=seed, vectorized=True, updating='deferred', popsize=25,
                                         maxiter=500, tol=1e-9, polish=False)
            searches.append(fit)
        worst = min(searches, key=lambda fit: fit.fun)
        item = dict(supported_arm_mass_kg=float(model.masses.sum()), fixed_base_mass_kg=model.base_mass,
                    total_arm_mass_kg=float(model.masses.sum())+model.base_mass, link_masses_kg=model.masses.tolist(),
                    grasp_in_tool_m=model.grasp.tolist(), independent_minima_kg=[float(f.fun) for f in searches],
                    cad_pose=model.describe(np.zeros(6)), most_demanding_found=model.describe(worst.x),
                    same_pose_80_percent_torque=model.describe(worst.x, fraction=.8),
                    same_pose_object_10cm_further=model.describe(worst.x, offset=.10),
                    same_pose_jaws_closed=ArmGravity(urdf, mounting, .05).describe(worst.x),
                    energy_gradient_validation='Passed: all 6 joints, 8 random poses, own mass and unit payload.')
        report['mounts'][mounting] = item
        print(mounting, json.dumps(item), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('Report:', args.output)


if __name__ == '__main__':
    main()
