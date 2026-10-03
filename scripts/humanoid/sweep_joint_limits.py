"""Measure the get-up task's self-collision joint limits by convex-hull sweeps (no simulator needed).

Each limb joint is swept alone from the standing pose of the included standing
policy until two unjointed collision hulls come within --clearance-mm; the limit
is the last clear angle minus --margin-deg. See arthrobot_tasks.humanoid.getup.joint_limits.

    python scripts/humanoid/sweep_joint_limits.py --output build/humanoid_training/joint_limits.json
    python scripts/humanoid/sweep_joint_limits.py --joints left_hip_1 right_hip_1 --output /tmp/hips.json

To train with new limits, replace source/arthrobot_tasks/humanoid/getup/data/joint_limits.json
(the get-up USD is rebuilt automatically) and re-make the pose banks.
"""
import argparse
import json
from pathlib import Path

from arthrobot import paths
from arthrobot_assets.humanoid.training_model import load_training_model
from arthrobot_tasks.humanoid.getup.joint_limits import measure_joint_limits
from arthrobot_tasks.humanoid.standing.policy import load_standing_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--step-deg', type=float, default=2.)
    parser.add_argument('--span-deg', type=float, default=170., help='Search range per direction (the cap).')
    parser.add_argument('--clearance-mm', type=float, default=3.)
    parser.add_argument('--margin-deg', type=float, default=5.)
    parser.add_argument('--joints', nargs='+', help='Only sweep these joints.')
    parser.add_argument('--output', type=Path, default=paths.BUILD_DIR / 'humanoid_training/joint_limits.json')
    args = parser.parse_args()
    robot, _ = load_training_model()
    nominal = load_standing_settings()['nominal_pose']['joint_positions']
    standing_pose = {joint.get('name'): nominal.get(joint.get('name'), 0.) for joint in robot.findall('joint')}
    report = measure_joint_limits(robot, standing_pose, args.step_deg, args.span_deg, args.clearance_mm,
                                  args.margin_deg, args.joints, log=lambda line: print(line, flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('Wrote', args.output)


if __name__ == '__main__':
    main()
