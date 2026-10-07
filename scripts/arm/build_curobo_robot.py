"""Build the arm's cuRobo model: a planning URDF and collision spheres fitted to the CAD meshes.

Needs cuRobo v2 (https://github.com/NVlabs/curobo, installed into the Isaac Lab environment; see
docs/arm_planning.md). Takes about 4 minutes. Writes build/arm_planning/arm_planning.urdf and
arm_curobo.yml (see arthrobot_tasks.arm_planning.curobo_robot).

    python scripts/arm/build_curobo_robot.py
"""
import argparse

from arthrobot_tasks.arm_planning import curobo_robot


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sphere-density', type=float, default=1.0)
    parser.add_argument('--collision-samples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(f'Wrote {curobo_robot.write_planning_urdf()}')
    print(f'Wrote {curobo_robot.build_config(args.sphere_density, args.collision_samples, args.seed)}')


if __name__ == '__main__':
    main()
