"""Simulation USD of the full humanoid (27 bodies, both grippers) for the viewer.

Training uses the lighter 21-body variant instead (see :mod:`arthrobot_tasks.humanoid.assets`).
Physics authored here:
- masses and principal inertias from the build report;
- 20 MG5010 position drives (80 N m/rad, 4 N m s/rad, 13 N m, 74 rpm, reflected armature);
- both DM4310 grippers (driven jaw, mimic jaw and pinion);
- convex-decomposition colliders, self-collision, 128/128 solver iterations.
"""
from pathlib import Path

from arthrobot import paths
from arthrobot.motors import POSITION_GAIN_NM_PER_RAD, VELOCITY_GAIN_NM_S_PER_RAD, load_motor
from arthrobot_assets.humanoid.build import write

CONTACT_OFFSET_M, MAX_DEPENETRATION_M_S = .0005, .2
SOLVER_ITERATIONS = 128


def output_dir() -> Path:
    return paths.build_dir('humanoid')


def ensure_humanoid_usd(fixed_torso: bool = True, rebuild: bool = False) -> tuple[Path, dict]:
    """Build the URDF and, if it changed, import and configure the USD; return the USD path and build report."""
    from arthrobot.sim.usd_import import import_if_changed
    directory = output_dir()
    urdf_path, report = write(directory)
    usd_path = directory / ('humanoid.usd' if fixed_torso else 'humanoid_free.usd')
    import_if_changed(urdf_path, usd_path, f'fixed-torso={fixed_torso}:v1', lambda path: _configure(path, report),
                      rebuild, fix_base=fixed_torso, collision_from_visuals=False)
    return usd_path, report


def _configure(usd_path: Path, report: dict) -> None:
    from pxr import Usd
    from arthrobot.sim.gripper_drive import configure_rack_pinion
    from arthrobot.sim.usd_physics import (apply_mass_properties, configure_articulation, configure_colliders,
                                           configure_joint_drive, filter_collision_pairs, joints_by_name,
                                           make_collision_meshes_editable, set_max_depenetration_velocity)
    stage = Usd.Stage.Open(str(usd_path))
    robot_path = str(stage.GetDefaultPrim().GetPath())
    root = stage.GetPrimAtPath(robot_path)
    apply_mass_properties(stage, robot_path, report['bodies'])
    set_max_depenetration_velocity(root, MAX_DEPENETRATION_M_S)
    motor = load_motor(report['hardware']['motor'])
    joints = joints_by_name(root)
    for spec in report['joints']:
        if spec['type'] == 'continuous':
            configure_joint_drive(joints[spec['name']], POSITION_GAIN_NM_PER_RAD, VELOCITY_GAIN_NM_S_PER_RAD,
                                  motor.rated_torque_nm, motor.max_speed_deg_s, motor.reflected_inertia_kg_m2)
    for gripper in report['grippers'].values():
        configure_rack_pinion(joints[gripper['pinion_joint']], [joints[name] for name in gripper['jaw_joints']])
        filter_collision_pairs(stage, [(f"{robot_path}/{gripper['pinion']}", f'{robot_path}/{jaw}')
                                       for jaw in gripper['jaws']])
    make_collision_meshes_editable(root)
    colliders = configure_colliders(root, 'convexDecomposition', CONTACT_OFFSET_M)
    assert colliders == report['collision_mesh_count'], ('Missing collision geometry', colliders)
    configure_articulation(root, SOLVER_ITERATIONS, SOLVER_ITERATIONS, self_collisions=True)
    stage.GetRootLayer().Save()
