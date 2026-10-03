"""Simulation USD of the arm for the viewer and for training (requires a running Isaac Sim app).

Physics authored on the imported asset:
- masses and principal inertias from the build report;
- six MG5010 position drives (80 N m/rad, 4 N m s/rad, rated torque and speed caps,
  reflected rotor inertia as armature);
- the DM4310 rack-and-pinion gripper (one driven jaw, mimic jaw and pinion);
- self-collision between non-neighboring bodies, convex-decomposition colliders,
  128 position / 128 velocity solver iterations.
"""
from pathlib import Path

from arthrobot import paths
from arthrobot.motors import POSITION_GAIN_NM_PER_RAD, VELOCITY_GAIN_NM_S_PER_RAD, load_motor
from arthrobot_assets.arm import ARM_BODIES, ARM_JOINTS, GRIPPER_BODIES, JAW_JOINTS, PINION_JOINT
from arthrobot_assets.arm.build import DEFAULT_MOTOR, write

USD_VARIANT = 'fixed-base:colliders-from-visuals:v1'
CONTACT_OFFSET_M, MAX_DEPENETRATION_M_S = .0005, .2
SOLVER_ITERATIONS = 128


def output_dir(motor_name: str = DEFAULT_MOTOR) -> Path:
    return paths.build_dir('arm' if motor_name == DEFAULT_MOTOR else f'arm_{motor_name}')


def ensure_arm_usd(motor_name: str = DEFAULT_MOTOR, gripper_mass_kg: float | None = None,
                   rebuild: bool = False) -> tuple[Path, dict]:
    """Build the URDF and, if it changed, import and configure the USD; return the USD path and build report."""
    from arthrobot.sim.usd_import import import_if_changed
    directory = output_dir(motor_name)
    urdf_path, report = write(directory, motor_name, gripper_mass_kg)
    usd_path = directory / 'arm.usd'
    import_if_changed(urdf_path, usd_path, USD_VARIANT, lambda path: _configure(path, report, motor_name), rebuild,
                      fix_base=True, collision_from_visuals=True)
    return usd_path, report


def _configure(usd_path: Path, report: dict, motor_name: str) -> None:
    from pxr import Usd
    from arthrobot.sim.gripper_drive import configure_rack_pinion
    from arthrobot.sim.usd_physics import (apply_mass_properties, configure_articulation, configure_colliders,
                                           configure_joint_drive, filter_collision_pairs, joints_by_name,
                                           make_collision_meshes_editable, set_max_depenetration_velocity)
    stage = Usd.Stage.Open(str(usd_path))
    robot_path = str(stage.GetDefaultPrim().GetPath())
    root = stage.GetPrimAtPath(robot_path)
    motor = load_motor(motor_name)
    apply_mass_properties(stage, robot_path, report['bodies'])
    joints = joints_by_name(root)
    for name in ARM_JOINTS:
        configure_joint_drive(joints[name], POSITION_GAIN_NM_PER_RAD, VELOCITY_GAIN_NM_S_PER_RAD,
                              motor.rated_torque_nm, motor.max_speed_deg_s, motor.reflected_inertia_kg_m2)
    configure_rack_pinion(joints[PINION_JOINT], [joints[name] for name in JAW_JOINTS])
    # Neighboring segments, and the pinion against its racks, never collide: the joints and
    # the mimic couplings already carry those forces. Every other pair of bodies may collide.
    neighbors = [(ARM_BODIES[index], ARM_BODIES[index + 1]) for index in range(len(ARM_BODIES) - 1)]
    neighbors += [(GRIPPER_BODIES[2], GRIPPER_BODIES[0]), (GRIPPER_BODIES[2], GRIPPER_BODIES[1])]
    filter_collision_pairs(stage, [(f'{robot_path}/{first}', f'{robot_path}/{second}') for first, second in neighbors])
    set_max_depenetration_velocity(root, MAX_DEPENETRATION_M_S)
    make_collision_meshes_editable(root)
    colliders = configure_colliders(root, 'convexDecomposition', CONTACT_OFFSET_M)
    assert colliders == report['included_visuals'], colliders
    configure_articulation(root, SOLVER_ITERATIONS, SOLVER_ITERATIONS, self_collisions=True)
    stage.GetRootLayer().Save()
