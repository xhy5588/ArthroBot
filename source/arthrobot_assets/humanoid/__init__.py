"""Wheel-legged ArthroBot humanoid: 20 motor modules, two arms with DM4310 grippers.

Each arm has six joints (``<side>_arm_1`` to ``_6``); each leg has two hip
joints, a knee and a driven wheel. The CAD export in
``cad/modular_humanoid_bipad`` is used read-only:

- :mod:`.build` regroups its 2,419 part meshes into 27 bodies (``build/humanoid/``);
- :mod:`.training_model` makes the 21-body training variant: grippers parked open,
  a forward-facing base frame and a wheel-balanced start pose;
- :mod:`.nominal_pose` solves the upright standing pose used by every task.
"""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CAD_DIR = PACKAGE_DIR / 'cad'
SOURCE_URDF = CAD_DIR / 'modular_humanoid_bipad/urdf/modular_humanoid_bipad.urdf'
HARDWARE_FILE = PACKAGE_DIR / 'hardware.json'

SIDES = ('left', 'right')
ARM_JOINTS = tuple(f'{side}_arm_{number}' for side in SIDES for number in range(1, 7))
LEG_JOINTS = tuple(f'{side}_{part}' for side in SIDES for part in ('hip_1', 'hip_2', 'knee'))
WHEEL_JOINTS = ('left_wheel', 'right_wheel')
# Motor order used by every policy: 12 arm joints, 6 leg joints, then the 2 wheels.
MOTOR_JOINTS = ARM_JOINTS + LEG_JOINTS + WHEEL_JOINTS
LIMB_JOINT_COUNT = len(ARM_JOINTS) + len(LEG_JOINTS)
WHEEL_BODIES = ('left_wheel_body', 'right_wheel_body')
