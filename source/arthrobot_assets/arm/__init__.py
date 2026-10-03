"""6-DOF ArthroBot arm: six motor modules and a DM4310 rack-and-pinion gripper.

The CAD export in ``cad/assembly_2`` is used read-only; :mod:`.build` regroups
its 617 part meshes into ten rigid bodies and writes ``build/arm/arm.urdf``.
"""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CAD_DIR = PACKAGE_DIR / 'cad'
SOURCE_URDF = CAD_DIR / 'assembly_2/urdf/assembly_2.urdf'

ARM_BODIES = ('base', 'shoulder_output', 'upper_arm_output', 'elbow_output',
              'forearm_output', 'wrist_output', 'tool_output')
GRIPPER_BODIES = ('jaw_0', 'jaw_1', 'gripper_pinion')
BODIES = ARM_BODIES + GRIPPER_BODIES

ARM_JOINTS = tuple(f'joint_{number}' for number in range(1, 7))
JAW_JOINTS = ('gripper_jaw_0', 'gripper_jaw_1')
PINION_JOINT = 'gripper_pinion_joint'
TOOL_BODY = 'tool_output'
