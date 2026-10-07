"""The arm with joint play and bracket flex: a simulation-only test model of a 3D-printed arm.

The reach task's arm (:mod:`arthrobot_tasks.arm_reach.env_cfg`) is perfectly rigid: each
MG5010 servo drives its link directly. A 3D-printed arm is not. Here every arm
joint becomes a chain of four joints on the spawned stage copy (arm.usd is
not modified):

    parent --revolute_i--> motor_out_i --gear_i--> gear_out_i --flex_i--> bracket_i --play_i--> link
             servo          motor        gearbox    output        PLA       bracket    printed
                            encoder      backlash   encoder       bending              play

- revolute_i: the unchanged servo (MG5010 drive, gains, torque and speed caps,
  reflected rotor inertia). Its angle is what the 18-bit motor encoder reads.
- gear_i: gearbox backlash, free rotation inside +-gap/2. The MG5010E-i36 also
  has a 14-bit encoder on the reducer output, which reads revolute_i + gear_i.
- flex_i: a lightly damped torsional spring about the joint axis, standing in
  for the PLA bracket bending.
- play_i: a second gap after the output encoder: looseness in the printed
  parts (screws in PLA, rod clamps). Neither encoder sees flex_i or play_i.

The extra bodies weigh 2 g each and sit on the joint axis, so masses and gravity
torques are essentially unchanged. Each flex joint carries an armature of 10% of
the inertia outboard of it: PhysX's iterative solver does not converge with the
near-massless bracket body alone (at 2e-6 kg m^2 and 8 iterations the springs
bent 46 deg under gravity instead of 0.9 deg). With the armature and 16
iterations the static deflections match k = torque / angle within 1%. It slows
each bracket's vibration by about 5%. Play and flex act only about each
joint's own axis; bending in other directions is not modeled. Values are
guesses until the real arm is measured; they are set per environment at run
time (apply_looseness), so one scene can hold several levels side by side.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim.utils import clone

from arthrobot_assets.arm import ARM_JOINTS
from arthrobot_tasks.arm_reach.env_cfg import ARM_CFG, spawn_training_arm

GEAR_JOINTS = [f'gear_{i}' for i in range(1, 7)]
FLEX_JOINTS = [f'flex_{i}' for i in range(1, 7)]
PLAY_JOINTS = [f'play_{i}' for i in range(1, 7)]
EXTRA_BODY_MASS_KG = 0.002
EXTRA_BODY_INERTIA_KG_M2 = 2e-6
FLEX_RANGE_DEG = 20.
FLEX_ARMATURE_FRACTION = 0.1
GEAR_ARMATURE_FRACTION = 0.1
SOLVER_POSITION_ITERATIONS = 16
# Flex damping as a fraction of critical damping, using the outboard inertia about each axis.
FLEX_DAMPING_RATIO = 0.05
# Outboard inertia about each joint axis at the zero pose (kinematics.ArmModel.axis_inertia), kg m^2.
AXIS_INERTIA_ZERO_POSE = (0.33537, 0.26683, 0.08227, 0.10062, 0.01352, 0.00085)


@dataclass(frozen=True)
class Looseness:
    name: str
    label: str
    gear_deg: float  # Gearbox backlash per joint, inside the output encoder (gap +-gear_deg / 2).
    play_deg: float  # Play in the printed parts, outside both encoders (gap +-play_deg / 2).
    flex_nm_per_rad: float  # Torsional stiffness of each bracket about its joint axis.


NONE = 0.001  # deg; a "closed" gap.
# Four amounts of looseness (total play, bracket stiffness). Each is run twice: with all of
# the play in the gearbox, where the output encoder sees it, and all of it in the printed
# brackets, where no encoder does. The real arm will be somewhere in between.
AMOUNTS = (('tight', 0.25, 2000.), ('typical', 0.5, 1000.), ('loose', 1.0, 500.), ('very_loose', 2.0, 250.))
LEVELS = {level.name: level for level in [
    # Validation only: the loose model with its looseness removed must match the rigid arm.
    Looseness('no_looseness', 'loose model, no play or flex', NONE, NONE, 1e5),
    Looseness('flex_only', 'flex only: 500 N m/rad, no play', NONE, NONE, 500.),
] + [level for name, play, flex in AMOUNTS for level in (
    Looseness(f'{name}_gearbox', f'{name.replace("_", " ")}: {play:g} deg play in gearbox, {flex:g} N m/rad',
              play, NONE, flex),
    Looseness(f'{name}_bracket', f'{name.replace("_", " ")}: {play:g} deg play in brackets, {flex:g} N m/rad',
              NONE, play, flex))]}


def _add_body(stage, path, transform, com):
    from pxr import Gf, UsdGeom, UsdPhysics
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTransformOp().Set(transform)
    prim = body.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(EXTRA_BODY_MASS_KG)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(EXTRA_BODY_INERTIA_KG_M2, EXTRA_BODY_INERTIA_KG_M2, EXTRA_BODY_INERTIA_KG_M2))
    mass.CreateCenterOfMassAttr(com)
    return prim.GetPath()


def _add_joint(stage, path, body0, body1, local_pos, local_rot, axis, limit_deg):
    from pxr import UsdPhysics
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    # Both bodies are posed like the original child, so the joint frame is the same on each side.
    joint.CreateLocalPos0Attr(local_pos)
    joint.CreateLocalRot0Attr(local_rot)
    joint.CreateLocalPos1Attr(local_pos)
    joint.CreateLocalRot1Attr(local_rot)
    joint.CreateAxisAttr(axis)
    joint.CreateLowerLimitAttr(-limit_deg)
    joint.CreateUpperLimitAttr(limit_deg)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
    drive.CreateStiffnessAttr(0.)
    drive.CreateDampingAttr(0.)


@clone
def spawn_loose_arm(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """The reach task's training arm, then motor_out_i / gear_out_i / bracket_i bodies and gear_i / flex_i /
    play_i joints."""
    from pxr import UsdGeom, UsdPhysics
    prim = spawn_training_arm.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    stage, root = prim.GetStage(), prim.GetPath()
    for i, name in enumerate(ARM_JOINTS, 1):
        servo = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(root.AppendPath(f'joints/{name}')))
        parent, = servo.GetBody0Rel().GetTargets()
        child, = servo.GetBody1Rel().GetTargets()
        local_pos, local_rot = servo.GetLocalPos1Attr().Get(), servo.GetLocalRot1Attr().Get()
        axis = servo.GetAxisAttr().Get()
        transform = UsdGeom.Xformable(stage.GetPrimAtPath(child)).GetLocalTransformation()
        motor = _add_body(stage, root.AppendChild(f'motor_out_{i}'), transform, local_pos)
        output = _add_body(stage, root.AppendChild(f'gear_out_{i}'), transform, local_pos)
        bracket = _add_body(stage, root.AppendChild(f'bracket_{i}'), transform, local_pos)
        servo.GetBody1Rel().SetTargets([motor])
        _add_joint(stage, root.AppendPath(f'joints/gear_{i}'), motor, output, local_pos, local_rot, axis, 0.5)
        _add_joint(stage, root.AppendPath(f'joints/flex_{i}'), output, bracket, local_pos, local_rot, axis, FLEX_RANGE_DEG)
        _add_joint(stage, root.AppendPath(f'joints/play_{i}'), bracket, child, local_pos, local_rot, axis, 0.5)
        # Parent and link are no longer directly jointed, so PhysX would let their
        # overlapping housings collide; filter them as the original joint did.
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(parent)).GetFilteredPairsRel().AddTarget(child)
    return prim


def loose_arm_cfg(prim_path: str, pos: tuple[float, float, float]) -> ArticulationCfg:
    actuators = dict(ARM_CFG.actuators)
    # Stiffness and damping are overwritten per environment by apply_looseness.
    actuators['flex'] = ImplicitActuatorCfg(
        joint_names_expr=FLEX_JOINTS, stiffness=1000., damping=1., effort_limit_sim=1e4, velocity_limit_sim=100.,
        armature={name: FLEX_ARMATURE_FRACTION * inertia for name, inertia in zip(FLEX_JOINTS, AXIS_INERTIA_ZERO_POSE)})
    actuators['gear'] = ImplicitActuatorCfg(
        joint_names_expr=GEAR_JOINTS, stiffness=0., damping=0., effort_limit_sim=0., velocity_limit_sim=100.,
        armature={name: GEAR_ARMATURE_FRACTION * inertia for name, inertia in zip(GEAR_JOINTS, AXIS_INERTIA_ZERO_POSE)})
    actuators['play'] = ImplicitActuatorCfg(
        joint_names_expr=PLAY_JOINTS, stiffness=0., damping=0., effort_limit_sim=0., velocity_limit_sim=100.,
        armature=0.)
    return ARM_CFG.replace(
        prim_path=prim_path,
        spawn=ARM_CFG.spawn.replace(
            func=spawn_loose_arm,
            articulation_props=ARM_CFG.spawn.articulation_props.replace(
                solver_position_iteration_count=SOLVER_POSITION_ITERATIONS)),
        init_state=ARM_CFG.init_state.replace(pos=pos),
        actuators=actuators)


def flex_damping(stiffness: torch.Tensor) -> torch.Tensor:
    """[N, 6] stiffness -> damping at FLEX_DAMPING_RATIO of critical."""
    inertia = torch.tensor(AXIS_INERTIA_ZERO_POSE, device=stiffness.device)
    return 2 * FLEX_DAMPING_RATIO * (stiffness * inertia).sqrt()


def apply_looseness(robot: Articulation, levels: list[Looseness]):
    """Write each environment's gaps and bracket stiffness (one level per environment)."""
    device = robot.device
    flex_ids = robot.find_joints(FLEX_JOINTS, preserve_order=True)[0]
    for names, field in ((GEAR_JOINTS, 'gear_deg'), (PLAY_JOINTS, 'play_deg')):
        half_gap = torch.tensor([math.radians(getattr(level, field)) / 2 for level in levels], device=device)
        limits = torch.stack([-half_gap, half_gap], -1)[:, None].repeat(1, 6, 1)
        robot.write_joint_position_limit_to_sim(limits, joint_ids=robot.find_joints(names, preserve_order=True)[0],
                                                warn_limit_violation=False)
    stiffness = torch.tensor([level.flex_nm_per_rad for level in levels], device=device)[:, None].repeat(1, 6)
    robot.write_joint_stiffness_to_sim(stiffness, joint_ids=flex_ids)
    robot.write_joint_damping_to_sim(flex_damping(stiffness), joint_ids=flex_ids)
