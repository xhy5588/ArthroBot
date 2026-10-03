"""Author the DM4310 rack-and-pinion drive on a USD robot.

Only jaw 0 is driven: a linear position drive whose force cap equals the motor
torque divided by the pinion radius. Jaw 1 and the pinion follow jaw 0 through
PhysX mimic joints (gearing -1 for the opposed jaw; -180 / (pi r) degrees per
meter for the pinion). The pinion's own drive is disabled. Velocity caps are not
authored on the gripper joints, because hard velocity constraints distort contact
forces; commands are rate-limited instead.
"""
import math

from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

from arthrobot.gripper import DM4310_GRIPPER, RackPinionGripper


def configure_rack_pinion(pinion_prim: Usd.Prim, jaw_prims: list[Usd.Prim],
                          gripper: RackPinionGripper = DM4310_GRIPPER) -> dict:
    """Configure drives, limits and mimic couplings; returns the gripper summary."""
    from omni.physx.bindings._physx import (MIMIC_JOINT_ATTRIBUTE_NAME_DAMPING_RATIO_ROTX,
                                            MIMIC_JOINT_ATTRIBUTE_NAME_NATURAL_FREQUENCY_ROTX)
    pinion_drive = UsdPhysics.DriveAPI.Apply(pinion_prim, 'angular')
    pinion_drive.CreateTypeAttr('force')
    for create in (pinion_drive.CreateStiffnessAttr, pinion_drive.CreateDampingAttr, pinion_drive.CreateMaxForceAttr,
                   pinion_drive.CreateTargetPositionAttr, pinion_drive.CreateTargetVelocityAttr):
        create(0.)
    pinion = UsdPhysics.RevoluteJoint(pinion_prim)
    pinion.CreateLowerLimitAttr(0.)
    pinion.CreateUpperLimitAttr(math.degrees(gripper.full_stroke_pinion_angle_rad))
    pinion_physx = PhysxSchema.PhysxJointAPI.Apply(pinion_prim)
    pinion_physx.CreateMaxJointVelocityAttr(float('inf'))
    pinion_physx.CreateArmatureAttr(0.)   # rotor inertia unknown; never borrow the arm's armature

    for index, jaw_prim in enumerate(jaw_prims):
        driven = index == 0
        jaw_prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, 'rotX')
        drive = UsdPhysics.DriveAPI.Apply(jaw_prim, 'linear')
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(gripper.linear_stiffness_n_per_m if driven else 0.)
        drive.CreateDampingAttr(gripper.linear_damping_n_s_per_m if driven else 0.)
        drive.CreateMaxForceAttr(gripper.jaw_force_limit_n if driven else 0.)
        drive.CreateTargetPositionAttr(0.)
        drive.CreateTargetVelocityAttr(0.)
        jaw = UsdPhysics.PrismaticJoint(jaw_prim)
        jaw.CreateLowerLimitAttr(0.)
        jaw.CreateUpperLimitAttr(gripper.jaw_stroke_m)
        PhysxSchema.PhysxJointAPI.Apply(jaw_prim).CreateMaxJointVelocityAttr(float('inf'))

    pinion_gearing_deg_per_m = -180. / (math.pi * gripper.pinion_radius_m)
    for follower, gearing in ((jaw_prims[1], -1.), (pinion_prim, pinion_gearing_deg_per_m)):
        mimic = PhysxSchema.PhysxMimicJointAPI.Apply(follower, 'rotX')
        mimic.CreateReferenceJointRel().SetTargets([jaw_prims[0].GetPath()])
        mimic.CreateGearingAttr(gearing)
        mimic.CreateOffsetAttr(0.)
        # Slight compliance stabilizes contact; it is a solver setting, not measured elasticity.
        follower.CreateAttribute(MIMIC_JOINT_ATTRIBUTE_NAME_NATURAL_FREQUENCY_ROTX,
                                 Sdf.ValueTypeNames.Float).Set(gripper.mimic_natural_frequency)
        follower.CreateAttribute(MIMIC_JOINT_ATTRIBUTE_NAME_DAMPING_RATIO_ROTX,
                                 Sdf.ValueTypeNames.Float).Set(gripper.mimic_damping_ratio)
    return gripper.summary()
