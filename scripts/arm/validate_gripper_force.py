"""Measure DM4310 gripper contact forces in an isolated, lossless rack-and-pinion fixture.

A pinion and two opposed jaws (with the same drive authoring as the robots)
squeeze a free block on a frictionless guide, at 0.5, 1 and 3 N m torque limits:
each jaw must press with torque / (2 r). A fourth case offsets a fixed block so
that only one jaw touches it: that jaw must carry the full torque / r while the
coupling stops the other. Finally the gripper must reopen. Zero gravity, 960 Hz.
This verifies the simulated transmission, not real gripper strength.

Writes ``build/arm/gripper_force_validation.json``.
"""
import json
import sys
import traceback

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx import get_physx_simulation_interface  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402

from arthrobot import paths  # noqa: E402
from arthrobot.gripper import DM4310_GRIPPER as GRIPPER, VALIDATED_PHYSICS_DT as DT  # noqa: E402
from arthrobot.sim.gripper_drive import configure_rack_pinion  # noqa: E402

RIG = '/World/Rig'
BLOCK = '/World/ForceBlock'
CASES = ((.5, 0.), (1., 0.), (3., 0.), (3., .01))   # (motor torque limit N m, block offset m)
SETTLE_S, MEASURE_S = 6., 2.


def add_body(stage, name, position, with_pad=False):
    path = f'{RIG}/{name}'
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(.1)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-4))
    if with_pad:
        pad = UsdGeom.Cube.Define(stage, path + '/collision')
        pad.CreateSizeAttr(1.)
        pad.AddScaleOp().Set(Gf.Vec3f(.01, .02, .02))
        UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(pad.GetPrim()).CreateContactOffsetAttr(.0001)
    return prim


def build_fixture(stage):
    """Articulated pinion + two jaws on a base, and a load block on a guide."""
    UsdGeom.Xform.Define(stage, RIG)
    base = add_body(stage, 'base', (0, 0, 0))
    add_body(stage, 'pinion', (0, 0, 1))
    UsdPhysics.ArticulationRootAPI.Apply(base)
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(base)
    articulation.CreateSolverPositionIterationCountAttr(128)
    articulation.CreateSolverVelocityIterationCountAttr(128)
    articulation.CreateSleepThresholdAttr(0.)
    UsdPhysics.FixedJoint.Define(stage, f'{RIG}/fixed').CreateBody1Rel().SetTargets([base.GetPath()])
    pinion = UsdPhysics.RevoluteJoint.Define(stage, f'{RIG}/pinion_joint')
    pinion.CreateBody0Rel().SetTargets([base.GetPath()])
    pinion.CreateBody1Rel().SetTargets([f'{RIG}/pinion'])
    pinion.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 1))
    pinion.CreateAxisAttr('Z')
    jaw_joints = []
    for index, side in enumerate((-1, 1)):
        jaw = add_body(stage, f'jaw_{index}', (side * .06, 0, 1), with_pad=True)
        PhysxSchema.PhysxContactReportAPI.Apply(jaw).CreateThresholdAttr(0.)
        joint = UsdPhysics.PrismaticJoint.Define(stage, f'{RIG}/jaw_joint_{index}')
        joint.CreateBody0Rel().SetTargets([base.GetPath()])
        joint.CreateBody1Rel().SetTargets([jaw.GetPath()])
        joint.CreateLocalPos0Attr(Gf.Vec3f(side * .06, 0, 1))
        joint.CreateAxisAttr('X')
        facing = Gf.Quatf(1.) if side < 0 else Gf.Quatf(0., 0., 0., 1.)   # both jaws close towards the center
        joint.CreateLocalRot0Attr(facing)
        joint.CreateLocalRot1Attr(facing)
        jaw_joints.append(joint.GetPrim())
    configure_rack_pinion(pinion.GetPrim(), jaw_joints)

    block = UsdGeom.Cube.Define(stage, BLOCK)
    block.CreateSizeAttr(1.)
    block_position = block.AddTranslateOp()
    block_position.Set(Gf.Vec3d(0, 0, 1))
    block.AddScaleOp().Set(Gf.Vec3f(.04, .03, .03))
    UsdPhysics.CollisionAPI.Apply(block.GetPrim())
    block_body = UsdPhysics.RigidBodyAPI.Apply(block.GetPrim())
    block_body.CreateRigidBodyEnabledAttr(True)
    block_body.CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(block.GetPrim()).CreateMassAttr(.5)
    PhysxSchema.PhysxRigidBodyAPI.Apply(block.GetPrim()).CreateSleepThresholdAttr(0.)
    PhysxSchema.PhysxCollisionAPI.Apply(block.GetPrim()).CreateContactOffsetAttr(.0001)
    # Keep the load faces parallel while leaving the block free along the squeeze axis.
    guide = UsdPhysics.PrismaticJoint.Define(stage, '/World/LoadGuide')
    guide.CreateBody1Rel().SetTargets([block.GetPath()])
    guide.CreateAxisAttr('X')
    guide.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 1))
    return block_body, block_position, guide


def main():
    world = World(stage_units_in_meters=1., physics_dt=DT, rendering_dt=DT)
    world.get_physics_context().set_gravity(0.)
    world.get_physics_context().set_solver_type('PGS')
    stage = omni.usd.get_context().get_stage()
    block_body, block_position, guide = build_fixture(stage)
    impulse = np.zeros(2)

    def on_contact(headers, data):
        for header in headers:
            first, second = (str(PhysicsSchemaTools.intToSdfPath(path)) for path in (header.collider0, header.collider1))
            for index in range(2):
                if BLOCK in (first, second) and any(path.startswith(f'{RIG}/jaw_{index}/') for path in (first, second)):
                    for contact in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                        impulse[index] += abs(data[contact].impulse[0])

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)  # noqa: F841
    rig = world.scene.add(Articulation(RIG, name='rig'))
    results = []
    for torque_nm, offset_m in CASES:
        # A free block makes both jaw forces equal at rest; a fixed, offset block loads one jaw only.
        one_jaw = bool(offset_m)
        block_body.CreateKinematicEnabledAttr(one_jaw)
        guide.CreateJointEnabledAttr(not one_jaw)
        block_position.Set(Gf.Vec3d(offset_m, 0, 1))
        block_body.CreateVelocityAttr(Gf.Vec3f(0.))
        block_body.CreateAngularVelocityAttr(Gf.Vec3f(0.))
        world.reset()
        pinion_id = rig.dof_names.index('pinion_joint')
        jaw_ids = [rig.dof_names.index(f'jaw_joint_{index}') for index in range(2)]
        rig.set_joint_positions(np.zeros((1, 3), dtype=np.float32))
        rig.set_joint_velocities(np.zeros((1, 3), dtype=np.float32))
        rig.set_max_efforts(np.array([[torque_nm / GRIPPER.pinion_radius_m]], dtype=np.float32), joint_indices=[jaw_ids[0]])
        target = np.zeros((1, 3), dtype=np.float32)
        target[0, pinion_id] = GRIPPER.full_stroke_pinion_angle_rad
        target[0, jaw_ids[0]] = GRIPPER.jaw_stroke_m
        rig.set_joint_position_targets(target)
        forces = []
        for step in range(round((SETTLE_S + MEASURE_S) / DT)):
            impulse[:] = 0
            world.step(render=False)
            if step >= round(SETTLE_S / DT):
                forces.append((impulse / DT).copy())
        positions = rig.get_joint_positions()[0]
        measured = np.mean(forces, axis=0)
        expected = [0., torque_nm / GRIPPER.pinion_radius_m] if one_jaw else [torque_nm / (2 * GRIPPER.pinion_radius_m)] * 2
        coupling_error = float(np.max(np.abs(positions[jaw_ids] - positions[pinion_id] * GRIPPER.pinion_radius_m)))
        result = dict(motor_torque_cap_nm=torque_nm, block_offset_m=offset_m,
                      measured_contact_force_per_jaw_n=measured.tolist(), contact_force_std_n=np.std(forces, axis=0).tolist(),
                      expected_ideal_force_per_jaw_n=expected, jaw_travel_m=positions[jaw_ids].tolist(),
                      coupling_error_m=coupling_error)
        results.append(result)
        print('GRIPPER_FORCE_CHECK: ' + json.dumps(result), flush=True)
        np.testing.assert_allclose(measured, expected, rtol=.03, atol=.5)
        assert coupling_error < .0001
        assert max(positions[jaw_ids]) < .04, 'A jaw passed through the block'
    rig.set_joint_position_targets(np.zeros((1, 3), dtype=np.float32))
    for _ in range(round(4 / DT)):
        world.step(render=False)
    np.testing.assert_allclose(rig.get_joint_positions()[0, jaw_ids], 0., atol=.0001)
    output = paths.build_dir('arm') / 'gripper_force_validation.json'
    output.write_text(json.dumps(dict(passed=True, fixture='Ideal lossless rack and pinion, zero gravity',
                                      checks=results, reopening_passed=True), indent=2) + '\n')
    print(f'GRIPPER_FORCE_VALIDATED: {output}', flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    print('GRIPPER_FORCE_FAILED', flush=True)
    sys.exit(1)
finally:
    app.close()
