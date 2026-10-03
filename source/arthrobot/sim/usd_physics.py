"""Physics properties authored on an imported USD robot.

USD angular drive gains are per degree, while Isaac Lab and the runtime APIs use
radians; the helpers below take SI/radian values and convert.
"""
import math

import numpy as np
from pxr import Gf, PhysxSchema, Usd, UsdPhysics
from scipy.spatial.transform import Rotation


def apply_mass_properties(stage: Usd.Stage, robot_path: str, bodies: dict) -> None:
    """Set mass, center of mass and principal inertia of every body from a build report's ``bodies``."""
    for name, properties in bodies.items():
        prim = stage.GetPrimAtPath(f'{robot_path}/{name}')
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f'Missing rigid body: {name}')
        moments, axes = np.linalg.eigh(np.array(properties['inertia_kg_m2']))
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        x, y, z, w = Rotation.from_matrix(axes).as_quat()
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(properties['mass_kg'])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*properties['center_of_mass_m']))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*moments))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))


def configure_joint_drive(joint_prim: Usd.Prim, stiffness_nm_per_rad: float, damping_nm_s_per_rad: float,
                          max_torque_nm: float, max_speed_deg_s: float, armature_kg_m2: float) -> None:
    """Force-type angular position drive, speed cap and reflected rotor inertia on a revolute joint."""
    drive = UsdPhysics.DriveAPI.Apply(joint_prim, 'angular')
    drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(stiffness_nm_per_rad * math.pi / 180)
    drive.CreateDampingAttr(damping_nm_s_per_rad * math.pi / 180)
    drive.CreateMaxForceAttr(max_torque_nm)
    drive.CreateTargetPositionAttr(0.)
    drive.CreateTargetVelocityAttr(0.)
    joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
    joint.CreateMaxJointVelocityAttr(max_speed_deg_s)
    joint.CreateArmatureAttr(armature_kg_m2)


def joints_by_name(root: Usd.Prim) -> dict[str, Usd.Prim]:
    return {prim.GetName(): prim for prim in Usd.PrimRange(root) if prim.IsA(UsdPhysics.Joint)}


def make_collision_meshes_editable(root: Usd.Prim) -> None:
    """The importer instances collision meshes; un-instance them so their approximation can be set."""
    for _ in range(4):
        instances = [prim for prim in Usd.PrimRange(root) if prim.IsInstance() and 'collisions' in str(prim.GetPath())]
        if not instances:
            return
        for prim in instances:
            prim.SetInstanceable(False)


def configure_colliders(root: Usd.Prim, approximation: str, contact_offset_m: float, rest_offset_m: float = 0.) -> int:
    """Set every collider's mesh approximation and offsets; return the collider count."""
    count = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(contact_offset_m)
            collision.CreateRestOffsetAttr(rest_offset_m)
            count += 1
    return count


def configure_articulation(root: Usd.Prim, position_iterations: int, velocity_iterations: int,
                           self_collisions: bool = True) -> list[str]:
    """Solver iterations and self-collision on every articulation root; returns their paths."""
    roots = []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            api.CreateEnabledSelfCollisionsAttr(self_collisions)
            api.CreateSleepThresholdAttr(0.)
            api.CreateSolverPositionIterationCountAttr(position_iterations)
            api.CreateSolverVelocityIterationCountAttr(velocity_iterations)
            roots.append(str(prim.GetPath()))
    if not roots:
        raise RuntimeError('No articulation root found')
    return roots


def set_max_depenetration_velocity(root: Usd.Prim, velocity_m_s: float) -> None:
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr(velocity_m_s)


def filter_collision_pairs(stage: Usd.Stage, pairs: list[tuple[str, str]]) -> None:
    """Disable contact between each pair of prims (absolute paths)."""
    for first, second in pairs:
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(first)).CreateFilteredPairsRel().AddTarget(second)
