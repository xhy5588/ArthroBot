"""Training USD assets of the humanoid (requires a running Isaac Sim app).

``robot.usd`` is the 21-body training model with a free torso:
- direct torque control: implicit drives are authored with 80/4 N m/rad gains but
  every task overrides them (the wheels' drives are always off);
- convex-decomposition colliders, self-collision, 32/8 solver iterations.

``getup_robot.usd`` is a light override layer on top of it for the get-up task:
one convex hull per collision mesh (the same geometry the joint-limit sweep uses),
the measured self-collision joint limits on the 18 limb joints, and no visual
meshes unless requested (videos and previews).
"""
import fcntl
import hashlib
import json
import math
from pathlib import Path

from arthrobot.motors import POSITION_GAIN_NM_PER_RAD, VELOCITY_GAIN_NM_S_PER_RAD, load_motor
from arthrobot_assets.humanoid.training_model import make_training_model, output_dir

TRAINING_VARIANT = 'free-torso:parked-grippers:v1'
GETUP_VARIANT = 'getup:v1'
JOINT_LIMITS_FILE = Path(__file__).resolve().parent / 'getup/data/joint_limits.json'
EXPECTED_BODIES, EXPECTED_JOINTS, EXPECTED_COLLIDERS = 21, 20, 150
CONTACT_OFFSET_M, MAX_DEPENETRATION_M_S = .001, 1.


def ensure_training_usd(rebuild: bool = False) -> tuple[Path, dict]:
    """Return ``build/humanoid_training/robot.usd`` (re-imported if the training URDF changed) and its model summary."""
    from arthrobot.sim.usd_import import import_if_changed
    urdf_path, model = make_training_model()
    usd_path = output_dir() / 'robot.usd'
    if not import_if_changed(urdf_path, usd_path, TRAINING_VARIANT, _configure_training_usd, rebuild,
                             fix_base=False, collision_from_visuals=False):
        print(f'HUMANOID_ASSET: using cached {usd_path}', flush=True)
    return usd_path, model


def _configure_training_usd(usd_path: Path) -> None:
    from pxr import PhysxSchema, Usd, UsdPhysics
    from arthrobot.sim.usd_physics import (configure_articulation, configure_colliders, configure_joint_drive,
                                           make_collision_meshes_editable, set_max_depenetration_velocity)
    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim()
    make_collision_meshes_editable(root)
    configure_articulation(root, position_iterations=32, velocity_iterations=8, self_collisions=True)
    set_max_depenetration_velocity(root, MAX_DEPENETRATION_M_S)
    motor = load_motor('mg5010')
    bodies = joints = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            bodies += 1
        if prim.IsA(UsdPhysics.Joint):
            assert not prim.IsA(UsdPhysics.FixedJoint), 'The training asset must not be anchored to the world'
        if prim.IsA(UsdPhysics.RevoluteJoint):
            is_wheel = prim.GetName().endswith('_wheel')
            configure_joint_drive(prim, 0. if is_wheel else POSITION_GAIN_NM_PER_RAD,
                                  0. if is_wheel else VELOCITY_GAIN_NM_S_PER_RAD, motor.rated_torque_nm,
                                  motor.max_speed_deg_s, motor.reflected_inertia_kg_m2)
            joints += 1
    colliders = configure_colliders(root, 'convexDecomposition', CONTACT_OFFSET_M)
    assert (bodies, joints, colliders) == (EXPECTED_BODIES, EXPECTED_JOINTS, EXPECTED_COLLIDERS), (bodies, joints, colliders)
    stage.GetRootLayer().Save()


def ensure_getup_usd(with_visuals: bool = False) -> tuple[Path, dict]:
    """Return the get-up override layer (rebuilt when the base asset or the limits change) and the limits."""
    from pxr import Sdf, Usd, UsdPhysics
    base_path, _ = ensure_training_usd()
    limits = json.loads(JOINT_LIMITS_FILE.read_text())
    usd_path = output_dir() / ('getup_robot_visual.usd' if with_visuals else 'getup_robot.usd')
    signature = hashlib.sha256(base_path.with_suffix('.revision').read_bytes() + JOINT_LIMITS_FILE.read_bytes()
                               + f'{GETUP_VARIANT}:visuals={with_visuals}'.encode()).hexdigest()
    stamp = usd_path.with_suffix('.revision')
    with (usd_path.parent / '.import.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if usd_path.exists() and stamp.exists() and stamp.read_text() == signature:
            return usd_path, limits
        base = Usd.Stage.Open(str(base_path))
        layer = Sdf.Layer.FindOrOpen(str(usd_path)) if usd_path.exists() else Sdf.Layer.CreateNew(str(usd_path))
        layer.Clear()
        layer.subLayerPaths.append(base_path.name)
        layer.defaultPrim = base.GetDefaultPrim().GetName()
        stage = Usd.Stage.Open(layer)
        stage.SetEditTarget(layer)
        hulls = limited = hidden = 0
        for prim in Usd.PrimRange(stage.GetDefaultPrim()):
            if prim.GetName() == 'visuals' and not with_visuals:
                stage.OverridePrim(prim.GetPath()).SetActive(False)
                hidden += 1
            elif prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.MeshCollisionAPI(stage.OverridePrim(prim.GetPath())).CreateApproximationAttr().Set('convexHull')
                hulls += 1
            elif prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in limits['limits']:
                joint = UsdPhysics.RevoluteJoint(stage.OverridePrim(prim.GetPath()))
                entry = limits['limits'][prim.GetName()]
                joint.CreateLowerLimitAttr().Set(math.degrees(entry['lower']))
                joint.CreateUpperLimitAttr().Set(math.degrees(entry['upper']))
                limited += 1
        assert hulls == EXPECTED_COLLIDERS and limited == 18 and (with_visuals or hidden == EXPECTED_BODIES)
        layer.Save()
        stamp.write_text(signature)
        print('GETUP_ASSET: ' + json.dumps(dict(path=str(usd_path), convex_hull_colliders=hulls,
                                                limited_joints=limited, hidden_visual_groups=hidden)), flush=True)
        return usd_path, limits
