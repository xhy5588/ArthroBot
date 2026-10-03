"""Record which pairs of robot bodies touched, for validation reports."""
from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdPhysics


class ContactMonitor:
    """Collect every contacting (body, body) pair under ``robot_path`` from PhysX contact reports."""

    def __init__(self, stage: Usd.Stage, robot_path: str):
        from omni.physx import get_physx_simulation_interface
        self.robot_path = robot_path.rstrip('/') + '/'
        self.pairs: set[tuple[str, str]] = set()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)

    def _on_contact(self, headers, _data) -> None:
        for header in headers:
            if not header.num_contact_data:
                continue
            first = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
            second = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            if first.startswith(self.robot_path) and second.startswith(self.robot_path):
                self.pairs.add(tuple(sorted((first, second))))
