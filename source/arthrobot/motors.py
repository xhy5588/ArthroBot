"""Joint-module motor specifications (planetary servo motors).

Every ArthroBot joint is one motor module. Specifications come from supplier
listings (``data/motors.json``); torque and speed caps are the continuous
ratings, and the peak torque is recorded but never used in simulation because
there is no thermal model.
"""
from dataclasses import dataclass
import json
import math

from arthrobot.paths import DATA_DIR

# Simulated position-controller gains for a joint module. These are simulation
# choices shared by every robot, not the motor driver's own controller gains.
POSITION_GAIN_NM_PER_RAD = 80.0
VELOCITY_GAIN_NM_S_PER_RAD = 4.0


@dataclass(frozen=True)
class MotorSpec:
    name: str
    manufacturer: str
    model: str
    mass_kg: float
    gear_ratio: float
    rated_torque_nm: float
    peak_torque_nm: float
    rated_speed_rpm: float
    max_speed_rpm: float
    rotor_inertia_kg_m2: float
    source: dict

    @property
    def max_speed_rad_s(self) -> float:
        return self.max_speed_rpm * math.pi / 30

    @property
    def max_speed_deg_s(self) -> float:
        return self.max_speed_rpm * 6.

    @property
    def reflected_inertia_kg_m2(self) -> float:
        """Rotor inertia seen at the output shaft; simulated as joint armature."""
        return self.rotor_inertia_kg_m2 * self.gear_ratio ** 2

    def summary(self) -> dict:
        return dict(name=self.name, manufacturer=self.manufacturer, model=self.model, mass_kg=self.mass_kg,
                    gear_ratio=self.gear_ratio, rated_torque_nm=self.rated_torque_nm,
                    peak_torque_nm=self.peak_torque_nm, max_speed_rpm=self.max_speed_rpm,
                    rotor_inertia_kg_m2=self.rotor_inertia_kg_m2,
                    reflected_inertia_kg_m2=self.reflected_inertia_kg_m2, source=self.source)


def load_motor(name: str) -> MotorSpec:
    """Load a motor by key, e.g. ``'mg5010'`` or ``'mg4010'``."""
    catalog = json.loads((DATA_DIR / 'motors.json').read_text())
    if name not in catalog:
        raise KeyError(f'Unknown motor {name!r}; available: {sorted(catalog)}')
    record = catalog[name]
    return MotorSpec(name=name, manufacturer=record['manufacturer'], model=record['model'],
                     mass_kg=record['mass_kg'], gear_ratio=record['gear_ratio'],
                     rated_torque_nm=record['rated_torque_nm'], peak_torque_nm=record['peak_torque_nm'],
                     rated_speed_rpm=record['rated_speed_rpm'], max_speed_rpm=record['max_speed_rpm'],
                     rotor_inertia_kg_m2=record['rotor_inertia_kg_m2'], source=record['source'])
