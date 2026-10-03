"""DM4310 rack-and-pinion parallel gripper (Seeed Studio reBot B601-DM design).

One motor turns a 16-tooth, module-1 pinion that drives two opposed racks, so
both jaws close by ``x = pitch_radius * theta``. Simulation models this as one
driven jaw coordinate; the second jaw and the pinion follow through mimic
constraints, so both jaws share a single motor budget:

    jaw force limit      F = torque / r          (375 N at 3 N m)
    equal-load per jaw   F = torque / (2 r)      (187.5 N at 3 N m)
    linear drive gains   K_linear = K_angular / r^2,  D_linear = D_angular / r^2

Specifications and their limitations are recorded in ``data/dm4310_gripper.json``.
"""
from dataclasses import dataclass
import json
import math

from arthrobot.paths import DATA_DIR

# Physics step at which the gripper contact model was validated (960 Hz, PGS 128/128).
VALIDATED_PHYSICS_DT = 1 / 960


@dataclass(frozen=True)
class RackPinionGripper:
    motor_model: str
    motor_mass_kg: float
    mechanism_mass_estimate_kg: float
    rated_torque_nm: float
    max_speed_rpm: float
    pinion_radius_m: float
    jaw_stroke_m: float
    drive_stiffness_nm_per_rad: float
    drive_damping_nm_s_per_rad: float
    mimic_natural_frequency: float
    mimic_damping_ratio: float
    target_jaw_speed_m_s: float
    record: dict

    @property
    def max_speed_rad_s(self) -> float:
        return self.max_speed_rpm * math.pi / 30

    @property
    def jaw_force_limit_n(self) -> float:
        """Shared force cap of the driven jaw coordinate: torque / pitch radius."""
        return self.rated_torque_nm / self.pinion_radius_m

    @property
    def equal_load_force_per_jaw_n(self) -> float:
        return self.rated_torque_nm / (2 * self.pinion_radius_m)

    @property
    def full_stroke_pinion_angle_rad(self) -> float:
        return self.jaw_stroke_m / self.pinion_radius_m

    @property
    def linear_stiffness_n_per_m(self) -> float:
        return self.drive_stiffness_nm_per_rad / self.pinion_radius_m ** 2

    @property
    def linear_damping_n_s_per_m(self) -> float:
        return self.drive_damping_nm_s_per_rad / self.pinion_radius_m ** 2

    @property
    def total_mass_estimate_kg(self) -> float:
        return self.motor_mass_kg + self.mechanism_mass_estimate_kg

    def summary(self) -> dict:
        return {**self.record, 'pinion_pitch_radius_m': self.pinion_radius_m,
                'output_angle_for_full_stroke_rad': self.full_stroke_pinion_angle_rad,
                'ideal_equal_load_force_per_jaw_n': self.equal_load_force_per_jaw_n,
                'active_torque_limit_nm': self.rated_torque_nm,
                'physics_dt_s': VALIDATED_PHYSICS_DT,
                'active_rack_force_limit_n': self.jaw_force_limit_n,
                'transmission': 'One equivalent rack actuator F=tau/r; jaw 1 = jaw 0 and theta = jaw 0 / r. '
                                'No independent second-jaw drive.'}


def load_dm4310_gripper() -> RackPinionGripper:
    record = json.loads((DATA_DIR / 'dm4310_gripper.json').read_text())
    return RackPinionGripper(
        motor_model=record['model'], motor_mass_kg=record['motor_mass_kg'],
        mechanism_mass_estimate_kg=record['mechanism_mass_estimate_kg'],
        rated_torque_nm=record['rated_output_torque_nm'],
        max_speed_rpm=record['maximum_no_load_output_speed_rpm'],
        pinion_radius_m=record['pinion_module_mm'] * record['pinion_teeth'] / 2000.,
        jaw_stroke_m=record['jaw_travel_m'],
        drive_stiffness_nm_per_rad=record['drive_stiffness_nm_per_rad'],
        drive_damping_nm_s_per_rad=record['drive_damping_nm_s_per_rad'],
        mimic_natural_frequency=record['mimic_natural_frequency'],
        mimic_damping_ratio=record['mimic_damping_ratio'],
        target_jaw_speed_m_s=record['target_jaw_speed_m_s'],
        record=record)


DM4310_GRIPPER = load_dm4310_gripper()
