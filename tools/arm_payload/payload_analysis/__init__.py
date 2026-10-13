"""Reusable static payload calculations for the standalone ARM_REBOT.

Distances are meters, mass kilograms, angles radians, and torque N*m.
Use ArmGravity.payload_interval for new analysis; its feasible flag and lower
bound retain information omitted by the older capacity-only interface.
"""
from .model import ArmGravity, origin
from .statics import (
    GRAVITY_M_S2,
    UNIT_TORQUE_TOLERANCE,
    PayloadInterval,
    common_maximum,
    gravity_torques,
    payload_intervals,
    torque_for_payload,
)

__all__ = [
    'ArmGravity', 'origin', 'GRAVITY_M_S2', 'UNIT_TORQUE_TOLERANCE',
    'PayloadInterval', 'common_maximum', 'gravity_torques',
    'payload_intervals', 'torque_for_payload',
]
