"""Pure signed gravity and static-payload calculations, in SI units.

At a fixed pose the gravity torque is a + m*b, where a is the arm's own
gravity torque, b is the torque produced by one kilogram at the payload COM,
and m >= 0 is payload mass. A motor holds with the opposite torque; symmetric
motor limits therefore require abs(a + m*b) <= cap for every joint.

These functions screen motor torque only. They do not establish reachability,
collision clearance, paths, grip, strength, thermal limits, or dynamic lifting.
"""
from typing import NamedTuple

import numpy as np

GRAVITY_M_S2 = 9.81
UNIT_TORQUE_TOLERANCE = 1e-10


class PayloadInterval(NamedTuple):
    """Intersection of all joint inequalities with m >= 0, for each pose.

    lower_kg and upper_kg have the broadcast input shape without its final
    joint axis. feasible identifies a nonempty interval. If feasible is False,
    neither endpoint is a capacity: the empty arm and every payload may fail.
    A feasible [0, 0] means only zero payload passes. A positive lower bound
    means a load is needed to counterbalance the arm. An infinite upper bound
    means this gravity-only model supplies no finite torque bound.
    """

    lower_kg: np.ndarray
    upper_kg: np.ndarray
    feasible: np.ndarray


def gravity_torques(joint_origins_m, joint_axes_world, body_centers_m,
                    body_masses_kg, payload_center_m,
                    gravity_vector_m_s2=(0., 0., -GRAVITY_M_S2)):
    """Return signed own-arm torque and payload torque per kilogram.

    Geometry arrays have shape (..., joints, 3); the payload COM has shape
    (..., 3), and masses have shape (joints,). Axes must be unit vectors.
    Each body group i is downstream of joint i, so joint j supports groups
    j through the end. Fixed-base mass is not one of these moving groups.

    Torque about a joint is axis dot sum((COM - pivot) cross mass*gravity).
    The cross product retains the sign: payload gravity can reinforce or
    oppose the arm's gravity torque. The second returned array uses 1 kg
    at the supplied payload COM, not necessarily at the gripper TCP.
    """
    pivots, axes, centers = np.broadcast_arrays(
        np.asarray(joint_origins_m, dtype=float),
        np.asarray(joint_axes_world, dtype=float),
        np.asarray(body_centers_m, dtype=float),
    )
    if pivots.ndim < 2 or pivots.shape[-1] != 3 or not pivots.shape[-2]:
        raise ValueError('Joint geometry must have shape (..., joints, 3).')
    count = pivots.shape[-2]
    masses = np.asarray(body_masses_kg, dtype=float)
    gravity = np.asarray(gravity_vector_m_s2, dtype=float)
    if masses.shape != (count,) or np.any(masses < 0) or not np.isfinite(masses).all():
        raise ValueError('Masses must be a finite nonnegative vector, one per joint.')
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError('Gravity must be a finite three-component acceleration.')
    payload = np.broadcast_to(np.asarray(payload_center_m, dtype=float),
                              pivots.shape[:-2] + (3,))
    own = np.zeros(pivots.shape[:-1])
    per_kg = np.zeros_like(own)
    for joint in range(count):
        forces = masses[joint:, None]*gravity
        moments = np.cross(centers[..., joint:, :] - pivots[..., joint, None, :], forces)
        own[..., joint] = (moments.sum(axis=-2)*axes[..., joint, :]).sum(axis=-1)
        unit_moment = np.cross(payload-pivots[..., joint, :], gravity)
        per_kg[..., joint] = (unit_moment*axes[..., joint, :]).sum(axis=-1)
    return own, per_kg


def torque_for_payload(own_torque_nm, payload_torque_per_kg_nm, payload_kg):
    """Evaluate signed a + m*b; the final output axis is the joint axis.

    Scalar mass evaluates every input pose at that mass. An array of masses
    must broadcast against the leading pose dimensions; for a load sweep,
    explicitly add dimensions to coefficients/masses as needed.
    """
    own, unit = np.broadcast_arrays(np.atleast_1d(np.asarray(own_torque_nm, dtype=float)),
                                    np.atleast_1d(np.asarray(payload_torque_per_kg_nm, dtype=float)))
    mass = np.asarray(payload_kg, dtype=float)
    if not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError('Payload mass must be finite and nonnegative.')
    return own + mass[..., None]*unit


def payload_intervals(own_torque_nm, payload_torque_per_kg_nm, torque_limits_nm,
                      unit_tolerance=UNIT_TORQUE_TOLERANCE):
    """Solve every signed inequality abs(a_j + m*b_j) <= cap_j for m >= 0.

    Inputs broadcast to (..., joints). For nonzero b, the joint interval is
    the sorted pair ((-cap-a)/b, (cap-a)/b). Intersect these intervals across
    joints and with nonnegative mass. For effectively zero b, an over-limit
    a makes the pose impossible; otherwise that joint imposes no mass bound.

    The 1e-10 N*m/kg threshold preserves the original calculator's numerical
    treatment of almost-vertical moment arms. It is not a safety margin.
    No absolute values are taken before solving the signed inequalities.
    """
    if not np.isfinite(unit_tolerance) or unit_tolerance < 0:
        raise ValueError('Unit-torque tolerance must be finite and nonnegative.')
    own, unit, caps = np.broadcast_arrays(
        np.atleast_1d(np.asarray(own_torque_nm, dtype=float)),
        np.atleast_1d(np.asarray(payload_torque_per_kg_nm, dtype=float)),
        np.atleast_1d(np.asarray(torque_limits_nm, dtype=float)),
    )
    if not own.shape[-1]:
        raise ValueError('At least one joint is required.')
    if not all(np.isfinite(value).all() for value in (own, unit, caps)) or np.any(caps < 0):
        raise ValueError('Torques must be finite and torque limits nonnegative.')
    active = np.abs(unit) > unit_tolerance
    divisor = np.where(active, unit, 1.)
    first, second = (-caps-own)/divisor, (caps-own)/divisor
    lower = np.maximum(0., np.where(active, np.minimum(first, second), -np.inf).max(axis=-1))
    upper = np.where(active, np.maximum(first, second), np.inf).min(axis=-1)
    feasible = (upper >= lower) & ~((~active) & (np.abs(own) > caps)).any(axis=-1)
    return PayloadInterval(lower, upper, feasible)


def common_maximum(lower_kg, upper_kg, feasible, tolerance=1e-9):
    """Largest common mass across target-wise unions of IK-branch intervals.

    Arrays have shape (branches, targets). Each target may choose a different
    IK branch, so feasible intervals are UNIONED across branches, then
    INTERSECTED across targets. Retaining each lower bound matters: a branch
    may need payload counterbalance, and the union may contain gaps.

    Return None when no common nonnegative mass exists (including zero),
    0.0 when only zero works, or infinity if all targets are unbounded in
    this ideal torque model. This does not validate unsampled targets.
    """
    lo, hi = np.asarray(lower_kg, dtype=float), np.asarray(upper_kg, dtype=float)
    valid = np.asarray(feasible, dtype=bool)
    if lo.ndim != 2 or lo.shape != hi.shape or lo.shape != valid.shape or 0 in lo.shape:
        raise ValueError('Intervals must have matching nonempty (branches, targets) shapes.')
    if np.isnan(lo).any() or np.isnan(hi).any() or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Intervals cannot contain NaN; tolerance must be finite and nonnegative.')
    if not valid.any(axis=0).all():
        return None
    ceiling = np.where(valid, hi, -np.inf).max(axis=0).min()
    for _ in range(lo.size+1):
        good = (valid & (lo <= ceiling+tolerance) & (hi >= ceiling-tolerance)).any(axis=0)
        if good.all():
            return float(ceiling)
        alternatives = np.where(valid[:, ~good] & (hi[:, ~good] < ceiling-tolerance),
                                hi[:, ~good], -np.inf)
        ceiling = alternatives.max(axis=0).min()
        if not np.isfinite(ceiling) or ceiling < 0:
            return None
    raise AssertionError('Interval intersection did not terminate')
