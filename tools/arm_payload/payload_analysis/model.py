"""URDF forward kinematics and gravity for the ArthroBot arm.

This module keeps the original mass, gripper aggregation, TCP, and mounting
assumptions. It can be imported without Isaac Sim.
"""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .statics import GRAVITY_M_S2, gravity_torques, payload_intervals

from . import paths  # noqa: F401  (puts the checkout's packages on the path if needed)
from arthrobot_assets.arm import ARM_BODIES, ARM_JOINTS  # noqa: E402
# The arm reach task's mount: joint 1 vertical, base bottom at the origin, TCP between the jaws.
from arthrobot_tasks.arm_reach.mount import BASE_POSITION as ROOT_POS  # noqa: E402
from arthrobot_tasks.arm_reach.mount import MOUNT_ROTATION  # noqa: E402,F401
from arthrobot_tasks.arm_reach.mount import TCP_IN_TOOL as TCP_OFFSET  # noqa: E402


def origin(node):
    """URDF xyz/rpy origin as a homogeneous transform; distances are meters."""
    result = np.eye(4)
    if node is not None:
        result[:3, 3] = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
        result[:3, :3] = Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return result


class ArmGravity:
    """Six-joint ArthroBot arm gravity model, with downstream mass groups.

    The default mount rotates CAD joint 1 upright as arthrobot_tasks/arm_reach/mount.py does.
    CAD inertial masses and centers are retained. Jaw/pinion masses are folded
    into tool_output exactly once; its DM4310 stator is already present.
    Payload COM defaults to the existing between-jaw TCP. No collisions,
    dynamic inertia, gripper friction, or material strength are evaluated.
    """
    def __init__(self, urdf, mounting='vertical', jaw_displacement=0.):
        if mounting not in ('vertical', 'cad_viewer'):
            raise ValueError(f"mounting must be 'vertical' or 'cad_viewer', not {mounting!r}")
        root = ET.parse(urdf).getroot()
        self.names = list(ARM_JOINTS)
        joints = [root.find(f"joint[@name='{n}']") for n in self.names]
        links = [root.find(f"link[@name='{j.find('child').get('link')}']") for j in joints]
        parents = [ARM_BODIES[0]] + [l.get('name') for l in links[:-1]]  # the fixed base link
        assert [j.find('parent').get('link') for j in joints] == parents
        self.origins = np.array([origin(j.find('origin')) for j in joints])
        self.axes = np.array([np.fromstring(j.find('axis').get('xyz'), sep=' ') for j in joints])
        self.axes /= np.linalg.norm(self.axes, axis=1)[:, None]
        self.caps = np.array([float(j.find('limit').get('effort')) for j in joints])
        self.masses = np.array([float(l.find('inertial/mass').get('value')) for l in links])
        self.centers = np.array([origin(l.find('inertial/origin'))[:3, 3] for l in links])
        self.base_mass = float(root.find(f"link[@name='{ARM_BODIES[0]}']/inertial/mass").get('value'))
        # Combine jaw and pinion masses with tool_output; its DM4310 stator is already included.
        first_moment = self.masses[-1]*self.centers[-1]
        for joint in root.findall('joint'):
            if joint.find('parent').get('link') != 'tool_output':
                continue
            link = root.find(f"link[@name='{joint.find('child').get('link')}']")
            transform = origin(joint.find('origin'))
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            if joint.get('type') == 'prismatic':
                transform[:3, 3] += transform[:3, :3] @ (axis*jaw_displacement)
            else:
                transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(axis*jaw_displacement/.008).as_matrix()
            mass = float(link.find('inertial/mass').get('value'))
            center = origin(link.find('inertial/origin'))[:3, 3]
            first_moment += mass*(transform[:3, :3] @ center+transform[:3, 3])
            self.masses[-1] += mass
        self.centers[-1] = first_moment/self.masses[-1]
        self.mount = np.eye(4)
        if mounting == 'vertical':
            self.mount[:3, :3], self.mount[:3, 3] = MOUNT_ROTATION, ROOT_POS
        self.grasp = np.array(TCP_OFFSET)
        self.tool_direction = self.grasp/np.linalg.norm(self.grasp)

    def evaluate(self, q, offset=0.):
        """Evaluate poses in radians, returning the legacy six-array tuple.

        Returns own gravity torque (N*m), payload torque per kilogram
        (N*m/kg), own gravitational potential energy (J), payload COM (m),
        joint pivots (m), and body COMs (m). A batch dimension is always
        retained. Motor holding torque has the opposite sign to gravity.

        offset is a payload COM displacement in meters along tool_direction;
        it does not change the arm's own mass or shift the joint frames.
        """
        q = np.atleast_2d(q)
        n = len(q)
        transform = np.broadcast_to(self.mount, (n, 4, 4)).copy()
        pivots, axes, centers = [], [], []
        for i in range(6):
            transform = transform @ self.origins[i]
            pivots.append(transform[:, :3, 3].copy())
            axes.append(transform[:, :3, :3] @ self.axes[i])
            motion = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            motion[:, :3, :3] = Rotation.from_rotvec(q[:, i, None]*self.axes[i]).as_matrix()
            transform = transform @ motion
            centers.append(transform[:, :3, :3] @ self.centers[i] + transform[:, :3, 3])
        centers = np.stack(centers, axis=1)
        payload = transform[:, :3, :3] @ (self.grasp+offset*self.tool_direction) + transform[:, :3, 3]
        pivots = np.stack(pivots, axis=1)
        axes = np.stack(axes, axis=1)
        own, per_kg = gravity_torques(pivots, axes, centers, self.masses, payload)
        potential = GRAVITY_M_S2*(centers[:, :, 2]*self.masses).sum(1)
        return own, per_kg, potential, payload, pivots, centers

    def payload_interval(self, q, fraction=1., offset=0.):
        """Return full feasible mass intervals without collapsing impossible to zero.

        fraction scales every motor cap. With a nonzero lower bound, payload
        counterbalance is required: smaller masses do not necessarily pass.
        """
        own, unit, *_ = self.evaluate(q, offset)
        return payload_intervals(own, unit, self.caps*fraction)

    def capacity(self, q, fraction=1., offset=0.):
        """Legacy upper-bound API: return zero for an impossible interval.

        Kept for existing optimization scripts and report compatibility.
        New code should use payload_interval to distinguish impossible from
        feasible zero payload and preserve counterbalancing lower bounds.
        """
        interval = self.payload_interval(q, fraction, offset)
        return np.where(interval.feasible, interval.upper_kg, 0.)

    def check_energy_gradient(self):
        """Cross-check signed torque against minus the energy gradient."""
        for q in np.random.default_rng(12).uniform(-np.pi, np.pi, (8, 6)):
            own, unit, *_ = self.evaluate(q)
            for i in range(6):
                delta = np.eye(6)[i]*1e-6
                plus, minus = self.evaluate(q+delta), self.evaluate(q-delta)
                np.testing.assert_allclose(own[0, i], -(plus[2][0]-minus[2][0])/2e-6, atol=2e-7)
                np.testing.assert_allclose(unit[0, i], -9.81*(plus[3][0, 2]-minus[3][0, 2])/2e-6, atol=2e-7)

    def describe(self, q, fraction=1., offset=0.):
        """Legacy report fields; use payload_interval for explicit feasibility."""
        own, unit, _, payload, pivots, _ = self.evaluate(q, offset)
        mass = float(self.capacity(q, fraction, offset)[0])
        total = own[0]+mass*unit[0]
        limiting = int(np.argmax(np.abs(total)/(self.caps*fraction)))
        return dict(payload_kg=mass, limiting_joint=self.names[limiting],
                    joint_angles_deg=np.rad2deg(q).tolist(), own_gravity_torque_nm=own[0].tolist(),
                    payload_torque_per_kg_nm=unit[0].tolist(), total_gravity_torque_nm=total.tolist(),
                    empty_arm_feasible=bool((np.abs(own[0]) <= self.caps*fraction).all()),
                    limiting_effective_horizontal_lever_m=abs(float(unit[0, limiting]))/9.81,
                    shoulder_to_payload_horizontal_m=float(np.linalg.norm((payload[0]-pivots[0, 0])[:2])),
                    payload_center_world_m=payload[0].tolist(), joint_origins_world_m=pivots[0].tolist(),
                    torque_fraction=fraction, payload_offset_m=offset)

