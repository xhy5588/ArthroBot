"""Training mount: joint 1 vertical (base yaw), base bottom at the environment origin.

The CAD lies on its side: joint 1 points along CAD +X and the zero pose reaches
along CAD +Y. On the real robot joint 1 is a vertical base yaw, so training
rotates the fixed base so that CAD +X is up and CAD +Y is forward. Targets and
observations use this mount frame: origin at the bottom of the base on the
joint 1 axis, x forward, z up. Pure NumPy, so tests import it without Isaac Sim.
"""
import numpy as np

# world = MOUNT_ROTATION @ cad: CAD X -> world Z, CAD Y -> world X, CAD Z -> world Y.
MOUNT_ROTATION = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
MOUNT_QUAT_WXYZ = (0.5, -0.5, -0.5, -0.5)
# Joint 1 axis at the lowest point of the base bracket, in the arm's base frame (CAD).
BASE_ANCHOR_IN_BASE = (0.0341, 0.38989, -0.01573)
# Base pose that puts BASE_ANCHOR_IN_BASE at the environment origin.
BASE_POSITION = tuple(float(value) for value in -MOUNT_ROTATION @ np.array(BASE_ANCHOR_IN_BASE))
# Tool center point (TCP): the grasp center between the jaw clips, in the tool frame
# (build report: grasp_center_in_tool_m).
TCP_IN_TOOL = (0.1669941280872268, 6.057105226678003e-05, -2.553963974384621e-05)
# Training-only joint range. The real cable limits are not yet recorded.
JOINT_LIMIT_DEG = 180.
