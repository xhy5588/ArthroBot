"""ArthroBot core library: shared robot-building blocks for every morphology.

Pure Python (NumPy/SciPy/trimesh) except :mod:`arthrobot.sim`, which needs a
running Isaac Sim application.

- :mod:`arthrobot.paths`  repository locations (source, build output, logs)
- :mod:`arthrobot.urdf`   URDF transforms, forward kinematics and file helpers
- :mod:`arthrobot.mass`   mass properties of meshes and rigid bodies
- :mod:`arthrobot.motors` joint-module motor specifications
- :mod:`arthrobot.parts`  standard structural parts (holders, rods, connectors)
- :mod:`arthrobot.gripper` the DM4310 rack-and-pinion gripper model
"""

__version__ = '0.1.0'
