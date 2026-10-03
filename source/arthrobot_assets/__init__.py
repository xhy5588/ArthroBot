"""ArthroBot robot descriptions, built from their Onshape CAD exports.

Each robot package holds its untouched CAD export under ``cad/`` and a builder
that turns it into a simulation-ready URDF (written to ``build/<robot>/``):

- :mod:`arthrobot_assets.module`   one joint module, as exported
- :mod:`arthrobot_assets.arm`      6-DOF arm with the DM4310 gripper
- :mod:`arthrobot_assets.humanoid` wheel-legged humanoid with two arms and two grippers
"""
