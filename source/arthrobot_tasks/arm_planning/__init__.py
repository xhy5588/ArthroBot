"""Arm motion: kinematics, IK, collision-free planning with cuRobo, and a loose-arm test model.

- :mod:`.kinematics`: torch FK, Jacobians, gravity torque, position and gripper-pose IK (no Isaac Sim).
- :mod:`.controllers`: IK with minimum-jerk joint paths and gravity compensation (no obstacle awareness).
- :mod:`.curobo_robot`: the arm's model for NVIDIA cuRobo (planning URDF, collision spheres).
- :mod:`.obstacles`, :mod:`.targets`: the table-and-wall test scene and its grasp targets.
- :mod:`.loose`, :mod:`.scene`: the arm with joint play and bracket flex, as a 3D-printed arm will have.

Scripts: ``scripts/arm/build_curobo_robot.py``, ``check_curobo_model.py``, ``plan_trajectories.py``,
``run_plans.py``, ``check_loose_arm.py``. Results and the reasoning behind them: ``docs/arm_planning.md``.
"""
