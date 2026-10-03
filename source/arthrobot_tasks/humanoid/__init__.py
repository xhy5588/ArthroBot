"""Humanoid learning tasks (Isaac Lab direct environments).

- :mod:`.assets`   training USD (21 bodies, grippers parked) and the light get-up variant
- :mod:`.standing` hybrid standing: 18 joint-target offsets + 2 wheel torques, PPO (rsl_rl)
- :mod:`.getup`    HoST-style get-up from lying poses, multi-critic PPO

Both tasks command the same 20 motors in the same order (``MOTOR_JOINTS``):
12 arm joints, 6 leg joints, 2 wheels. Every motor is capped at 13 N m and 74 rpm.
"""
