"""ArthroBot learning tasks for Isaac Lab.

- :mod:`arthrobot_tasks.arm_reach` registers ``ArthroBot-Arm-Reach-v0`` (and ``-Play-v0``):
  move the arm's grasp center to random 3D targets (manager-based env, rsl_rl PPO).
- :mod:`arthrobot_tasks.humanoid` holds the humanoid's standing and get-up tasks
  (direct envs with their own training scripts in ``scripts/humanoid/``).

Importing this package registers the gym tasks; it imports no Isaac modules itself.
"""
from arthrobot_tasks import arm_reach  # noqa: F401
