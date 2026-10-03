"""Get-up: stand up from lying on the floor (adapted from HoST, Huang et al. 2025, arXiv:2502.08378).

- :mod:`.env`          the environment (lying start poses, relative joint targets, 5 reward groups)
- :mod:`.ppo`          multi-critic PPO with one value head per reward group
- :mod:`.evaluation`   held-out evaluation and its summary metrics
- :mod:`.poses`        banks of collision-free lying (and upright) start poses
- :mod:`.joint_limits` self-collision joint limits from convex-hull sweeps
- ``data/``            the joint limits and pose banks the included policy was trained with
"""
