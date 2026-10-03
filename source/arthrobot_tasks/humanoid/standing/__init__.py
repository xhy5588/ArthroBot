"""Hybrid standing: keep the wheel-legged humanoid upright and still on its wheels.

The policy outputs 20 numbers each 1/60 s: 18 joint-target offsets (nominal pose
+ 0.25 rad x action, tracked by an explicit 80/4 PD controller recomputed every
physics step) and 2 wheel torques (13 N m x action). There is no balance
controller or root support: balance is learned with PPO (rsl_rl).

- :mod:`.env`         the environment and its configuration
- :mod:`.control`     the hybrid PD + wheel-torque law
- :mod:`.observation` the 96-number observation, shared with the get-up hand-over
- :mod:`.rewards`     dense standing reward terms
- :mod:`.policy`      loading the trained actor outside rsl_rl (hand-over tests)
- :mod:`.evaluation`  qualification criteria and held-out evaluation during training
"""
