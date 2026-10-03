# Recovery v2: sustained balance and directional recovery

This experiment changes the objective; rewards/losses are not comparable to v1.
The latest v1 actor and exploration standard deviation are transferred into a
121-input actor. Its two new goal inputs have zero initial weights. The critic,
optimizer and update counter start fresh. Mechanical settings, 20 actions, gains,
torque limits, grippers and 64 headless environments are unchanged.

## Objective

At 60 Hz, reward is:

```
gamma * Phi(next) - Phi(current)
- dt * (0.25 + 0.0001 * sum(torque**2)
        + 0.05 * sum(physical_action_change**2)
        + 0.5 * upright_quality * sum(torso_angular_velocity**2))
+ 100 if the continuous hold goal is completed
- 25 if the episode ends without completing it
```

Gamma is 0.999 in both PPO and shaping. Phi contains bounded upright, height,
near-standing and squared uninterrupted-hold progress, with maximum 40. Phi at
**every terminal state is zero**, including the 45-second timeout. Timeouts are
true finite-horizon terminals and do not receive RSL-RL timeout bootstrapping.
The remaining time and hold goal are observed. Discounted shaping telescopes to
minus the initial potential for a completed episode: repeated partial-pose loops
cannot create additional discounted shaping return. Logged undiscounted returns
can still differ; success rates remain the primary metric. Effort/regularization
costs can affect trajectory preference; tests do not prove optimal behavior or
physical recoverability. No repeatable standing or wheel-support bonus remains.

## Curriculum and evaluation

Each of 64 environments keeps an assigned start family, preventing shorter easy
episodes from replacing difficult starts: 8 upright, 8 general tilt, and 12 each
for front/back/left/right. For each family independently, 28 successes among the
last 40 episodes at its current stage advance that family one stage.

| Stage | Directional initial lean | Required continuous hold |
|---|---|---|
| 0 | 15–30 degrees | 0.5 seconds |
| 1 | 30–50 degrees | 1 second |
| 2 | 50–70 degrees | 2 seconds |
| 3 | 80–95 degrees | 2 seconds |
| 4 | 85–95 degrees | 5 seconds |
| 5 | 85–95 degrees | 10 seconds |

Small pose jitter is applied; geometry is floor-placed for every sampled pose.
Upright starts remain upright as the hold goal increases. General tilted starts
use random roll/pitch directions. Front/back/left/right starts lean along their
specified direction, approaching fully lying poses at later stages. Early-stage
successes are **curriculum successes**, not proof of recovery from lying.

Evaluation always uses the original fixed held-out geometry: upright, level-0
tilted, and truly lying front/back/left/right poses. Every evaluation requires 10
continuous seconds under the original tilt, height, velocity, support and drift
thresholds, with exploration disabled. This benchmark does not get easier with
the curriculum. The training pose bank is not certified dynamically recoverable.

## Logs and launching

Use `Recovery/<family>_curriculum_success_fraction` and `_curriculum_level` to
follow training. `Recovery/<family>_success_fraction` remains strict 10-second
success. `Evaluation/<family>/success_fraction` is the fixed benchmark. Per-family
episode return and each reward component, target hold, strict success and all hold
interruption diagnostics are recorded. Do not judge success by aggregate reward.

```
/home/eric/env_isaaclab/bin/python -u onshape/modular_humanoid_bipad/training/recovery_v2.py \
  --mode train --num-envs 64 --headless --iterations 10000 --eval-interval 500 \
  --checkpoint /absolute/path/to/model.pt
```

A v1 checkpoint transfers only the actor and starts a new v2 experiment. A v2
checkpoint resumes actor, critic, optimizer, family levels and update count; the
iteration argument then means additional updates. Physical episodes/RNG are
restarted on resume, as in v1. `--benchmark --iterations 8 --eval-interval 0`
isolates a smoke run without changing active/latest pointers. Checkpoints and
source snapshots preserve both experiments.
