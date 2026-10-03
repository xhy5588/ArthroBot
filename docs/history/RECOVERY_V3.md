# Protected standing-to-recovery experiment

This starts from the original successful standing actor at
`whole_body_runs/20260923_175447/model_4999.pt`, not the collapsed v2 policy.
The frozen teacher is used in a gated training loss only; it never supplies the
robot's actions during student evaluation or rollouts. The learned policy still
controls all 20 motors, with grippers open, PD 80/4, 13 N m torque limits and the
original ±0.25-radian position-target scale.

## Ordered experiment

1. Run the original actor for a full 30-second evaluation window with 32 upright
   and 32 mild-tilt trials (eight each front/back/left/right). Disable early success
   termination. Standing qualification uses the original thresholds: duration at
   least 29.95 s, max tilt 20 degrees, min torso height .46 m, mean speed .15 m/s,
   max axis drift .25 m, both-wheel support fraction .8, no invalid/escaped state.
   Also record longest uninterrupted strict hold and 10-second recovery success.
2. If fewer than 90% of standing trials qualify, stop the experiment before training.
3. Run two separate 100-update trials from the identical original actor, with the
   same seed, banks, fixed initial difficulty, reward and teacher coefficient.
   One uses original physical exploration; the other halves both joint and wheel
   standard deviations. Standard deviations remain fixed during each trial.
4. Qualify trials at >=90% standing success, finite losses, and <10% action clipping
   averaged over the last 50 updates. Rank qualified trials by deterministic mild
   recovery success, then noisy valid-balance fraction, then lower clipping/noise.
   These are provisional single-seed comparisons, not statistical proof.
5. Resume the winning trial for at most 900 more updates (1000 total), evaluating
   every 100. Preserve a best checkpoint. Two consecutive evaluations below 90%
   standing success restore the best checkpoint and stop. If neither trial passes,
   stop and report the failure rather than launching a long run.

The sequential controller is `run_recovery_v3_experiment.py`. It records status,
comparison and selection JSON files in the experiment directory. Nonzero process
exit or a user stop prevents it from moving on to another run. Runtime errors stop
the sequence; they do not silently resume training.

## Teacher loss and observations

The first 96 student observation inputs and all 20 action units match the original
standing actor. New observation columns start with zero actor weights, preserving
its mean actions exactly at initialization. The student has 120 inputs: recovery
contacts, hold/time fields, and target hold divided by 10; it has no difficulty-level
observation that would extrapolate at evaluation.

Teacher matching applies only when observed torso tilt is below 12 degrees,
height is .46–.65 m, body horizontal speed is below .2 m/s, angular speed below
.6 rad/s, and both wheels support the robot. The loss is 5 times mean squared
normalized action-mean error over eligible samples/actions. An actor output
gradient hook adds the exact derivative before PPO gradient clipping. Tests compare
it against direct autograd of the same gated loss. Teacher weights are frozen.

The critic and optimizer are initialized fresh. PPO uses 48 steps/env, fixed
actor LR 3e-6, critic LR 3e-4 and gamma .999. The v2 task reward is unchanged,
including terminal-aware potential shaping, +100 goal completion, -25 failure,
time/motion costs and finite-horizon timeout handling. The imitation loss is
separate from environment reward and is logged as retention loss.

## Training mixture and curriculum

With 64 robots, 32 always rehearse upright standing with a 10-second target.
The remaining 32 are assigned permanently to four recovery directions.
The initial lean is approximately 5–10 degrees with pose jitter, with a 0.5-second
recovery hold goal. Trials freeze this difficulty to compare exploration fairly.
During the subsequent guarded run, each direction advances independently after
28 successes in 40 current-stage episodes, demotes below 8/40, and samples older
levels on 25% of resets. Generation IDs prevent stale statistics from triggering
another stage change. Hold duration and lean difficulty change separately:

| Stage | Lean range (degrees) | Hold (seconds) |
|---|---|---|
| 0 | 5–10 | .5 |
| 1 | 5–10 | 1 |
| 2 | 10–15 | 1 |
| 3 | 10–15 | 2 |
| 4 | 15–25 | 2 |
| 5 | 25–40 | 2 |
| 6 | 40–60 | 2 |
| 7 | 60–80 | 2 |
| 8 | 80–95 | 2 |
| 9 | 80–95 | 5 |
| 10 | 80–95 | 10 |

The action range is not automatically enlarged. Failure at larger tilts may require
a separate measured action-range experiment. Fully lying recovery has not been
demonstrated. The fixed baseline evaluation deliberately measures this experiment's
first milestone (standing retention plus mild recovery), not full lying recovery.
Geometry is floor-placed; no new physical recoverability guarantee is assumed.

## Charts

- `Evaluation/standing_success_fraction`: full 30-second standing qualification.
- `Evaluation/recovery_success_fraction`: 10-second strict hold from fixed mild tilts.
- `Evaluation/recovery_best_hold_mean_s`: early recovery progress.
- `Recovery/retention_loss`, `Recovery/retention_eligible_fraction`.
- Per-direction curriculum stage/success and hold-break diagnostics.
- `Recovery/action_clip_fraction`, post-update KL and torque saturation.

Teacher-free deterministic evaluation is essential: training with imitation does
not establish that the student retained the skill. All three original baselines
remain saved, and v3 checkpoints use a separate latest pointer.
