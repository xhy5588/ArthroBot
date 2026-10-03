# Randomized get-up and balance experiment

The revised objective and independent directional curriculum are documented in
[RECOVERY_V2.md](RECOVERY_V2.md). This document describes the preserved v1 baseline.

Run locally without the Isaac Sim window:

```bash
bash onshape/modular_humanoid_bipad/training/run_recovery.sh
```

Launcher default: 64 parallel environments, 10,000 PPO updates, 48 transitions per
environment per update (30.72 million transitions for a fresh run), without rendering.
Every environment simulates and contributes to learning. To use the previously
tested live setup, launch with `HEADLESS=0 NUM_ENVS=16 bash onshape/modular_humanoid_bipad/training/run_recovery.sh`.
This is a new experiment; recovery from lying has **not** been demonstrated yet.

The successful standing run `whole_body_runs/20260923_175447/model_4999.pt` stays
unchanged. We copy its actor into a new network, initialize the critic and optimizer
fresh, and expand the observation from 96 to 119 values (21 floor-contact flags,
balance-hold fraction, time remaining). New input weights start at zero. Rescaling
the first and last actor layers preserves its physical mean joint targets despite
the wider action range. The first 18 previous-action input columns also change
units. Exploration noise is rescaled with a minimum standard deviation of 0.04
in the new units; this is deliberately more exploration than the final standing run.

All 20 motors remain learned: 18 position offsets with explicit 80/4 PD at 240 Hz
and two wheel torques. Joint target range is nominal ±1.5 radians; torque remains
limited to ±13 Nm, speed to 74 rpm. Grippers remain parked open. No external root
support, scripted get-up trajectory, or balance controller is added.

## Starts and curriculum

Each reset samples 20% upright, 60% tilted, and 20% lying (front/back/left/right
equally). The tilted curriculum ranges are 15–25°, 25–45°, 45–65°, and 65–85°;
these are sampled roll/pitch components along a random direction, so actual torso
tilt is approximate. Joint angles and yaw are also randomized. Poses are selected
from a reproducible finite bank of 216 configurations; evaluation uses a separate
36-configuration bank with seed 1043. These are geometry-valid floor placements,
not a proof that every configuration is dynamically recoverable. Self-intersection
and recoverability of all poses have not been exhaustively certified.

Mesh vertices determine each root height so the lowest collision geometry begins
4 mm above the floor. This avoids starting lying robots buried in the ground.
The curriculum advances only after at least 28 successes in the last 40 tilted
trials at its current level and 16 successes in the last 20 upright trials, within
the recent 200 episodes. Lying trials occur from the start, even at difficulty 0.

## Success and reward

Episodes last at most 45 seconds. Low torso height, large tilt, and non-wheel floor
contact are allowed during recovery. Only success, invalid states, moving over
3 m from the starting COM, a root below −0.2 m, or timeout ends the episode.

Success requires **10 consecutive seconds** with:

- Torso tilt below 15° and torso-frame origin height between 0.46 and 0.65 m.
- Horizontal COM speed below 0.15 m/s and torso angular speed below 0.5 rad/s.
- Both wheels supporting over 3 N; each other body under 10 N of floor force.
- COM staying within 0.15 m of where this uninterrupted balance hold began.

Breaking any condition resets the hold timer. A timeout is a failure, even if
the robot briefly stood. Upright starts measure balance retention; lying starts
measure recovery. These outcomes are logged separately.

`recovery_math.py:reward_terms` is the exact reward implementation. Per-second
terms are 1.5 times upright progress, 2 times normalized height progress, 10 for
meeting the balance conditions, up to 2 for wheel support near standing, a
near-standing velocity penalty, −0.0003 times summed squared torque, and −0.05
times summed squared target change (joint radians, normalized wheel actions).
Multiply by control dt; add 20 once on success or subtract 10 for invalid/escaped
termination. There is no penalty simply for supporting the body on arms during
get-up. PPO entropy coefficient is 0.005. The actor uses a fixed learning rate
of 3e-6 and the fresh critic 3e-4. An initial smoke test with the standing run's
adaptive 3e-4 rate caused post-update KL of 26.6 and 100% probability-ratio clipping
on the first update, so that setup was rejected before the long run. Wider action
units require smaller actor output weights and more careful optimization. Other
baseline PPO parameters are kept. The standing model itself is unchanged.

## Evidence and limitations

`recovery_runs/<timestamp>/settings.json` records settings and source hashes;
`source/` snapshots code. `metrics.csv`, TensorBoard, and `status.json` include
total/value/policy loss, entropy, KL, clipping, torque saturation, curriculum and
per-start success rates. Missing episode returns are blank/null, never fake zeros.
`recent_episodes.json` stores recovery time, best balance hold and termination.
New continuations also record passive hold diagnostics (version 1 in settings).
The reward, success thresholds, action noise and curriculum rules are unchanged.
`hold_diagnostics.json` contains the latest PPO rollout's counts, split by starting
family; each completed episode has its own `hold_diagnostics` in the episode JSON.
TensorBoard/CSV charts include:

- `Recovery/hold_all_breaks`: number of active holds interrupted during that rollout.
- `Recovery/hold_all_<condition>_break_fraction`: fraction of those interruptions
  at which that gate failed. Conditions are `tilt`, `height`, `linear_speed`,
  `angular_speed`, `left_wheel_contact`, `right_wheel_contact`, `body_contact`, `drift`.
- `Recovery/hold_all_<condition>_failure_fraction`: fraction of all robot control
  steps failing that gate, including robots that have never begun a hold. Drift
  is evaluated only against an active hold's anchor; no active hold means no drift
  failure. Use the other gates to diagnose inability to start standing.
- Replace `all` with `upright`, `tilted`, `front`, `back`, `left` or `right` to
  compare starting poses. `_balanced_fraction` shows valid standing time fraction.

Several gates can fail together; break fractions can sum above 100%. These report
coincident violations, not proof of physical root cause. With no broken holds the
break fractions are null/blank and not added to TensorBoard, rather than misleading
zeros. They are interval statistics, not a cumulative count across the run. An
episode timeout or reset alone is not a balance interruption. Old checkpoints/logs
do not contain this information and cannot reconstruct it retroactively.
Evaluations log the first trial per robot only, separately under
`Evaluation/<family>/success_fraction` and `Evaluation/<family>/hold_*`, and preserve
training diagnostic counters. JSON evaluation reports retain the same breakdown.

Scheduled evaluation every 1,000 updates uses the held-out bank and the first
completed trial per environment. Sixty-four environments mean ten or eleven trials per
family; this is an early indicator, not a statistically strong qualification.

The first recovery run used 12 environments and was checkpointed after 102 updates
to test 16. The continuation retains the policy, optimizer and curriculum, with
9,898 additional updates planned to reach 10,000 overall. The full history spans
both run directories; the continuation's transition/time counters start at zero.
Changing environment count increases the rollout batch from 576 to 768 transitions.
On this 32 GiB workstation, system RAM is tight despite spare GPU memory; use
measured transitions per second and available host RAM to choose environment count.

After the [32/64-robot benchmark](RECOVERY_SCALING.md), the user selected 64 robots
without rendering. The main run resumes its latest checkpoint, including optimizer
and curriculum, and keeps the 10,000 total-update target. Each new update now uses
3,072 transitions. The 16-robot RAM constraint above refers to the live-window setup;
64 robots fit comfortably in the tested headless setup. The benchmark checkpoints
are separate and are not used as the continuation source.

Checkpoints are saved atomically every 25 updates, with a separate
`latest_recovery.txt`. Resume with `recovery.py --mode train --checkpoint PATH`;
this restores policy, optimizer, update index and curriculum level, but starts
new physical episodes and does not restore simulator/RNG state or rolling episode
statistics. Saved iterations are zero-indexed. SIGINT/SIGTERM and the GUI stop
button request a checkpoint at the next completed PPO update.

Smoke check (headless, six start families):

```bash
/home/eric/env_isaaclab/bin/python onshape/modular_humanoid_bipad/training/recovery.py --mode check --num-envs 6 --headless
/home/eric/env_isaaclab/bin/python onshape/modular_humanoid_bipad/training/test_recovery.py
```

This curriculum is inspired by staged get-up training in
[HoST](https://arxiv.org/abs/2502.08378) and
[HUMANUP](https://arxiv.org/abs/2502.12152). It does not reproduce those systems:
this wheeled robot has different geometry, actuator strength and available support
contacts. Some lying poses may require changes to mechanics, action reach or the
curriculum. Falling less, rising higher, longer holds and measured recovery success
matter more than whether PPO's total loss declines monotonically.
