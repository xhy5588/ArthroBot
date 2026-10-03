# Phased recovery after standing

The v3 run was saved and stopped at update 685. Its checkpoint, optimizer state,
settings and pose banks are archived under
`milestones/20260925_133848_standing_update685/`. The last v3 evaluation was at
update 600; the saved update-685 actor is evaluated again before this run trains.

The new run is `recovery_runs/20260925_133848_phased`. Its process and log are
recorded in `recovery_phased_launch.json`. No old checkpoint is overwritten.

| Phase | Initial lean from upright | Training hold target | Evaluation hold target |
|---|---|---|---|
| Large tilt | 25–40 degrees | 2 seconds | 10 seconds |
| Half lying | 40–60 and 60–80 degrees | 2 seconds | 10 seconds |
| Lying | 80–95 degrees | 5 seconds | 10 seconds |

Ranges are nominal: existing pose-bank orientation jitter adds up to roughly
3.4 degrees per roll/pitch axis. Evaluation files record actual starting tilt.
All reset geometry is placed above the floor using collision-mesh bounds.
This does not establish physical recoverability or absence of self-intersection.

The 64 headless robots comprise 32 standing and eight for each recovery
direction. Of recovery resets, 75% use the current phase and 25% rehearse earlier
poses. The previous automatic per-direction curriculum is replaced by explicit
phase gates; standing practice always has a 10-second training hold goal.

Evaluation occurs before training and every 100 PPO updates thereafter. Each
test runs 32 standing and 32 phase-specific directional trials for 30 seconds
without exploration noise or early success resets. Each direction contributes
eight trials. Phase 2 tests four trials in each angular band per direction.
The held-out bank has a separate geometry seed (2043) and 48 poses per group;
evaluation alternates sampling seeds 2043 and 2044. Distinct poses are sampled
within each direction/band. This is a validation set used for curriculum decisions,
not an untouched final test set or proof of real-robot generalization.

Advance only after **two consecutive evaluations** meet all of:

- At least 90% of 32 standing trials qualify for the full 30 seconds.
- At least 80% of 32 recovery trials attain an uninterrupted strict 10-second hold.
- At least 75% of the eight trials in **each** direction succeed.

The strict hold includes torso orientation/height, linear/angular speed, wheel
support, non-wheel floor contact and drift. Short training successes are logged
separately and do not qualify a phase. Passing one direction cannot hide failure
of another direction. Qualified phases are saved as `phase_N_passed.pt`.

Each phase has a budget of 2,000 additional PPO updates, at most 6,000 across all
three phases. A phase that does not qualify stops at its budget; it is not advanced
just because time elapsed. Two successive standing evaluations below 90% restore
the best checkpoint for that phase and stop. The phase-specific best checkpoint
requires standing retention and ranks the weakest recovery direction, overall
recovery success, then average best hold. Numeric checkpoints are saved every 25
updates, after evaluations, and when a stop signal is handled between updates.
SIGTERM during evaluation is handled after that evaluation, which may take minutes.
Resume restores model/optimizer and phase progress, with fresh physical episodes.

The 20 learned motors, open grippers, free base, self-collision, ±0.25-rad target
range, PD 80/4, 13-Nm torque limits, actor/critic learning rates, frozen exploration
standard deviations, v2 reward and gated standing-teacher penalty are retained.
The teacher contributes training gradients near standing only. It never supplies
actions during evaluation. This experiment isolates initial-pose difficulty;
larger recovery motions may eventually require a separately validated wider
action range. No fully lying recovery success is assumed.

Watch `phase_status.json` and `latest_evaluation.json` for the current stage and
measured success. TensorBoard tags `PhaseEvaluation/standing`, `/recovery`, and
`/front_success`, `/back_success`, `/left_success`, `/right_success` are the phase
gates. `metrics.csv` also contains loss, retention loss, action clipping, torque
saturation and strict balance-condition diagnostics. Total loss alone does not
establish get-up success.

The runner is `train_recovery_phased.py`; its `--checkpoint` requires a v3 or
phased checkpoint with settings and pose banks alongside it. The new checkpoint
pointer is `latest_recovery_phased.txt`; the v3 pointer remains at the saved run.
