# Humanoid

A wheel-legged humanoid built from 20 MG5010 motor modules: two six-joint arms
with DM4310 grippers, and two legs with two hip joints, a knee and a driven wheel.
Code: `source/arthrobot_assets/humanoid/` (model), `source/arthrobot_tasks/humanoid/`
(tasks) and `scripts/humanoid/`.

| Task | Status (simulation only) | Included policy |
| --- | --- | --- |
| [Standing](#standing) | all 32 deterministic 15 s trials pass | `checkpoints/humanoid_standing/model_4999.pt` |
| [Get-up from lying](#get-up-from-lying) | stands up from 99.4% of held-out lying poses with no assistance, in 1.6 s | `checkpoints/humanoid_getup/model_10000.pt` |
| [Hand-over](#hand-over-from-get-up-to-standing) | 99.4% stay up after switching to the standing policy | both of the above |

Nothing has been tried on the real robot yet.

## Model

The Onshape export (`cad/modular_humanoid_bipad/`) is used read-only. It has 178
links and 2,419 meshes; 2,252 meshes hang directly off the root, so its links are
not the robot's bodies. `arthrobot_assets.humanoid.build` regroups them:

- **27 rigid bodies and 26 joints:** 20 motor joints, plus the jaws and pinion of
  each gripper (each gripper is still one motor).
- **Motor ownership:** each motor's housing moves with its input-side body and its
  output carrier with its output-side body. The axis of all 20 motors is checked
  against the motor housing.
- **Excluded:** 669 meshes of disconnected spares (a third torso module with six
  unjointed motors, two loose wheels, two old gripper motors). The 1,750 kept
  meshes keep their exact CAD positions.
- **Collisions:** 150 collision meshes, made of 40 motor exterior hulls and 110
  detailed bracket and gripper meshes.
- **Mass:** 13.27 kg. The torso/battery modules and wheels have not been weighed
  and use solid-PLA density estimates; enter measured values in
  `source/arthrobot_assets/humanoid/hardware.json`.

| Branch | Joints (in order) |
| --- | --- |
| Left / right arm | `<side>_arm_1` … `<side>_arm_6` |
| Left / right leg | `<side>_hip_1`, `<side>_hip_2`, `<side>_knee`, `<side>_wheel` |

Every policy uses the motor order in `MOTOR_JOINTS`: 12 arm joints, 6 leg joints,
2 wheels.

`training_model.py` makes the 21-body training variant. It parks the grippers open
and merges them into the tool bodies, and it turns the torso frame to x forward,
y left, z up. `nominal_pose.py` solves the upright standing pose. The legs are set so
that the center of mass is above the wheel axle with both wheels level, and the
shoulders are raised so the grippers clear the legs. The torso starts at 0.516 m;
the largest static joint torque in this pose is 6.4 N m.

```bash
python -m arthrobot_assets.humanoid.build             # build/humanoid/humanoid.urdf
python -m arthrobot_assets.humanoid.training_model    # build/humanoid_training/robot.urdf
python scripts/humanoid/view.py                       # suspended motor test rig with a control panel
python scripts/humanoid/view.py --validate --headless
```

`view.py --validate` drives every motor both ways, spins each wheel more than one
turn in both directions, cycles both grippers, and then holds a pose under gravity.
Results:

- largest motor target error: 0.016°;
- largest body position error against forward kinematics: 0.0003 mm;
- largest gripper coupling error: 0.002 mm;
- both grippers stop at 48.9 mm per jaw on contact;
- gravity-hold sag: up to 3.8° (PD deflection).

## Standing

Hybrid control. All 20 motors are learned, the torso is free, and there is no
balance controller.

| | |
| --- | --- |
| Actions 0–17 | joint targets = standing pose + 0.25 rad × action, tracked by PD 80 N m/rad, 4 N m s/rad at 240 Hz (angle errors wrap) |
| Actions 18–19 | wheel torque = 13 N m × action |
| Limits | every motor clipped at 13 N m and 74 rpm; implicit PhysX drives off |
| Observation (96) | body velocities, gravity direction, sin/cos of joint angles, joint speeds, last actions, wheel contacts, torso height, COM offset, heading |
| Reward | upright, survival, height, still COM, small drift and heading change, both wheels down, arm clearance from the floor, soft posture, torque, action-change and body-rotation costs |
| PPO | rsl_rl, [256, 256, 128], 48 steps per robot, γ 0.995, entropy 0.002, initial std 0.25, no observation normalization |
| Physics | full asset (convex decomposition), PGS 32/8 at 240 Hz, 60 Hz policy |

**Result** (`model_4999.pt`, 5,000 updates with 12 robots). The evaluation runs
32 deterministic 15 s trials, and **all 32 qualify**. To qualify, a trial must keep
tilt below 20°, the torso above 0.46 m and both wheels down at least 80% of the
time, with a mean speed under 0.15 m/s and drift under 25 cm. The largest tilt was 1.6° and the largest drift 1 cm.

```bash
python scripts/humanoid/train_standing.py --mode check --num-envs 16           # physics and control checks
python scripts/humanoid/train_standing.py --mode train --headless --iterations 5000 --eval-interval 250
python scripts/humanoid/train_standing.py --mode evaluate --headless --num-envs 16 --seconds 30
python scripts/humanoid/train_standing.py --mode preview                       # watch it in the GUI
```

## Get-up from lying

Adapted from [HoST](https://arxiv.org/abs/2502.08378) (Huang et al., 2025) to a robot
with wheels instead of feet. The code is in `source/arthrobot_tasks/humanoid/getup/`.

**Fast physics.** The asset is a light override layer of the training USD: one
convex hull per collision mesh, the joint limits below, and no visual meshes.
Physics is TGS 8/1 at 240 Hz. `benchmark_physics.py` measures the speed-up, with
random PD motion from random orientations:

| Setup | Robots | Transitions/s |
| --- | ---: | ---: |
| full asset, PGS 32/8 | 64 | 229 |
| get-up asset, PGS 32/8 | 64 | 517 |
| get-up asset, TGS 8/1 | 64 | 2,420 |
| get-up asset, TGS 8/1 | 1,024 | 27,603 |

The standing policy still balances in this setup (`validate_standing_physics.py`):
32/32 trials for 30 s, with a largest tilt of 1.3° and a largest drift of 1.2 cm.

**Joint limits from self-collision.** Each limb joint is swept alone from the
standing pose until its convex hulls come within 3 mm of an unjointed body. The
limit is that angle minus 5° (`getup/data/joint_limits.json`). These are geometric
limits only, not the real cable or bracket limits, so combined poses still rely on
simulated self-collision.

**Task:**

- **Starts:** pose banks (`getup/data/pose_banks/`) with the robot lying on its back,
  front, left or right side (±25° jitter, random yaw), or in a random orientation.
  Arms are the standing pose ± 0.6 rad and legs ± 0.4 rad, clipped to the limits.
  The bank holds 2,000 training poses and 160 held-out poses. Self-colliding poses
  are rejected.
- **Episode:** 12 s. The first 0.6 s is passive, so the robot settles first.
- **Actions:** limb target = current angle + bound × action, clipped to the limits.
  The bound decays from 1.0 to 0.25 rad over 3,000 updates. The wheel actions are
  torques.
- **Observations:** the actor sees 6 frames of angular velocity, gravity direction,
  joint angles and speeds, and previous actions. The critic also sees torso height,
  linear velocity, the contact state of all 21 bodies and the curriculum state.
- **Reward groups**, each with its own critic and advantage weight:

  | Group | Weight | Terms |
  | --- | ---: | --- |
  | task | 2.5 | torso height, uprightness, standing |
  | regularization | 0.1 | joint acceleration, action rate and smoothness, torque, power, soft joint and speed limits |
  | style | 1 | hip abduction, shanks pointing down, wheel track, low tilt rate, no body–floor contact near standing |
  | target | 1 | near standing: stillness, level torso, height, both wheels down, and the ending pose (below) |
  | safety | 1.5 | new self-contact, contact left from settling, limb speed above 50%, torque above 9 N m, torso rotation rate; near standing, arm joints resting on their stops |

- **Ending pose** (target group, near standing): the robot should end in the standing
  policy's pose, so that policy can take over.
  - Arms, HoST-style: `exp(−0.1 × Σ error²)` over the 12 arm joints.
  - Legs: `exp(−1.0 × Σ error²)` over the 6 leg joints.
  - Arms, per joint: a linear term, so the pull doesn't vanish when every arm joint is far away.
  - While getting up the arms are free; these terms only count once the robot is near
    standing (torso above 0.465 m, tilt below 30°).

- **Curricula:**
  - an upward pull on the torso starts at 65 N (about half the body weight) and
    drops by 13 N after each held-out evaluation with at least 15% success;
  - the action bound shrinks as above.
- **Randomization:**
  - at startup: friction, restitution, link masses ±10%, torso payload −0.3 to
    +0.8 kg, torso COM ±2 cm;
  - at each reset: PD gains ±15%, motor strength 85–100%, joint offsets ±0.03 rad,
    action delay 0–25 ms;
  - observation noise.
- **PPO:** multi-critic PPO (`getup/ppo.py`; rsl_rl has no multi-critic version).
  Advantages are computed and normalized per group. The actor is [512, 256, 128] and
  the critic [512, 256] with one head per group. It adds L2C2-style smoothness and
  bootstraps timeouts from the true final state.
  - Entropy bonus 0.005.
  - Action noise std between 0.2 and 0.6; the arm actions are capped at 0.25.

**Success** means standing (torso above 0.46 m, tilt below 15°, both wheels down) for
2 s in a row. Evaluations are deterministic, with no pull force and no observation
noise.

**Result** (`model_10000.pt`):

| Measure | 4,096 robots cycling the 160 held-out poses (training evaluation) | `evaluate_getup.py` (320 robots, seed 7) |
| --- | ---: | ---: |
| stood up and held 2 s | 99.8% | 99.4% |
| time to stand (after the motors switch on) | 1.62 s | 1.60 s |
| stood up without creating self-contact | 25.9% | 25.9% |
| largest arm-joint distance from the standing pose at the end | 1.64 rad | 1.64 rad |
| peak limb joint speed | 5.4 rad/s (motor max 7.75) | 5.3 rad/s |

**How it was trained.** It was trained from scratch, in two stages. The docstring of
`train_getup.py` gives both commands, and every setting is recorded in
`checkpoints/humanoid_getup/settings.json`.

| Updates | Change | Outcome |
| --- | --- | --- |
| 0–5,700 | ending pose (HoST-style arm and leg terms), entropy 0.005, noise cap 0.6, 12 s episodes | gets up with the assist; pull force 65 → 52 N by update 5,250; but 10 of 12 arm joints rest on their joint stops |
| 5,700–12,000 | per-joint linear arm term, arm-noise cap 0.25, joint-stop penalty near standing | pull force 52 → 0 N by update 6,500; 99% stand up from 6,750; hand-over 92–100% from 7,500 |

Update 10,000 has the best full hand-over test and the smallest arm error, so it is
the included policy. The per-update history is in
`checkpoints/humanoid_getup/evaluation_history.json`.

**Why the arms used to stay raised.** The previous included policy (update 25,500 of
an earlier training line) stood up just as reliably. But it ended with its arms over
its head, and only 45–58% survived the hand-over. Three things caused it:
1. **Its posture reward was too narrow.** It was `exp(−2 × Σ error²)` over all 18 limb
   joints, which pays nothing when the arms are far away. HoST's is `exp(−0.1 × Σ error²)`
   over the upper body only.
2. **The arms random-walk into the joint stops.** Arm actions move the target relative
   to the current angle, and arm noise stayed high (0.5–0.8) because the arms barely
   affect the reward. So the arms wander until they rest on a stop.
3. **Even the HoST-style term is flat from the stops.** With 10 joints there, the sum of
   squared errors is about 55 rad², and `exp(−0.1 × 55) ≈ 0.004`.

The per-joint linear term, the lower arm-noise cap and the joint-stop penalty fixed
this. The arms now hang down at the sides.

```bash
python scripts/humanoid/train_getup.py --num-envs 4096                    # train (headless); logs/humanoid_getup/
python scripts/humanoid/evaluate_getup.py                                 # held-out evaluation + self-contact by body
python scripts/humanoid/record_getup.py                                   # MP4 of the five start families
python scripts/humanoid/watch_getup_run.py --run-dir logs/humanoid_getup/<run> --every 250 --handover
                                                                          # video + hand-over test per new checkpoint
python scripts/humanoid/validate_standing_physics.py                      # standing policy in the get-up physics
python scripts/humanoid/benchmark_physics.py --num-envs 1024              # physics throughput
python scripts/humanoid/make_pose_banks.py lying --seed 42 --per-family 400
python scripts/humanoid/sweep_joint_limits.py
```

Stop training with SIGTERM: the trainer finishes the current update, saves and exits.

**Demo clips** (`docs/media/humanoid_*.gif`). They need the RTX workaround (see the
README) and are written to `build/humanoid_training/getup_videos/`. Each rendered robot
needs about 1.5 GB of host memory, so record one family per run:

```bash
V=build/humanoid_training/getup_videos
# Get-up, then the hand-over to the standing policy: all five start families, 10 s
python scripts/humanoid/record_getup.py --plain --handover --seconds 10
python scripts/make_gif.py $V/update_010000_handover_plain.mp4 docs/media/humanoid_getup_starts.gif \
    --width 840 --fps 10 --colors 64
# The same for one robot, close up
python scripts/humanoid/record_getup.py --plain --handover --families front --tile-width 800 --tile-height 450 --seconds 10
python scripts/make_gif.py $V/update_010000_front_handover_plain.mp4 docs/media/humanoid_getup_to_standing.gif \
    --width 600 --fps 10 --colors 64
```

**Joint-limit provenance.** The committed `joint_limits.json` was swept from an
earlier standing pose, which differs from the current one by up to 2.6° (left hip).
The included policy was trained with these limits, so they stay unchanged.
`sweep_joint_limits.py` sweeps from the current standing pose. Re-sweep, re-make the
pose banks and retrain together.

## Hand-over from get-up to standing

`handoff.py` runs the get-up policy. Once a robot has stood for 1 s, even-numbered
robots switch to the standing policy and odd-numbered robots keep the get-up policy.
The standing policy's heading and COM inputs are measured from the moment it takes
over.

| 640 robots, 30 s, seed 7 | Stayed standing | Average largest tilt |
| --- | ---: | ---: |
| switched to the standing policy (319 robots) | 99.4% | 6.6° |
| kept the get-up policy (320 robots) | 100% | 3.1° |

With the previous get-up policy only 45–58% survived the hand-over, because its arms
were raised over its head (see "Why the arms used to stay raised" above).
`record_getup.py --handover` records the whole sequence; see the demo clips above.

**What is still missing.** The arms end down at the sides, but not exactly in the
standing pose: the largest arm-joint error is about 1.6 rad, so every robot still has an
arm joint outside the standing policy's ±0.25 rad range at hand-over. The standing
policy copes with that. Next steps:
1. bring the arms fully into the standing pose;
2. test pushes and starts outside the training poses;
3. move on to locomotion from the standing pose.

The tilt-recovery experiments that came before the get-up task are summarized in
[history/](history/README.md).
