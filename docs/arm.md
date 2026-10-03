# Arm

A six-joint arm built from six MG5010 motor modules, with a DM4310 rack-and-pinion
gripper. Code: `source/arthrobot_assets/arm/` (model), `source/arthrobot_tasks/arm_reach/`
(reach task) and `scripts/arm/`.

## From the CAD export to a simulation model

The Onshape export (`source/arthrobot_assets/arm/cad/assembly_2/`) is used read-only.
Its links describe the CAD assembly, not the robot's moving bodies, so
`arthrobot_assets.arm.build` regroups its 617 part meshes:

- **10 rigid bodies:** the base, six motor output segments, two jaws and the pinion.
  Every motor's housing parts move with its input side and its output carrier
  with its output side.
- **82 meshes excluded:** a detached, unused seventh MG4010 assembly. Their IDs are
  listed in `build/arm/model_report.json`.
- **Jaw direction fixed:** the export made a moving jaw the parent of the rail.
  Both jaws are now children of the tool body; each jaw carries its rack, clip and
  slider.
- **Geometry preserved:** every one of the 535 included meshes keeps its CAD
  position exactly (checked to 1e-12 m in the zero pose).
- **Joints:** six continuous joints (no angle limits in the model). The gripper has a
  50 mm stroke per jaw.

Masses come from `source/arthrobot/data/parts.json` (weighed or supplier parts) and
`motors.json`. Each part's mass is spread over its CAD volume at uniform density, and
the body inertias follow from the parallel-axis theorem. Total: **3.825 kg** with
MG5010 motors (2.565 kg with MG4010). Not yet calibrated: the 160 g gripper mechanism
estimate, the MG5010 geometry (the CAD still shows MG4010 motors), friction and backlash.

```bash
python -m arthrobot_assets.arm.build        # writes build/arm/arm.urdf and model_report.json
python scripts/arm/view.py                  # GUI with joint and gripper panels
python scripts/arm/view.py --validate          # 28 s headless gripper and arm check
python scripts/arm/validate_gripper_force.py   # isolated jaw-force fixture (headless)
```

## Gripper transmission

- **Drive:** one DM4310 motor rated at 3 N m, turning a module-1, 16-tooth pinion
  (8 mm pitch radius).
- **Coupling:** jaw 0 is the driven coordinate. Jaw 1 and the pinion follow it
  through PhysX mimic joints, so there is one shared torque budget.
- **Force limit:** the driven jaw is capped at torque / radius = 375 N. With equal
  loads that is 187.5 N per jaw; a single blocked jaw can take the full 375 N.
  These are ideal, lossless model forces, not measured hardware ratings.
- **Physics:** gripper contact is validated at 960 Hz with PGS 128/128 iterations.

## Validation

| Check | Result |
| --- | --- |
| Assembled arm (`view.py --validate`, 26,880 steps) | quarter, half and full closure, contact, reopening; worst rack-pinion error 0.0018 mm |
| Force fixture (`validate_gripper_force.py`) | 31.25 / 62.50 / 187.50 N per jaw at 0.5 / 1 / 3 N m; 375 N on one blocked jaw; worst coupling error 0.089 mm |
| Model tests (`tests/test_arm_model.py`) | mesh and mass accounting, jaw membership, full stroke, pinion axis, shared actuator budget, physical inertias |

## Motor study: MG4010 vs MG5010

The first arm prototype used MG4010 modules. A lifting test compared them with the
MG5010 profile (460 g, 13 N m rated, 74 rpm, 36:1). The test used the same PD gains,
self-collision and the CAD orientation, in which joint 1 is horizontal and carries
the arm's weight. The table gives each joint angle at the end of a 10 s hold:

| Command | MG4010 | MG5010 | MG5010 self-contact |
| --- | ---: | ---: | --- |
| Joint 2 +60° | 54.1° | 57.0° | none |
| Joint 2 −60° | −17.7° | −54.7° | base / upper arm |
| Joint 4 +60° | 25.0° | 56.9° | none |
| Joint 4 +90° | 25.1° | 87.7° | none |
| Joint 4 +120° | 25.1° | 106.6° | base / tool |
| Joint 2 +45°, joint 4 −45° | 42.0°, −44.9° | 42.2°, −44.9° | none |

The MG4010 could not lift the forearm (joint 4 stayed near 25°). With self-collision
off, the MG5010 reached every pose within 5°, so its two larger shortfalls in the table
(joint 2 −60° and joint 4 +120°) come from self-contact. The MG5010 became the
default module.

The study shows only that the unloaded model can be lifted in these poses. It is not
a payload rating or a thermal test. In that orientation, a static torque search over
all joint angles found a worst-case payload of 0.26 kg, limited by joint 1. With joint 1 vertical,
as on the real robot, the worst static joint torque over 4,000 random poses is
9.8 N m (joint 2), below the 13 N m rating.

## Reach task

Task `ArthroBot-Arm-Reach-v0`: move the grasp center between the jaws (the TCP) to
random 3D targets. It is an Isaac Lab manager-based task trained with rsl_rl PPO
using the official `train.py` and `play.py`.

**Mount.** The CAD lies on its side. Training rotates the base so that joint 1 is a
vertical base-yaw axis, as on the real robot. The origin of the mount frame
(`arm_reach/mount.py`) is the bottom of the base on the joint 1 axis, with x forward
and z up.

| | |
| --- | --- |
| Actions (6) | joint position targets, zero pose + 0.5 rad × action |
| Observations (24) | joint positions and velocities, TCP position, target position, last action |
| Targets | 0.15–0.40 m forward, ±0.25 m sideways, 0.05–0.40 m up; a new target every 4 s |
| Rewards | −0.2 × distance, +0.1 × (1 − tanh(distance / 5 cm)), joint-limit penalty, action-rate and joint-velocity penalties (ramped up by a curriculum) |
| Episode | 12 s; 120 Hz physics, 30 Hz policy; resets at zero pose ± 0.25 rad |

Training-only settings: joint limits of ±180° (the real cable limits are not
recorded yet), one convex hull per collision mesh, and TGS 8/1 iterations.

```bash
python scripts/rsl_rl/train.py --task ArthroBot-Arm-Reach-v0 --headless --num_envs 1024
python scripts/rsl_rl/play.py --task ArthroBot-Arm-Reach-Play-v0 --checkpoint checkpoints/arm_reach/model_550.pt
python scripts/arm/evaluate.py                 # 256 arms, 1,536 targets, no action noise
python scripts/arm/check_reach_env.py          # hold, limits, gains, contacts, random-action stability
```

**Result** (`checkpoints/arm_reach/model_550.pt`; 1,024 arms, 600 updates take about
12 minutes on an RTX 4080 SUPER):

- TCP error just before the next target appears: median **2.2 mm**, 90th percentile
  5.5 mm, 99th percentile 11.7 mm;
- **97.7%** of targets within 1 cm and 99.9% within 2 cm;
- median time to come within 2 cm: 0.3 s.

Training past about 600 updates made the policy worse, at both entropy 0.01 and 0.001.
The worst targets are close to the base, where the arm has to fold.

## Not modeled yet

- Real joint ranges and cable limits.
- A table or base plate.
- Grasping: three of the four silicone pads are missing from the CAD.
- Motor friction, backlash, latency and torque–speed curves. The drive gains are
  simulation choices, not measured controller gains.
