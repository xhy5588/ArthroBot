"""Plan a pick sequence two ways around a table and a wall, and save the joint trajectories. No Isaac Sim.

    python scripts/arm/plan_trajectories.py [--obstacle-margin 0.01 --name plans_margin15]

Targets are top-down grasps that alternate left and right of the wall
(arthrobot_tasks.arm_planning.obstacles), starting from a home pose above the wall.

  ik_joint  the IK controller: pose IK, then a minimum-jerk joint path
            (arthrobot_tasks.arm_planning.controllers). It knows nothing about obstacles.
  curobo    cuRobo's MotionPlanner with the arm's sphere model (scripts/arm/build_curobo_robot.py),
            the table, the wall and self-collision.

Both are then checked with cuRobo's collision checker (scene and self). Writes
logs/arm_planning/<name>.npz (joint trajectories at the control rate) and <name>.json.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState

from arthrobot import paths
from arthrobot_tasks.arm_planning.controllers import IKController
from arthrobot_tasks.arm_planning.curobo_robot import config_path
from arthrobot_tasks.arm_planning.kinematics import ArmModel
from arthrobot_tasks.arm_planning.obstacles import curobo_scene
from arthrobot_tasks.arm_planning.targets import FLIP, home_pose, matrix_to_quaternion, pick_targets

CONTROL_HZ = 30.
DEVICE = 'cuda:0'


def plan_ik_joint(model, home, positions, rotations):
    """The current controller's joint paths: list (per sequence) of lists (per target) of [T, 6] arrays."""
    sequences = positions.shape[0]
    no = torch.zeros(sequences, dtype=torch.bool)
    ik = IKController(model, sequences, 'cpu', no, no)
    ik.q_desired = home.expand(sequences, 6).clone()
    plans = [[] for _ in range(sequences)]
    for k in range(positions.shape[1]):
        ik.new_target(positions[:, k], rotations[:, k])
        steps = int(math.ceil(float(ik.duration.max()) * CONTROL_HZ))
        path = []
        for _ in range(steps):
            ik.elapsed += 1 / CONTROL_HZ
            blend = ik.elapsed / ik.duration
            s = blend.clamp(0., 1.)[:, None]
            path.append(ik.q_from + s**3 * (10 - 15 * s + 6 * s**2) * (ik.q_goal - ik.q_from))
        ik.q_desired = ik.q_goal.clone()
        for i in range(sequences):
            # Each arm's own move ends at its own duration; trim the hold.
            own = int(math.ceil(float(ik.duration[i]) * CONTROL_HZ))
            plans[i].append(torch.stack(path)[:own, i].numpy())
    return plans


def plan_curobo(planner, model, home, positions, rotations):
    """cuRobo trajectories (same layout as plan_ik_joint) and per-target success and planning time."""
    plans, success, seconds = [], [], []
    for i in range(positions.shape[0]):
        q = home.float().to(DEVICE)
        row, ok_row, time_row = [], [], []
        for k in range(positions.shape[1]):
            # Try the grasp orientation our IK reaches first, then the same grasp turned 180 deg.
            _, reached = model.pose(model.ik_pose(positions[i, k][None], rotations[i, k][None], q.double().cpu()[None])[0])
            candidates = [rotations[i, k], rotations[i, k] @ FLIP]
            if float((reached[0] - candidates[1]).abs().sum()) < float((reached[0] - candidates[0]).abs().sum()):
                candidates.reverse()
            started, result = time.time(), None
            for rotation in candidates:
                goal = GoalToolPose(tool_frames=['tcp'],
                                    position=positions[i, k].float().view(1, 1, 1, 1, 3).to(DEVICE),
                                    quaternion=matrix_to_quaternion(rotation[None]).view(1, 1, 1, 1, 4).to(DEVICE))
                result = planner.plan_pose(goal, JointState.from_position(q.view(1, 6), joint_names=planner.joint_names))
                if result is not None and bool(result.success.any()):
                    break
            time_row.append(time.time() - started)
            ok = result is not None and bool(result.success.any())
            ok_row.append(ok)
            if ok:
                path = result.get_interpolated_plan().position.reshape(-1, 6)
                path = path[:int(result.path_buffer_last_tstep[0]) + 1] if hasattr(result, 'path_buffer_last_tstep') \
                    and result.path_buffer_last_tstep is not None else path
                row.append(path.cpu().numpy())
                q = path[-1].clone()
            else:
                row.append(q.view(1, 6).cpu().numpy())  # Failed: stay put.
        plans.append(row)
        success.append(ok_row)
        seconds.append(time_row)
    return plans, np.array(success), np.array(seconds)


def collision_check(checker, plans):
    """Per trajectory: True if any waypoint penetrates the scene or the arm itself (cuRobo's sphere model)."""
    hits = np.zeros((len(plans), len(plans[0])), dtype=bool)
    for i, row in enumerate(plans):
        for k, path in enumerate(row):
            q = torch.tensor(path, dtype=torch.float32, device=DEVICE).view(-1, 1, 6)
            scene, own = checker.get_scene_self_collision_distance_from_joints(q)
            hits[i, k] = bool((scene.view(len(q), -1).amax(-1) > 0).any() or (own.view(len(q), -1).amax(-1) > 0).any())
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sequences', type=int, default=8)
    parser.add_argument('--targets', type=int, default=6)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--out', type=Path, default=paths.LOGS_DIR / 'arm_planning')
    parser.add_argument('--obstacle-margin', type=float, default=0.0,
                        help='Grow every obstacle by this much (m) when planning, e.g. 0.01 for a loose arm.')
    parser.add_argument('--name', default='plans', help='Output file stem.')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = ArmModel()
    home = home_pose(model)
    positions, rotations = pick_targets(model, args.sequences, args.targets, args.seed)
    print(f'home pose (deg): {np.degrees(home.numpy()).round(1).tolist()}')

    ik_plans = plan_ik_joint(model, home, positions, rotations)

    planner = MotionPlanner(MotionPlannerCfg.create(
        robot=str(config_path()), scene_model=curobo_scene(args.obstacle_margin), position_tolerance=0.001,
        orientation_tolerance=0.01,
        interpolation_dt=1 / CONTROL_HZ))
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    curobo_plans, success, seconds = plan_curobo(planner, model, home, positions, rotations)

    # Collisions are judged with the standard 5 mm margin, whatever margin the plan used.
    checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=str(config_path()), scene_model=curobo_scene(), collision_activation_distance=0.0))
    ik_hits, curobo_hits = collision_check(checker, ik_plans), collision_check(checker, curobo_plans)

    durations = lambda plans: np.array([[len(p) / CONTROL_HZ for p in row] for row in plans])
    summary = {
        'obstacle_margin_m': args.obstacle_margin,
        'targets': int(success.size),
        'home_deg': np.degrees(home.numpy()).round(2).tolist(),
        'curobo_success': round(float(success.mean()), 3),
        'curobo_plan_time_s': {'median': round(float(np.median(seconds)), 3), 'max': round(float(seconds.max()), 3)},
        'ik_joint_colliding_fraction': round(float(ik_hits.mean()), 3),
        'curobo_colliding_fraction': round(float(curobo_hits[success].mean()), 3) if success.any() else None,
        'move_duration_s': {'ik_joint_median': round(float(np.median(durations(ik_plans))), 2),
                            'curobo_median': round(float(np.median(durations(curobo_plans)[success])), 2)},
    }
    print(json.dumps(summary, indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'{args.name}.json').write_text(json.dumps(summary, indent=2) + '\n')
    pad = lambda plans: _pad(plans)
    np.savez_compressed(args.out / f'{args.name}.npz', home=home.numpy(), positions=positions.numpy(),
                        rotations=rotations.numpy(), ik_joint=pad(ik_plans)[0], ik_joint_steps=pad(ik_plans)[1],
                        curobo=pad(curobo_plans)[0], curobo_steps=pad(curobo_plans)[1], curobo_success=success,
                        ik_joint_collides=ik_hits, curobo_collides=curobo_hits, control_hz=CONTROL_HZ)


def _pad(plans):
    """Ragged per-target paths -> [S, K, T_max, 6] (held at the last point) and step counts [S, K]."""
    steps = np.array([[len(p) for p in row] for row in plans])
    out = np.zeros((*steps.shape, int(steps.max()), 6), dtype=np.float32)
    for i, row in enumerate(plans):
        for k, path in enumerate(row):
            out[i, k, :len(path)] = path
            out[i, k, len(path):] = path[-1]
    return out, steps


if __name__ == '__main__':
    main()
