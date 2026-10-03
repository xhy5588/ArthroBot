"""Physics throughput of the humanoid: asset variant x solver x robot count (headless).

One configuration per process. Robots are dropped in random orientations and
driven by random PD targets within the joint limits plus random wheel torques,
which resembles get-up rollouts. Prints one JSON line with control transitions
per second (4 physics steps per control step at 240 Hz).

    python scripts/humanoid/benchmark_physics.py --asset getup --solver tgs --num-envs 1024
    python scripts/humanoid/benchmark_physics.py --asset training --solver pgs --position-iterations 32 --velocity-iterations 8
"""
import argparse
import json
import math
import os
import time
import traceback

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--asset', choices=['training', 'getup'], default='getup',
                    help='training: convex decomposition (standing); getup: one convex hull per mesh.')
parser.add_argument('--solver', choices=['pgs', 'tgs'], default='tgs')
parser.add_argument('--position-iterations', type=int, default=8)
parser.add_argument('--velocity-iterations', type=int, default=1)
parser.add_argument('--num-envs', type=int, default=64)
parser.add_argument('--steps', type=int, default=240, help='Control steps timed after a 120-step warm-up.')
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

PHYSICS_DT = 1 / 240
PHYSICS_STEPS_PER_CONTROL = 4
TARGET_NOISE_RAD = .15


def main():
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import random_orientation
    from arthrobot.motors import POSITION_GAIN_NM_PER_RAD, VELOCITY_GAIN_NM_S_PER_RAD, load_motor
    from arthrobot_tasks.humanoid.assets import ensure_getup_usd, ensure_training_usd

    motor = load_motor('mg5010')
    usd_path = (ensure_training_usd() if args.asset == 'training' else ensure_getup_usd())[0]

    @configclass
    class BenchmarkSceneCfg(InteractiveSceneCfg):
        ground = TerrainImporterCfg(prim_path='/World/ground', terrain_type='plane')
        robot = ArticulationCfg(
            prim_path='{ENV_REGEX_NS}/Robot',
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path), activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, solver_position_iteration_count=args.position_iterations,
                    solver_velocity_iteration_count=args.velocity_iterations)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., .4)),
            actuators={'motors': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=0., damping=0.,
                                                     effort_limit_sim=motor.rated_torque_nm,
                                                     velocity_limit_sim=motor.max_speed_rad_s,
                                                     armature=motor.reflected_inertia_kg_m2)})
        contacts = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*', history_length=1, update_period=PHYSICS_DT)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=PHYSICS_DT, device=args.device,
        physx=sim_utils.PhysxCfg(solver_type=0 if args.solver == 'pgs' else 1, gpu_max_rigid_contact_count=2**22,
                                gpu_max_rigid_patch_count=2**19, gpu_found_lost_pairs_capacity=2**22)))
    scene = InteractiveScene(BenchmarkSceneCfg(num_envs=args.num_envs, env_spacing=3.))
    sim.reset()
    robot = scene['robot']
    n, torque_limit = args.num_envs, motor.rated_torque_nm
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    root_state[:, 3:7] = random_orientation(n, device=sim.device)
    robot.write_root_state_to_sim(root_state)
    lower = robot.data.soft_joint_pos_limits[..., 0].clamp(min=-math.pi)
    upper = robot.data.soft_joint_pos_limits[..., 1].clamp(max=math.pi)
    is_wheel = torch.tensor(['wheel' in name for name in robot.joint_names], device=sim.device)
    targets = robot.data.joint_pos.clone()

    def control_step():
        nonlocal targets
        targets = (targets + TARGET_NOISE_RAD * torch.randn_like(targets)).clamp(lower, upper)
        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            torques = (POSITION_GAIN_NM_PER_RAD * (targets - robot.data.joint_pos)
                       - VELOCITY_GAIN_NM_S_PER_RAD * robot.data.joint_vel).clamp(-torque_limit, torque_limit)
            torques[:, is_wheel] = torque_limit * (2 * torch.rand(n, int(is_wheel.sum()), device=sim.device) - 1)
            robot.set_joint_effort_target(torques)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(PHYSICS_DT)

    for _ in range(120):
        control_step()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.steps):
        control_step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    finite = bool(torch.isfinite(robot.data.root_state_w).all() and torch.isfinite(robot.data.joint_pos).all())
    control_rate_hz = 1 / (PHYSICS_DT * PHYSICS_STEPS_PER_CONTROL)
    print('HUMANOID_PHYSICS_BENCHMARK: ' + json.dumps(dict(
        asset=args.asset, solver=args.solver, position_iterations=args.position_iterations,
        velocity_iterations=args.velocity_iterations, num_envs=n, transitions_per_s=args.steps * n / seconds,
        realtime_factor=args.steps / control_rate_hz * n / seconds, finite=finite,
        mean_root_height_m=float((robot.data.root_pos_w[:, 2] - scene.env_origins[:, 2]).mean()))), flush=True)


exit_code = 1
try:
    main()
    exit_code = 0
except BaseException:
    traceback.print_exc()
finally:
    # Kit can hang while shutting down after headless physics runs; the result is already printed.
    os._exit(exit_code)
