"""When a standing trial counts as success, and held-out evaluation during training."""
import csv
import json
import random
import statistics

import numpy as np
import torch

from arthrobot_assets.humanoid import MOTOR_JOINTS

CRITERIA = dict(minimum_trials=16, minimum_accepted_fraction=.9, duration_s=14.9, maximum_tilt_deg=20.,
                minimum_torso_height_m=.46, maximum_mean_horizontal_speed_m_s=.15, maximum_absolute_axis_drift_m=.25,
                minimum_both_wheels_contact_fraction=.8)


def is_standing_trial(record: dict) -> bool:
    """A full episode upright, high, still and on both wheels."""
    return (record['survived'] and record['seconds'] >= CRITERIA['duration_s']
            and record['max_torso_tilt_deg'] <= CRITERIA['maximum_tilt_deg']
            and record['minimum_torso_height_m'] >= CRITERIA['minimum_torso_height_m']
            and record['mean_horizontal_speed_m_s'] <= CRITERIA['maximum_mean_horizontal_speed_m_s']
            and abs(record['forward_m']) <= CRITERIA['maximum_absolute_axis_drift_m']
            and abs(record['lateral_m']) <= CRITERIA['maximum_absolute_axis_drift_m']
            and record['both_wheels_contact_fraction'] >= CRITERIA['minimum_both_wheels_contact_fraction'])


def qualify_standing(episodes: list[dict]) -> dict:
    """A policy qualifies when at least 16 trials ran and at least 90% of them stood."""
    accepted = sum(is_standing_trial(record) for record in episodes)
    fraction = accepted / max(1, len(episodes))
    return dict(qualified=len(episodes) >= CRITERIA['minimum_trials'] and fraction >= CRITERIA['minimum_accepted_fraction'],
                accepted_trials=accepted, completed_trials=len(episodes), accepted_fraction=fraction, criteria=CRITERIA)


@torch.inference_mode()
def evaluate_standing(runner, wrapper, env, run_dir, completed_updates: int, seed: int = 43, seconds: float = 30.):
    """Deterministic held-out trials between PPO chunks; training state is restored afterwards.

    The rollout storage is empty between chunks. RNG states, episode counters and
    statistics are saved and restored; the environment is reset before learning resumes.
    """
    rng_states = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    counters = env.total_resets, env.total_successes
    saved_records = env.completed_episode_records.copy()
    evaluation_dir = run_dir / 'evaluations'
    evaluation_dir.mkdir(exist_ok=True)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        policy = runner.get_inference_policy(device=env.device)
        env.reset()
        env.completed_episode_records.clear()
        trace = []   # a full-rate trace of robot 0 for the first 300 steps, sampled before each action
        for step in range(round(seconds / env.step_dt)):
            actions = policy(wrapper.get_observations())
            if step < 300:
                com, velocity = env.com_state()
                trace.append(torch.cat((torch.tensor([step * env.step_dt], device=env.device), actions[0],
                                        env.commanded_torques(actions)[0], env.robot.data.joint_vel[0, env.motor_ids],
                                        com[0] - env.start_com[0], velocity[0], env.torso_height()[:1],
                                        env.robot.data.projected_gravity_b[0])).cpu().tolist())
            env.step(actions)
            if (step + 1) % 600 == 0:
                print(f'STANDING_EVALUATION_PROGRESS: update {completed_updates}, '
                      f'{(step + 1) * env.step_dt:.0f}/{seconds:g} simulated seconds', flush=True)
        rows = env.completed_episode_records.copy()
        result = dict(update=completed_updates, seed=seed, exploration=False, completed_trials=len(rows),
                      qualification=qualify_standing(rows),
                      mean_episode_seconds=statistics.mean([row['seconds'] for row in rows]) if rows else 0.,
                      survived_fraction=statistics.mean([row['survived'] for row in rows]) if rows else 0.,
                      episodes=rows)
        (evaluation_dir / f'update_{completed_updates:06d}.json').write_text(json.dumps(result, indent=2) + '\n')
        with (evaluation_dir / f'trace_{completed_updates:06d}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_s'] + [f'raw_action/{name}' for name in MOTOR_JOINTS]
                            + [f'commanded_torque_nm/{name}' for name in MOTOR_JOINTS]
                            + [f'velocity_rad_s/{name}' for name in MOTOR_JOINTS]
                            + ['com_dx_m', 'com_dy_m', 'com_dz_m', 'com_vx_m_s', 'com_vy_m_s', 'com_vz_m_s',
                               'torso_height_m', 'gravity_x', 'gravity_y', 'gravity_z'])
            writer.writerows(trace)
        if runner.writer is not None:
            for name, value in (('mean_episode_seconds', result['mean_episode_seconds']),
                                ('survived_fraction', result['survived_fraction']),
                                ('standing_success_fraction', result['qualification']['accepted_fraction'])):
                runner.writer.add_scalar(f'Evaluation/{name}', value, completed_updates - 1)
            runner.writer.flush()
        print('STANDING_EVALUATION: ' + json.dumps({key: value for key, value in result.items() if key != 'episodes'}),
              flush=True)
        return result
    finally:
        python_state, numpy_state, torch_state, cuda_state = rng_states
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        env.reset()
        env.completed_episode_records[:] = saved_records
        env.total_resets, env.total_successes = counters
        env.extras['log'] = {}
        env.control_stats.zero_()
        env.control_stat_steps = 0
        runner.train_mode()
