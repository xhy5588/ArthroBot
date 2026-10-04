"""Held-out evaluation of the get-up policy and the metrics reported during training.

An evaluation is deterministic (the policy mean, no pull force, no observation noise).
Every robot runs: robot ``i`` replays held-out pose ``i % pose_count`` under its own
randomized physics, so each pose is tried several times. Success (``stood_2s``)
means standing (torso above 0.46 m, tilt below 15 deg, both wheels on the floor)
for 2 continuous seconds within the 10 s episode.
"""
from collections import Counter
from pathlib import Path

import torch

from arthrobot_tasks.humanoid.getup.ppo import ActorCritic
from arthrobot_tasks.humanoid.standing.policy import CHECKPOINT as STANDING_CHECKPOINT

CHECKPOINT = STANDING_CHECKPOINT.parents[1] / 'humanoid_getup/model_25500.pt'
# GetupEnvCfg fields a training run records in its settings, under the same names.
RECORDED_CFG_FIELDS = ('self_contact_weight', 'inherited_contact_weight', 'joint_speed_weight', 'joint_speed_soft',
                       'torque_weight', 'torque_soft_nm', 'torso_rate_weight', 'torso_rate_soft', 'posture_l1_weight',
                       'arm_posture_l1_weight', 'posture_progress_weight', 'ready_tolerance', 'height_schedule_s',
                       'strength_range', 'posture_weight', 'arm_pose_width', 'leg_pose_width')


def apply_run_settings(cfg, settings: dict) -> None:
    """Configure rewards and motor-strength randomization as in the training run."""
    for field in RECORDED_CFG_FIELDS:
        if field in settings:
            value = settings[field]
            setattr(cfg, field, tuple(value) if isinstance(value, list) else value)


def action_bound_at(iteration: int, final_action_bound: float, bound_iterations: int) -> float:
    """Action-bound curriculum: 1 rad, decaying linearly to ``final_action_bound`` over ``bound_iterations``."""
    progress = min(1., iteration / max(1, bound_iterations))
    return 1. - (1. - final_action_bound) * progress


def load_getup_policy(checkpoint: Path = CHECKPOINT, device='cpu') -> tuple[ActorCritic, dict, dict]:
    """The actor-critic of a get-up checkpoint, with the run settings and the curriculum state."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    settings, state = saved['settings'], saved['state']
    model = ActorCritic(settings['observation'], settings['critic'], settings['actions'],
                        len(settings['reward_groups'])).to(device)
    model.load_state_dict(saved['ppo']['model'])
    return model, settings, state


def first_episode_per_env(completed: list[dict]) -> list[dict]:
    first = {}
    for record in completed:
        first.setdefault(record['env'], record)
    return list(first.values())


@torch.no_grad()
def evaluate_held_out(env, model: ActorCritic, first_pose: int, pose_count: int) -> list[dict]:
    """Episode records of deterministic rollouts from bank rows ``first_pose .. first_pose + pose_count - 1``."""
    pull_force, observation_noise = env.pull_force_n, env.cfg.obs_noise
    env.pull_force_n, env.cfg.obs_noise = 0., False
    records, evaluated = [], 0
    while evaluated < pose_count:
        count = min(env.num_envs, pose_count - evaluated)
        env.forced_pose = torch.arange(env.num_envs, device=env.device) % count + first_pose + evaluated
        obs, _ = env.reset()
        env.completed.clear()   # the reset recorded the interrupted training episodes
        for _ in range(int(env.max_episode_length)):
            obs, *_ = env.step(model.mean(obs['policy']).clamp(-1., 1.))
        records += first_episode_per_env(env.completed)
        evaluated += count
    env.pull_force_n, env.cfg.obs_noise = pull_force, observation_noise
    env.forced_pose = None
    env.completed.clear()
    return records


def summarize(records: list[dict], families) -> dict:
    """Success rates and safety metrics of a list of episode records."""
    summary = dict(episodes=len(records))
    if not records:
        return summary

    def mean(key):
        return sum(record[key] for record in records) / len(records)

    def fraction(condition):
        return sum(bool(condition(record)) for record in records) / len(records)

    summary.update(stood_2s=fraction(lambda r: r['stood_2s']), standing_at_end=fraction(lambda r: r['standing_at_end']),
                   max_height_m=mean('max_height_m'))
    summary['new_contact_episodes'] = fraction(lambda r: r['new_contact_s'] > 0)
    summary['new_contact_s'] = mean('new_contact_s')
    summary['inherited_contact_s'] = mean('inherited_contact_s')
    # Stood 2 s, created no self-contact, and released any contact left from the settle within 0.5 s.
    clean = [r['stood_2s'] and r['new_contact_s'] == 0 and r['inherited_contact_s'] <= .5 for r in records]
    # Standing within the standing policy's command range (every limb near the nominal pose) for 1 s.
    ready = [r['best_ready_s'] >= 1. for r in records]
    summary['clean_new_success'] = sum(clean) / len(records)
    summary['ready_success'] = sum(ready) / len(records)
    summary['handover_ready_success'] = sum(a and b for a, b in zip(clean, ready)) / len(records)
    for key in ('peak_limb_speed', 'over_speed_s', 'saturated_s', 'peak_torso_rate'):
        summary[key] = mean(key)
    summary['self_contact_episodes'] = fraction(lambda r: r['self_contact_s'] > 0)
    summary['self_contact_s'] = mean('self_contact_s')
    summary['self_contact_standing_episodes'] = fraction(lambda r: r['self_contact_standing_s'] > 0)
    # Stood 2 s without any self-contact after the motors switched on.
    summary['clean_success'] = fraction(lambda r: r['stood_2s'] and r['self_contact_s'] == 0)
    # Ending pose: every arm joint within the ready tolerance of the standing pose for 1 s while standing.
    summary['arm_ready_success'] = fraction(lambda r: r['best_arm_ready_s'] >= 1.)
    summary['final_arm_error_rad'] = mean('final_arm_error_rad')
    times = [r['first_standing_s'] for r in records if r['first_standing_s'] is not None]
    summary['first_standing_s'] = sum(times) / len(times) if times else None
    for family in families:
        rows = [r for r in records if r['family'] == family]
        if rows:
            summary[f'{family}_stood_2s'] = sum(r['stood_2s'] for r in rows) / len(rows)
            summary[f'{family}_episodes'] = len(rows)
    return summary


def self_contact_by_body(records: list[dict]) -> dict[str, dict]:
    """For each body: the share of episodes with self-contact and the mean contact time per episode (s)."""
    seconds, episodes = Counter(), Counter()
    for record in records:
        for body, time in record['self_contact_bodies'].items():
            seconds[body] += time
            episodes[body] += 1
    return {body: dict(episodes=episodes[body] / len(records), mean_s=total / len(records))
            for body, total in seconds.most_common()}
