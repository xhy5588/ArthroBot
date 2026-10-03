"""Training metrics for standing runs: CSV, status file and TensorBoard scalars.

Wraps the rsl_rl runner's ``log`` method; it never changes training itself.
"""
import csv
import json
import math
import statistics
import time

from arthrobot_tasks.humanoid.standing.evaluation import qualify_standing


def total_ppo_loss(losses: dict, value_coefficient: float, entropy_coefficient: float) -> float:
    """The minibatch-mean PPO objective of rsl_rl (without RND or symmetry terms)."""
    return losses['surrogate'] + value_coefficient * losses['value_function'] - entropy_coefficient * losses['entropy']


def install_monitor(runner, env, run_dir) -> None:
    assert runner.alg.rnd is None and runner.alg.symmetry is None
    original_log = runner.log
    loss_history = []
    started = time.monotonic()

    def log(locals_, *args, **kwargs):
        losses = locals_['loss_dict']
        losses['total'] = total_ppo_loss(losses, runner.alg.value_loss_coef, runner.alg.entropy_coef)
        if not all(math.isfinite(float(value)) for value in losses.values()):
            raise FloatingPointError(f'Non-finite PPO loss: {losses}')
        original_log(locals_, *args, **kwargs)
        loss_history.append(float(losses['total']))
        recent = env.completed_episode_records[-100:]

        def recent_mean(key):
            return statistics.mean([record[key] for record in recent]) if recent else 0.

        row = dict(update=locals_['it'] + 1, training_transitions=runner.tot_timesteps,
                   elapsed_seconds=time.monotonic() - started, total_loss=float(losses['total']),
                   total_loss_mean_100=statistics.mean(loss_history[-100:]),
                   value_loss=float(losses['value_function']), policy_loss=float(losses['surrogate']),
                   entropy=float(losses['entropy']), learning_rate=runner.alg.learning_rate,
                   mean_reward=statistics.mean(locals_['rewbuffer']) if locals_['rewbuffer'] else 0.,
                   mean_episode_seconds=recent_mean('seconds'), training_survival_fraction=recent_mean('survived'),
                   training_standing_fraction=qualify_standing(recent)['accepted_fraction'],
                   mean_horizontal_speed_m_s=recent_mean('mean_horizontal_speed_m_s'),
                   collection_seconds=locals_['collection_time'], learning_seconds=locals_['learn_time'])
        row.update(getattr(runner.alg, 'diagnostics', {}))
        saturation, torque_squared, action_change = (env.control_stats / max(1, env.control_stat_steps)).tolist()
        row.update(torque_saturation_fraction=saturation, torque_rms_nm=math.sqrt(torque_squared),
                   action_change_rms=math.sqrt(action_change))
        env.control_stats.zero_()
        env.control_stat_steps = 0
        metrics_path = run_dir / 'metrics.csv'
        write_header = not metrics_path.exists()
        with metrics_path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        status = run_dir / 'status.tmp'
        status.write_text(json.dumps(row, indent=2) + '\n')
        status.replace(run_dir / 'status.json')
        runner.writer.add_scalar('Loss/total_mean_100', row['total_loss_mean_100'], locals_['it'])
        for key in getattr(runner.alg, 'diagnostics', {}):
            runner.writer.add_scalar(f'PPO_Diagnostics/{key}', row[key], locals_['it'])
        for key in ('torque_saturation_fraction', 'torque_rms_nm', 'action_change_rms'):
            runner.writer.add_scalar(f'Control/{key}', row[key], locals_['it'])
        for key in ('mean_episode_seconds', 'training_survival_fraction', 'training_standing_fraction',
                    'mean_horizontal_speed_m_s', 'training_transitions'):
            runner.writer.add_scalar(f'Standing/{key}', row[key], locals_['it'])
        runner.writer.flush()
        runner.standing_status = row

    runner.log = log
    runner.standing_status = {}
