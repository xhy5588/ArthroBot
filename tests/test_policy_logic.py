"""CPU checks of the policy-side logic: hybrid control, rewards, success criteria, PPO and PPO diagnostics."""
import copy

import numpy as np
import pytest
import torch

from arthrobot_assets.humanoid import MOTOR_JOINTS
from arthrobot_tasks.humanoid.getup.evaluation import action_bound_at, summarize
from arthrobot_tasks.humanoid.getup.joint_limits import PlacedHull, hulls_intersect
from arthrobot_tasks.humanoid.getup.poses import merge_banks
from arthrobot_tasks.humanoid.getup.ppo import ActorCritic, MultiCriticPPO, Normalizer
from arthrobot_tasks.humanoid.standing.control import hybrid_torques
from arthrobot_tasks.humanoid.standing.diagnostics import install_diagnostics
from arthrobot_tasks.humanoid.standing.evaluation import qualify_standing
from arthrobot_tasks.humanoid.standing.monitor import total_ppo_loss
from arthrobot_tasks.humanoid.standing.observation import OBSERVATION_SIZE, heading
from arthrobot_tasks.humanoid.standing.policy import standing_actor
from arthrobot_tasks.humanoid.standing.rewards import standing_reward_terms


# ---------------------------------------------------------------- hybrid control
def test_each_action_drives_only_its_motor_within_the_torque_limit():
    zero = torch.zeros(20, 20)
    expected = torch.diag(torch.tensor([2.] * 18 + [1.3] * 2))   # 0.1 x (80 N m/rad x 0.25 rad) and 0.1 x 13 N m
    torch.testing.assert_close(hybrid_torques(torch.eye(20) * .1, zero, zero, zero), expected)
    assert bool((hybrid_torques(torch.full_like(zero, 100.), zero, zero, zero) == 13.).all())


def test_feedback_acts_on_limbs_only():
    zero = torch.zeros(1, 20)
    positions, velocities = zero.clone(), zero.clone()
    positions[:, 0], positions[:, 18:] = .05, 100.
    velocities[:, 1], velocities[:, 18:] = .5, 100.
    expected = zero.clone()
    expected[:, 0], expected[:, 1] = -4., -2.
    torch.testing.assert_close(hybrid_torques(zero, positions, velocities, zero), expected)


def test_joint_error_wraps_for_continuous_joints():
    positions = torch.zeros(1, 20)
    positions[:, 2] = 2 * torch.pi + .02
    assert float(hybrid_torques(torch.zeros(1, 20), positions, torch.zeros(1, 20), torch.zeros(1, 20))[0, 2]) == \
        pytest.approx(-1.6, abs=1e-4)


def test_heading_of_a_yaw_rotation():
    angle = torch.tensor(.7)
    quaternion = torch.stack((torch.cos(angle / 2), torch.zeros(()), torch.zeros(()), torch.sin(angle / 2)))
    assert float(heading(quaternion[None])) == pytest.approx(.7)
    assert standing_actor()(torch.zeros(1, OBSERVATION_SIZE)).shape == (1, 20)


# ---------------------------------------------------------------- standing rewards and success criteria
def standing_terms(height=.528, speed=0., forward=0., action=0., previous=0., clearance=.06):
    return standing_reward_terms(
        torch.tensor([[0., 0., -1.]]), torch.tensor([[speed, 0., 0.]]), torch.zeros(1, 3),
        torch.full((1, 20), action), torch.full((1, 20), previous), torch.full((1, 20), 13. * action),
        torch.ones(1, 2, dtype=torch.bool), torch.tensor([[forward, 0.]]), torch.zeros(1), torch.tensor([height]),
        .528, torch.tensor([clearance]), torch.zeros(1, 18))


def test_standing_reward_separates_good_and_bad_states():
    assert sum(standing_terms().values()) > 0.
    assert standing_terms(height=.40)['height'] - standing_terms(height=.35)['height'] > 1.
    assert standing_terms(speed=.45)['stationary'] - standing_terms(speed=.60)['stationary'] > .5
    assert standing_terms(clearance=.06)['arm_clearance'] > standing_terms(clearance=.03)['arm_clearance'] > \
        standing_terms(clearance=0.)['arm_clearance']


def test_drift_penalty_is_symmetric():
    assert standing_terms()['position'] > standing_terms(forward=.5)['position']
    assert standing_terms(forward=.5)['position'] == standing_terms(forward=-.5)['position']


def test_constant_supporting_torque_is_not_penalized_as_jitter():
    steady, reversal = standing_terms(action=.5, previous=.5), standing_terms(action=.5, previous=-.5)
    assert steady['torque_cost'] == reversal['torque_cost']
    assert steady['action_rate'] == 0. and float(reversal['action_rate']) == pytest.approx(-2.)
    assert sum(steady.values()) > 0.


def test_total_loss_subtracts_the_entropy_bonus():
    assert total_ppo_loss(dict(surrogate=-.1, value_function=2., entropy=3.), value_coefficient=.5,
                          entropy_coefficient=.02) == pytest.approx(.84)


def test_standing_qualification_needs_16_good_trials():
    trial = dict(survived=True, seconds=15., max_torso_tilt_deg=10., minimum_torso_height_m=.5,
                 mean_horizontal_speed_m_s=.02, forward_m=.1, lateral_m=.05, both_wheels_contact_fraction=.95)
    assert not qualify_standing([])['qualified']
    assert not qualify_standing([trial] * 15)['qualified']
    assert qualify_standing([trial] * 16)['qualified']
    for change in ({'survived': False}, {'minimum_torso_height_m': .35}, {'mean_horizontal_speed_m_s': .5},
                   {'forward_m': 1.}, {'max_torso_tilt_deg': 25.}, {'both_wheels_contact_fraction': .5}):
        assert not qualify_standing([dict(trial, **change)] * 16)['qualified']


# ---------------------------------------------------------------- rsl_rl diagnostics (standing)
@pytest.mark.parametrize('normalize', [True, False])
def test_diagnostics_do_not_change_the_ppo_update(normalize):
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic as RslActorCritic
    from tensordict import TensorDict
    torch.manual_seed(17)
    obs = TensorDict({'policy': torch.randn(4, 6)}, batch_size=[4])
    policy = RslActorCritic(obs, {'policy': ['policy'], 'critic': ['policy']}, 20, actor_hidden_dims=[16],
                            critic_hidden_dims=[16], init_noise_std=.25, noise_std_type='log',
                            actor_obs_normalization=normalize)
    algorithm = PPO(policy, device='cpu', num_learning_epochs=2, num_mini_batches=2)
    algorithm.init_storage('rl', 4, 8, obs, [20])
    with torch.inference_mode():
        for _ in range(8):
            algorithm.act(obs)
            obs = TensorDict({'policy': torch.randn(4, 6)}, batch_size=[4])
            algorithm.process_env_step(obs, torch.randn(4), torch.zeros(4, dtype=torch.bool), {})
        algorithm.compute_returns(obs)
    instrumented = copy.deepcopy(algorithm)
    install_diagnostics(instrumented)
    rng_state = torch.get_rng_state()
    expected = algorithm.update()
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng_state)
    assert instrumented.update() == expected
    assert torch.equal(expected_rng, torch.get_rng_state())
    for name, value in algorithm.policy.state_dict().items():
        torch.testing.assert_close(value, instrumented.policy.state_dict()[name], rtol=0, atol=0)
    assert all(np.isfinite(value) for value in instrumented.diagnostics.values())
    assert instrumented.diagnostics['pre_update_kl'] >= 0.
    if not normalize:
        assert instrumented.diagnostics['pre_update_kl'] == pytest.approx(0., abs=1e-6)


# ---------------------------------------------------------------- get-up
def test_action_bound_curriculum():
    assert action_bound_at(0, .25, 3000) == 1.
    assert action_bound_at(1500, .25, 3000) == pytest.approx(.625)
    assert action_bound_at(25500, .25, 3000) == .25


def test_normalizer_matches_batch_statistics():
    normalizer = Normalizer(3, initial_count=1e-8)
    data = torch.randn(1000, 3) * torch.tensor([1., 2., 3.]) + torch.tensor([5., -1., 0.])
    normalizer.update(data[:400])
    normalizer.update(data[400:])
    torch.testing.assert_close(normalizer.mean, data.mean(0), atol=1e-4, rtol=0)
    torch.testing.assert_close(normalizer.var, data.var(0, unbiased=False), atol=1e-3, rtol=1e-4)


def test_multi_critic_ppo_runs_one_update():
    torch.manual_seed(0)
    envs, steps, groups = 8, 6, 5
    model = ActorCritic(10, 12, 4, groups, actor_hidden=(16,), critic_hidden=(16,))
    ppo = MultiCriticPPO(model, (2.5, .1, 1., 1., 1.), envs, steps, 'cpu', minibatches=2, epochs=2)
    obs, critic = torch.randn(envs, 10), torch.randn(envs, 12)
    for step in range(steps):
        actions = ppo.act(obs, critic)
        assert actions.shape == (envs, 4)
        next_obs, next_critic = torch.randn(envs, 10), torch.randn(envs, 12)
        timeout = torch.zeros(envs, dtype=torch.bool)
        timeout[step % envs] = True
        ppo.process(torch.randn(envs, groups), torch.zeros(envs, dtype=torch.bool), timeout, next_obs, next_critic,
                    next_critic)
        obs, critic = next_obs, next_critic
    ppo.returns(critic)
    assert ppo.storage['combined'].mean().abs() < 1e-5
    stats = ppo.update()
    assert all(np.isfinite(value) for value in stats.values())
    assert .05 - 1e-6 <= stats['action_std'] <= 1. + 1e-6
    restored = MultiCriticPPO(ActorCritic(10, 12, 4, groups, actor_hidden=(16,), critic_hidden=(16,)),
                              (2.5, .1, 1., 1., 1.), envs, steps, 'cpu')
    restored.load_state_dict(ppo.state_dict())
    torch.testing.assert_close(restored.model.mean(obs), model.mean(obs))


def test_summary_counts_clean_and_ready_successes():
    record = dict(family='back', stood_2s=True, standing_at_end=True, max_height_m=.55, first_standing_s=2.,
                  new_contact_s=0., inherited_contact_s=0., best_ready_s=1.5, peak_limb_speed=4., over_speed_s=0.,
                  saturated_s=0., peak_torso_rate=2., self_contact_s=0., self_contact_standing_s=0.,
                  best_arm_ready_s=1.2, final_arm_error_rad=.1)
    touching = dict(record, family='front', new_contact_s=.2, self_contact_s=.2, best_ready_s=0., best_arm_ready_s=0.)
    fallen = dict(record, family='front', stood_2s=False, standing_at_end=False, first_standing_s=None,
                  best_arm_ready_s=0., final_arm_error_rad=2.8)
    summary = summarize([record, touching, fallen], ('back', 'front'))
    assert summary['stood_2s'] == pytest.approx(2 / 3)
    assert summary['clean_new_success'] == pytest.approx(1 / 3)
    assert summary['handover_ready_success'] == pytest.approx(1 / 3)
    assert summary['first_standing_s'] == pytest.approx(2.)
    assert summary['front_stood_2s'] == pytest.approx(.5) and summary['back_episodes'] == 1
    assert summary['arm_ready_success'] == pytest.approx(1 / 3)
    assert summary['final_arm_error_rad'] == pytest.approx(1.)


def test_group_diagnostics_report_every_group():
    torch.manual_seed(2)
    model = ActorCritic(6, 7, 2, 3, actor_hidden=(8,), critic_hidden=(8,))
    ppo = MultiCriticPPO(model, (2., 1., 1.), 4, 5, 'cpu', group_names=('a', 'b', 'c'))
    obs, critic = torch.randn(4, 6), torch.randn(4, 7)
    for _ in range(5):
        ppo.act(obs, critic)
        rewards = torch.stack((torch.randn(4), torch.randn(4) * .1, torch.zeros(4)), -1)   # group c never varies
        ppo.process(rewards, torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool), obs, critic, critic)
    ppo.returns(critic)
    diagnostics = ppo.diagnostics
    assert set(diagnostics) == {f'{kind}/{group}' for kind in ('advantage_raw_std', 'advantage_share', 'explained_variance')
                                for group in 'abc'}
    assert all(np.isfinite(value) for value in diagnostics.values())
    assert diagnostics['advantage_share/a'] > diagnostics['advantage_share/b']


def test_convex_hull_intersection_with_clearance():
    def cube(offset):
        corners = np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)]) + offset
        return PlacedHull(corners, corners.min(0), corners.max(0))
    assert hulls_intersect(cube([0., 0., 0.]), cube([.5, .5, .5]), 0.)
    assert not hulls_intersect(cube([0., 0., 0.]), cube([1.01, 0., 0.]), 0.)
    assert hulls_intersect(cube([0., 0., 0.]), cube([1.01, 0., 0.]), .02)


def test_merged_pose_bank_keeps_family_order():
    lying = dict(joint_order=list(MOTOR_JOINTS), families=['back', 'front'],
                 poses=[dict(family='back', joints=[0.] * 20, root_pose=[0.] * 7)])
    upright = dict(joint_order=list(MOTOR_JOINTS), families=['standing'],
                   poses=[dict(family='standing', joints=[0.] * 20, root_pose=[0.] * 7)])
    merged = merge_banks(lying, upright)
    assert merged['families'] == ['back', 'front', 'standing'] and len(merged['poses']) == 2
