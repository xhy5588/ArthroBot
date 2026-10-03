"""Measure how much each PPO update changes the policy, without changing PPO or its random numbers.

After installation, ``algorithm.diagnostics`` holds, for the latest update: the
KL divergence before and after it over the whole rollout, the clip fraction,
the explained variance of the value function, the fraction of clipped actions,
and the mean action noise.
"""
import torch
from torch.distributions import Normal, kl_divergence


def install_diagnostics(algorithm) -> None:
    assert not algorithm.policy.is_recurrent and algorithm.symmetry is None and algorithm.rnd is None
    original_update = algorithm.update
    algorithm.diagnostics = {}

    @torch.no_grad()
    def measure():
        storage, policy = algorithm.storage, algorithm.policy
        observations = storage.observations.flatten(0, 1)
        mean = policy.act_inference(observations)
        std = policy.log_std.exp() if policy.noise_std_type == 'log' else policy.std
        current = Normal(mean, std.expand_as(mean))
        old = Normal(storage.mu.flatten(0, 1), storage.sigma.flatten(0, 1))
        actions = storage.actions.flatten(0, 1)
        ratio = (current.log_prob(actions).sum(-1) - storage.actions_log_prob.flatten()).exp()
        returns = storage.returns.flatten()
        values = policy.evaluate(observations).flatten()
        unexplained = (returns - values).var(unbiased=False) / returns.var(unbiased=False).clamp_min(1e-8)
        return dict(kl=float(kl_divergence(old, current).sum(-1).mean()),
                    clip_fraction=float(((ratio - 1.).abs() > algorithm.clip_param).float().mean()),
                    explained_variance=float(1. - unexplained),
                    action_clip_fraction=float((actions.abs() > 1.).float().mean()),
                    policy_std_mean=float(std.mean()))

    def update():
        before = measure()
        losses = original_update()
        # rsl_rl's storage.clear() only resets its cursor, so the rollout tensors are still valid.
        after = measure()
        algorithm.diagnostics = dict(pre_update_kl=before['kl'], post_update_kl=after.pop('kl'),
                                     post_update_clip_fraction=after.pop('clip_fraction'), **after)
        return losses

    algorithm.update = update
