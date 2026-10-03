"""Multi-critic PPO in the style of HoST (standalone; does not use RSL-RL).

- One value head per reward group, with GAE computed per group.
- Each group's advantages are normalized separately, then combined with fixed
  group weights (HoST: task 2.5, regularization 0.1, style 1, target 1).
- Asymmetric critic: the actor sees the proprioceptive history only; the critic
  also sees privileged simulator state.
- Smoothness (L2C2-style): penalize differences in policy mean and value between
  each state and a random interpolation towards its successor.
- Timeouts bootstrap from the value of the true final state.

Module and buffer names (``actor_norm``, ``critic_norm``, ``actor``, ``critic``,
``log_std``) are the checkpoint keys; renaming them breaks saved checkpoints.
"""
import torch
from torch import nn


def mlp(inputs: int, hidden: tuple[int, ...], outputs: int) -> nn.Sequential:
    layers, size = [], inputs
    for width in hidden:
        layers += [nn.Linear(size, width), nn.ELU()]
        size = width
    layers.append(nn.Linear(size, outputs))
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    """Running mean and variance; updated by the algorithm, never inside ``forward``."""

    def __init__(self, size: int, initial_count: float = 1e-2):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size))
        self.register_buffer('var', torch.ones(size))
        self.register_buffer('count', torch.tensor(initial_count))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean, batch_var, batch_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, total = batch_mean - self.mean, self.count + batch_count
        self.mean += delta * batch_count / total
        self.var = (self.var * self.count + batch_var * batch_count
                    + delta.square() * self.count * batch_count / total) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / (self.var.sqrt() + 1e-5)).clamp(-5., 5.)


class ActorCritic(nn.Module):
    def __init__(self, actor_obs: int, critic_obs: int, actions: int, groups: int, init_std: float = .8,
                 actor_hidden=(512, 256, 128), critic_hidden=(512, 256)):
        super().__init__()
        self.actor_norm, self.critic_norm = Normalizer(actor_obs), Normalizer(critic_obs)
        self.actor = mlp(actor_obs, actor_hidden, actions)
        self.critic = mlp(critic_obs, critic_hidden, groups)
        self.log_std = nn.Parameter(torch.full((actions,), float(torch.log(torch.tensor(init_std)))))

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.actor_norm(obs))

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """One value per reward group: [N, groups]."""
        return self.critic(self.critic_norm(critic_obs))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mean(obs), self.log_std.clamp(-4.6, 0.).exp())


class MultiCriticPPO:
    def __init__(self, model: ActorCritic, group_weights, num_envs: int, steps: int, device, gamma=.99, lam=.95,
                 clip=.2, epochs=5, minibatches=4, lr=1e-3, desired_kl=.01, entropy=.005, value_coef=1.,
                 max_grad_norm=1., smooth_policy=.1, smooth_value=.1, min_std=.05, max_std=1.):
        self.model, self.device = model, device
        self.weights = torch.tensor(group_weights, device=device)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatches, self.lr, self.desired_kl = epochs, minibatches, lr, desired_kl
        self.entropy, self.value_coef, self.max_grad_norm = entropy, value_coef, max_grad_norm
        self.smooth_policy, self.smooth_value = smooth_policy, smooth_value
        # Actions are clipped to [-1, 1], so a larger std only saturates commands
        # (an early run drifted to std 2.1 under the entropy bonus).
        self.min_std, self.max_std = min_std, max_std
        self.log_std_bounds = (float(torch.log(torch.tensor(min_std))), float(torch.log(torch.tensor(max_std))))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.num_envs, self.steps = num_envs, steps
        self.storage = None
        self.step = 0

    def _allocate(self, obs, critic_obs, actions: int) -> None:
        n, t, g = self.num_envs, self.steps, len(self.weights)

        def zeros(*shape):
            return torch.zeros(t, n, *shape, device=self.device)
        self.storage = dict(obs=zeros(obs.shape[-1]), next_obs=zeros(obs.shape[-1]),
                            critic=zeros(critic_obs.shape[-1]), next_critic=zeros(critic_obs.shape[-1]),
                            actions=zeros(actions), log_prob=zeros(), mu=zeros(actions), sigma=zeros(actions),
                            rewards=zeros(g), values=zeros(g), dones=zeros(), bootstrap=zeros(g))
        self.step = 0

    @torch.no_grad()
    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        distribution = self.model.distribution(obs)
        action = distribution.sample()
        if self.storage is None:
            self._allocate(obs, critic_obs, action.shape[-1])
        storage, i = self.storage, self.step
        storage['obs'][i], storage['critic'][i], storage['actions'][i] = obs, critic_obs, action
        storage['log_prob'][i] = distribution.log_prob(action).sum(-1)
        storage['mu'][i], storage['sigma'][i] = distribution.mean, distribution.stddev
        storage['values'][i] = self.model.value(critic_obs)
        return action

    @torch.no_grad()
    def process(self, rewards, terminated, truncated, next_obs, next_critic, terminal_critic) -> None:
        """Store one step. ``rewards`` is [N, groups]; ``terminal_critic`` is the critic
        observation of the true final state (before the reset)."""
        storage, i = self.storage, self.step
        storage['rewards'][i] = rewards
        storage['dones'][i] = (terminated | truncated).float()
        # Timeout: add gamma * V(final state). Termination: no bootstrap.
        storage['bootstrap'][i] = self.gamma * self.model.value(terminal_critic) * truncated.float()[:, None]
        # Smoothness pairs: the successor state within the same episode, else the state itself.
        same_episode = ~(terminated | truncated)
        storage['next_obs'][i] = torch.where(same_episode[:, None], next_obs, storage['obs'][i])
        storage['next_critic'][i] = torch.where(same_episode[:, None], next_critic, storage['critic'][i])
        self.step += 1

    @torch.no_grad()
    def returns(self, last_critic: torch.Tensor) -> None:
        storage = self.storage
        next_value = self.model.value(last_critic)
        advantage = torch.zeros_like(next_value)
        storage['advantages'] = torch.zeros_like(storage['values'])
        for t in reversed(range(self.steps)):
            alive = (1. - storage['dones'][t])[:, None]
            reward = storage['rewards'][t] + storage['bootstrap'][t]
            delta = reward + self.gamma * next_value * alive - storage['values'][t]
            advantage = delta + self.gamma * self.lam * alive * advantage
            storage['advantages'][t] = advantage
            next_value = storage['values'][t]
        storage['returns'] = storage['advantages'] + storage['values']
        # Normalize each group, then combine with the fixed group weights.
        advantages = storage['advantages']
        advantages = (advantages - advantages.mean((0, 1))) / (advantages.std((0, 1)) + 1e-8)
        combined = (advantages * self.weights).sum(-1)
        storage['combined'] = (combined - combined.mean()) / (combined.std() + 1e-8)

    def update(self) -> dict:
        total = self.steps * self.num_envs
        flat = {key: value.reshape(total, *value.shape[2:]) for key, value in self.storage.items()}
        stats = dict(value=0., surrogate=0., entropy=0., kl=0., clip_fraction=0., smooth_policy=0., smooth_value=0.)
        updates = 0
        for _ in range(self.epochs):
            for batch in torch.randperm(total, device=self.device).chunk(self.minibatches):
                obs, critic_obs = flat['obs'][batch], flat['critic'][batch]
                distribution = self.model.distribution(obs)
                log_prob = distribution.log_prob(flat['actions'][batch]).sum(-1)
                with torch.no_grad():
                    old = torch.distributions.Normal(flat['mu'][batch], flat['sigma'][batch])
                    kl = torch.distributions.kl_divergence(old, distribution).sum(-1).mean()
                if self.desired_kl:
                    if kl > 2. * self.desired_kl:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif kl < self.desired_kl / 2. and kl > 0.:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for group in self.optimizer.param_groups:
                        group['lr'] = self.lr
                ratio = (log_prob - flat['log_prob'][batch]).exp()
                advantage = flat['combined'][batch]
                surrogate = -torch.min(ratio * advantage, ratio.clamp(1 - self.clip, 1 + self.clip) * advantage).mean()
                value = self.model.value(critic_obs)
                old_value, returns = flat['values'][batch], flat['returns'][batch]
                clipped_value = old_value + (value - old_value).clamp(-self.clip, self.clip)
                value_loss = torch.max((value - returns).square(), (clipped_value - returns).square()).mean(0).sum()
                entropy = distribution.entropy().sum(-1).mean()
                # Smoothness: a random interpolation towards the successor state.
                mix = torch.rand(len(batch), 1, device=self.device)
                mixed_obs = obs + mix * (flat['next_obs'][batch] - obs)
                mixed_critic = critic_obs + mix * (flat['next_critic'][batch] - critic_obs)
                smooth_policy = (self.model.mean(mixed_obs) - distribution.mean).square().sum(-1).mean()
                smooth_value = (self.model.value(mixed_critic) - value).square().sum(-1).mean()
                loss = (surrogate + self.value_coef * value_loss - self.entropy * entropy
                        + self.smooth_policy * smooth_policy + self.smooth_value * smooth_value)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    self.model.log_std.clamp_(*self.log_std_bounds)
                updates += 1
                for key, value_ in (('value', value_loss), ('surrogate', surrogate), ('entropy', entropy), ('kl', kl),
                                    ('clip_fraction', ((ratio - 1).abs() > self.clip).float().mean()),
                                    ('smooth_policy', smooth_policy), ('smooth_value', smooth_value)):
                    stats[key] += float(value_)
        self.step = 0
        # The normalizers change only after the update, so acting and learning within one
        # iteration see the same input scaling (the stored log-probabilities stay valid).
        with torch.no_grad():
            self.model.actor_norm.update(flat['obs'])
            self.model.critic_norm.update(flat['critic'])
        stats = {key: total_ / updates for key, total_ in stats.items()}
        stats['learning_rate'] = self.lr
        stats['action_std'] = float(self.model.log_std.exp().mean())
        return stats

    def state_dict(self) -> dict:
        return dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(), lr=self.lr)

    def load_state_dict(self, state: dict, optimizer: bool = True) -> None:
        self.model.load_state_dict(state['model'])
        if optimizer:
            self.optimizer.load_state_dict(state['optimizer'])
            self.lr = state['lr']
