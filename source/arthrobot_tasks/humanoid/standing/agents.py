"""PPO settings for hybrid standing (rsl_rl)."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class StandingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 48
    max_iterations = 1000
    save_interval = 25
    experiment_name = 'humanoid_standing'
    obs_groups = {'policy': ['policy'], 'critic': ['policy']}
    clip_actions = 1.
    # Observations are already bounded and scaled. Running normalization would shift the
    # action distribution during collection, before PPO's first update (measured KL ~0.30).
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=.25, noise_std_type='log', actor_obs_normalization=False, critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 128], critic_hidden_dims=[256, 256, 128], activation='elu')
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1., use_clipped_value_loss=True, clip_param=.2, entropy_coef=.002,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=3e-4, schedule='adaptive',
        gamma=.995, lam=.95, desired_kl=.01, max_grad_norm=1.)
