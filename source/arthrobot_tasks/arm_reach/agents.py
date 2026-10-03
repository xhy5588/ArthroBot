"""PPO settings for the arm reach task (rsl_rl).

Accuracy peaked near iteration 550 and then declined in two runs (entropy
coefficient 0.01 and 0.001), mostly on targets close to the base, so training
stops at 600 iterations.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ArmReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 600
    save_interval = 50
    experiment_name = 'arm_reach'
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1., actor_obs_normalization=False, critic_obs_normalization=False,
        actor_hidden_dims=[128, 128], critic_hidden_dims=[128, 128], activation='elu')
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1., use_clipped_value_loss=True, clip_param=.2, entropy_coef=.01,
        num_learning_epochs=8, num_mini_batches=4, learning_rate=1e-3, schedule='adaptive',
        gamma=.99, lam=.95, desired_kl=.01, max_grad_norm=1.)
