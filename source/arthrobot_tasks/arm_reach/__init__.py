"""Arm reach task: move the grasp center to random targets in front of the arm.

Registers ``ArthroBot-Arm-Reach-v0`` (1,024 arms) and ``ArthroBot-Arm-Reach-Play-v0``
(16 arms, no observation noise) for Isaac Lab's ``ManagerBasedRLEnv``.
"""
import gymnasium as gym

TASK = 'ArthroBot-Arm-Reach-v0'
PLAY_TASK = 'ArthroBot-Arm-Reach-Play-v0'

for task_id, env_cfg in ((TASK, 'ArmReachEnvCfg'), (PLAY_TASK, 'ArmReachEnvCfg_PLAY')):
    gym.register(
        id=task_id,
        entry_point='isaaclab.envs:ManagerBasedRLEnv',
        disable_env_checker=True,
        kwargs={
            'env_cfg_entry_point': f'{__name__}.env_cfg:{env_cfg}',
            'rsl_rl_cfg_entry_point': f'{__name__}.agents:ArmReachPPORunnerCfg',
        },
    )
