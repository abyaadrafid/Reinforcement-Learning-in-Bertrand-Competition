from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec


def algo_config_builder(cfg):
    match cfg.algo:
        case "DQN":
            config = DQNConfig()
        case "A2C":
            config = A2CConfig()
        case "DDPG":
            config = DDPGConfig()
        case "PPO":
            config = PPOConfig()
        case _:
            return NotImplementedError("Not a valid algorithm")
    config.exploration_config.update(
        {
            "initial_epsilon": cfg.exploration.initial_epsilon,
            "final_epsilon": cfg.exploration.final_epsilon,
            "epsilon_timesteps": cfg.exploration.epsilon_timesteps,
        }
    )
    return config
