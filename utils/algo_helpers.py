import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete
from omegaconf import OmegaConf
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from environments.SimpleOligopolyEnv import SimpleOligopolyEnv
from loggers.action_logger import ActionLogger


def env_creator(env_config):
    match env_config.get("name", None):
        case "duopoly_env":
            return SimpleOligopolyEnv(seed=0, config=env_config)
        case _:
            raise NotImplementedError("Env not supported yet")


def algo_config_builder(cfg):
    match cfg.algo:
        case "DQN":
            config = DQNConfig()
            config.exploration_config.update(
                {
                    "initial_epsilon": cfg.exploration.initial_epsilon,
                    "final_epsilon": cfg.exploration.final_epsilon,
                    "epsilon_timesteps": cfg.exploration.epsilon_timesteps,
                }
            )
        case "A2C":
            config = A2CConfig()
        case "DDPG":
            config = DDPGConfig()
        case "PPO":
            config = PPOConfig()
        case _:
            raise NotImplementedError("Not a valid algorithm")

    return config


def policy_builder(cfg):
    policies = {}
    for agent_id in cfg.env.agent_ids:
        policies[agent_id] = PolicySpec(
            observation_space=env_creator(cfg.env).observation_space,
            action_space=env_creator(cfg.env).action_space,
            config=algo_config_builder(cfg.training),
        )
    return policies


def experiment_config_builder(cfg):
    spaces = {
        "observation_space": Box(
            low=-cfg.env.max_price,
            high=cfg.env.max_price,
            shape=(cfg.env.memory_size * cfg.env.num_sellers,),
        ),
        "action_space": Box(low=-cfg.env.max_price, high=cfg.env.max_price, shape=(1,))
        if cfg.env.action_type == "cont"
        else Discrete(cfg.env.disc_action_size),
    }
    exp_config = {
        "env": cfg.env.name,
        "env_config": spaces | OmegaConf.to_container(cfg.env),
        "framework": "torch",
        "train_batch_size": cfg.training.bs,
        "callbacks": ActionLogger,
        "multiagent": {
            "policies": policy_builder(cfg),
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: agent_id,
        },
    }
    return exp_config
