"""
To be used exclusively for symmetric agents
"""

import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete
from omegaconf import OmegaConf
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.algorithms.ddpg.ddpg_torch_policy import DDPGTorchPolicy
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec

from environments.SimpleOligopolyEnv import SimpleOligopolyEnv
from loggers.action_logger import ActionLogger


def env_creator(env_config):
    """
    Create Environment According to environment specific configs
    """
    match env_config.get("name", None):
        case "duopoly_env":
            return SimpleOligopolyEnv(seed=0, config=env_config)
        case _:
            raise NotImplementedError("Env not supported yet")


def algo_config_builder(cfg, index: int):
    """
    Set Algorithm configs accordingly
    """
    match cfg.algo[index]:
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
        case "A3C":
            config = A3CConfig()
        case "DDPG":
            config = DDPGConfig()
        case "PPO":
            config = PPOConfig()
        case "Random":
            config = {}
        case _:
            print(cfg.algo[index])
            raise NotImplementedError("Algorithm not supported yet")

    return config.training(_enable_learner_api=False).rl_module(
        _enable_rl_module_api=False
    )


def get_policy_class(policy_name):
    match policy_name:
        case "DQN":
            return DQNTorchPolicy
        case "PPO":
            return PPOTorchPolicy
        case "A2C" | "A3C":
            return A3CTorchPolicy
        case "DDPG":
            return DDPGTorchPolicy
        case "Random":
            return RandomPolicy
        case _:
            print(policy_name)
            raise NotImplementedError("Algorithm not supported yet")


def policy_builder(cfg):
    """
    Create a dictionary of agent_id:PolicySpec mapping
    """
    policies = {}
    for idx, agent_id in enumerate(cfg.env.agent_ids):
        policies[agent_id] = PolicySpec(
            policy_class=get_policy_class(cfg.training.algo[idx]),
            observation_space=env_creator(cfg.env).observation_space,
            action_space=env_creator(cfg.env).action_space,
            config=algo_config_builder(cfg.training, idx),
        )
    return policies


def get_trainable_policies(cfg):
    """
    Get a list of policies to train
    Fixed/Random policies dont get updated
    """
    return [
        agent_id if not cfg.training.algo[idx] == "Random" else None
        for idx, agent_id in enumerate(cfg.env.agent_ids)
    ]


def env_config_builder(cfg):
    # Observation and Action spaces made available for RLLib

    spaces = {
        "observation_space": Box(
            low=-cfg.env.max_price,
            high=cfg.env.max_price
            * 2,  # expanding obs space to support actions following calvano
            shape=(cfg.env.memory_size * cfg.env.num_sellers,),
        ),
        "action_space": Box(low=-cfg.env.max_price, high=cfg.env.max_price, shape=(1,))
        if cfg.env.action_type == "cont"
        else Discrete(cfg.env.disc_action_size),
    }
    # Send in configs for the envs too
    return spaces | OmegaConf.to_container(cfg.env)


def experiment_config_builder(cfg, sym: bool = True):
    """
    Create experiment specific configs
    """
    if sym:
        exp_config = {
            "env": cfg.env.name,
            "env_config": env_config_builder(cfg),
            "framework": "torch",
            "train_batch_size": cfg.training.bs,
            "callbacks": ActionLogger,
            "multiagent": {
                "policies": policy_builder(cfg),
                "policy_mapping_fn": lambda agent_id, *args, **kwargs: agent_id,
            },
        }
    # Put every config together
    else:
        exp_config = (
            AlgorithmConfig()
            .environment(env=cfg.env.name, env_config=env_config_builder(cfg))
            .framework("torch")
            .callbacks(ActionLogger)
            .multi_agent(
                policies=policy_builder(cfg),
                policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
                policies_to_train=get_trainable_policies(cfg),
            )
            .training(_enable_learner_api=False)
            .rl_module(_enable_rl_module_api=False)
        ).to_dict()
    return exp_config
