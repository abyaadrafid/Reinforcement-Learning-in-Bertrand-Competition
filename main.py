import sys

sys.path.append("../")

import ray
from ray import tune
from ray.air.config import RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

import wandb
from environments.DuopolyEnv import DuopolyEnv
from loggers.action_logger import ActionLogger

env_config = {"max_price": 900, "min_price": 500, "memory_size": 5, "num_seller": 2}


def env_creator(env_config):
    return DuopolyEnv(env_config)


config = {
    "env": "duopoly_env",
    "callbacks": ActionLogger,
    "framework": "torch",
    "num_workers": 1,
    "multiagent": {
        "policies": {
            "agent0": PolicySpec(
                observation_space=env_creator(env_config).observation_space,
                action_space=env_creator(env_config).action_space,
                config=PPOConfig.overrides(framework_str="torch"),
            ),
            "agent1": PolicySpec(
                observation_space=env_creator(env_config).observation_space,
                action_space=env_creator(env_config).action_space,
                config=PPOConfig.overrides(framework_str="torch"),
            ),
        },
        "policy_mapping_fn": lambda agent_id, *args, **kwargs: agent_id,
    },
}


def main():
    ray.init()
    wandb.init(project="RLAC_TEST")
    register_env("duopoly_env", env_creator)

    stop = {"training_iteration": 2}
    tune.run("PPO", stop=stop, config=config)


if __name__ == "__main__":
    main()
