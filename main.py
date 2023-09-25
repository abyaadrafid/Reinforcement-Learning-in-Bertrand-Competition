import sys

sys.path.append("../")

import ray
from ray import tune
from ray.air.config import RunConfig, ScalingConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.train.rl import RLTrainer
from ray.tune.registry import register_env

import wandb
from environments.DuopolyEnv import DuopolyEnv
from loggers.action_logger import ActionLogger

env_config = {"max_price": 900, "min_price": 500, "memory_size": 5, "num_seller": 2}
wandb.init(project="TEST_RLAC")
price_table = wandb.Table(columns=["mean_price_agent_0", "mean_price_agent_1"])


def env_creator(env_config):
    return DuopolyEnv(env_config)


def logger():
    return ActionLogger(price_table)


config = {
    "env": "duopoly_env",
    "framework": "torch",
    "callbacks": logger,
    "multiagent": {
        "policies": {
            "agent0": PolicySpec(
                observation_space=env_creator(env_config).observation_space,
                action_space=env_creator(env_config).action_space,
                config=A2CConfig.overrides(framework_str="torch"),
            ),
            "agent1": PolicySpec(
                observation_space=env_creator(env_config).observation_space,
                action_space=env_creator(env_config).action_space,
                config=A2CConfig.overrides(framework_str="torch"),
            ),
        },
        "policy_mapping_fn": lambda agent_id, *args, **kwargs: agent_id,
    },
    "rollouts": {"batch_mode": "complete_episodes"},
}


def main():
    ray.init()
    env_creator(env_config=env_config)
    register_env("duopoly_env", env_creator)
    trainer = RLTrainer(
        run_config=RunConfig(
            stop={"training_iteration": 5},
            callbacks=[WandbLoggerCallback(project="TEST_RLAC")],
        ),
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
        algorithm="A2C",
        config=config,
    )
    result = trainer.fit()


if __name__ == "__main__":
    main()
