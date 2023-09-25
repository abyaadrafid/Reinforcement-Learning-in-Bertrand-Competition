import sys

sys.path.append("../")

import ray
from ray import tune
from ray.air.config import RunConfig, ScalingConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.train.rl import RLTrainer
from ray.tune.registry import register_env

import wandb
from environments.DuopolyEnv import DuopolyEnv
from loggers.action_logger import ActionLogger

env_config = {"max_price": 900, "min_price": 500, "memory_size": 5, "num_seller": 2}


def env_creator(env_config):
    return DuopolyEnv(env_config)


config = {
    "env": "duopoly_env",
    "framework": "torch",
    "evaluation_num_workers": 1,
    "evaluation_interval": 1,
    "callbacks": ActionLogger,
    "evaluation_config": {"input": "sampler"},
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
    wandb.init(project="RLAC_Test")
    env_creator(env_config=env_config)
    register_env("duopoly_env", env_creator)
    trainer = RLTrainer(
        run_config=RunConfig(
            stop={"training_iteration": 5},
            callbacks=[WandbLoggerCallback(project="TEST_RLAC")],
        ),
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
        algorithm="PPO",
        config=config,
    )
    result = trainer.fit()


if __name__ == "__main__":
    main()
