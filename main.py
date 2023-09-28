import sys

sys.path.append("../")

import ray
from ray import tune
from ray.air.config import RunConfig, ScalingConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.policy.policy import PolicySpec
from ray.train.rl import RLTrainer
from ray.tune.registry import register_env

import wandb
from environments.DuopolyEnv import DuopolyEnv
from loggers.action_logger import ActionLogger, SharedMetrics

env_config = {"max_price": 900, "min_price": 500, "memory_size": 5, "num_seller": 2}


def env_creator(env_config):
    return DuopolyEnv(env_config)


config = {
    "env": "duopoly_env",
    "num_workers": 2,
    "framework": "torch",
    "callbacks": ActionLogger,
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
}


def main():
    ray.init()
    wandb.init(project="RLAC_CUSTOM_METRICS")
    shared_metrics_actor = SharedMetrics.remote()
    env_creator(env_config=env_config)
    register_env("duopoly_env", env_creator)
    trainer = RLTrainer(
        run_config=RunConfig(
            stop={"training_iteration": 2},
            callbacks=[WandbLoggerCallback(project="TEST_RLAC")],
        ),
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
        algorithm="A2C",
        config=config,
    )
    result = trainer.fit()
    shared_metrics = ray.get(shared_metrics_actor.get_result.remote())
    price_table = wandb.Table(columns=["mean_price_agent_0", "mean_price_agent_1"])

    for ep_metric in shared_metrics:
        price_table.add_data(*ep_metric.get("ep_mean_price"))

    wandb.log({"Price Table": price_table})


if __name__ == "__main__":
    main()
