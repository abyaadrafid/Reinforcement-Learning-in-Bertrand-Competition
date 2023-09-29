import sys

import ray

import wandb

sys.path.append("../")

from ray.air.config import RunConfig, ScalingConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.policy.policy import PolicySpec
from ray.train.rl import RLTrainer
from ray.tune.registry import register_env

from environments.DuopolyEnv import DuopolyEnv
from loggers.action_logger import ActionLogger, SharedMetrics

# config to be passed into the environment
# WILL BE MOVED TO CONFIG FILES LATER WITH HYDRA SUPPORT
env_config = {"max_price": 900, "min_price": 500, "memory_size": 5, "num_seller": 2}

# creating a dummy env to supply the policy builders with the obs and action spaces
def env_creator(env_config):
    return DuopolyEnv(env_config)


# general run configs
# MOST WILL BE MOVED TO CONFIG
# POLICYSPEC WILL HAVE ITS OWN BUILDER BASED ON THE ALGO CHOICE
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

    # initializing wandb
    """
    group naming convention : envtype_algo_symmetrical_numsteps_numiterations
    envtype : D = Duopoly
              O = Oligolpoly
              M = Monopoly
    numsteps = Number of steps to run for each episode.
               In classical setting we would want this to run infinitely
    """
    # NAMES WILL BE SET FROM CONFIG
    wandb.init(project="RLAC_CUSTOM_METRICS", group="D_A2C_SYM_500_500")

    # Initializing a ray remote actor to aggregate data stored in custom callbacks
    shared_metrics_actor = SharedMetrics.remote()

    # Registering env with RLLib
    env_creator(env_config=env_config)
    register_env("duopoly_env", env_creator)

    # Set up trainer
    trainer = RLTrainer(
        run_config=RunConfig(
            # THIS WILL BE SET FROM CONFIG
            stop={"training_iteration": 500},
            callbacks=[WandbLoggerCallback(project="RLAC_CUSTOM_METRICS")],
        ),
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
        # THIS WILL BE SET FROM CONFIG
        algorithm="A2C",
        config=config,
    )
    result = trainer.fit()

    # MOVE THE COLLECTION AND LOGGING TO CORRESPONDING MODULE
    # Collect data from custom callbacks
    shared_metrics = ray.get(shared_metrics_actor.get_result.remote())

    # Make wandb table
    price_table = wandb.Table(columns=["mean_price_agent_0", "mean_price_agent_1"])

    # Iterate through the list of metrics and add to table
    for ep_metric in shared_metrics:
        price_table.add_data(*ep_metric.get("ep_mean_price"))

    # Send it to wandb
    wandb.log({"Price Table": price_table})


if __name__ == "__main__":
    main()
