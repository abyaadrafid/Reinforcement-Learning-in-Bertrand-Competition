import hydra
import ray
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete
from omegaconf import DictConfig, OmegaConf
from ray.air.config import RunConfig, ScalingConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.train.rl import RLTrainer
from ray.tune.registry import register_env

import wandb
from environments.SimpleOligopolyEnv import SimpleOligopolyEnv
from loggers.action_logger import ActionLogger, SharedMetrics


def env_creator(env_config):
    return SimpleOligopolyEnv(seed=0, config=env_config)


def exploration_config_builder(cfg):
    config = DQNConfig()
    config.exploration_config.update(
        {
            "initial_epsilon": cfg.initial_epsilon,
            "final_epsilon": cfg.final_epsilon,
            "epsilon_timesteps": cfg.epsilon_timesteps,
        }
    )
    return config


@hydra.main(version_base=None, config_path="config/", config_name="simpleconf.yaml")
def run(cfg: DictConfig):
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
    config = {
        "env": "duopoly_env",
        "train_batch_size": 64,
        "env_config": spaces | OmegaConf.to_container(cfg.env),
        "num_workers": 2,
        "framework": "torch",
        "callbacks": ActionLogger,
        "multiagent": {
            "policies": {
                "agent0": PolicySpec(
                    observation_space=env_creator(cfg.env).observation_space,
                    action_space=env_creator(cfg.env).action_space,
                    config=exploration_config_builder(cfg.training.exploration),
                ),
                "agent1": PolicySpec(
                    observation_space=env_creator(cfg.env).observation_space,
                    action_space=env_creator(cfg.env).action_space,
                    config=exploration_config_builder(cfg.training.exploration),
                ),
            },
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: agent_id,
        },
    }
    env_creator(env_config=OmegaConf.to_container(cfg.env))
    register_env("duopoly_env", env_creator)

    shared_metrics_actor = SharedMetrics.remote()

    # Set up trainer
    trainer = RLTrainer(
        run_config=RunConfig(
            # THIS WILL BE SET FROM CONFIG
            stop={"training_iteration": 2},
            callbacks=[WandbLoggerCallback(project="RLAC_DEBUG")],
        ),
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
        # THIS WILL BE SET FROM CONFIG
        algorithm="DQN",
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
    ray.init()
    wandb.init(project="RLAC_DEBUG", group="D_DEBUG")
    run()
