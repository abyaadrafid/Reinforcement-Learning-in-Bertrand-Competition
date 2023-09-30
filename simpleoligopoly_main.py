import hydra
from gymnasium.spaces.box import Box
from omegaconf import DictConfig, OmegaConf
from ray.air.config import RunConfig, ScalingConfig
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.policy.policy import PolicySpec
from ray.train.rl import RLTrainer
from ray.tune.registry import register_env

from environments.SimpleOligopolyEnv import SimpleOligopolyEnv
from loggers.action_logger import ActionLogger, SharedMetrics


def env_creator(env_config):
    return SimpleOligopolyEnv(seed=0, config=env_config)


@hydra.main(version_base=None, config_path="config/", config_name="simpleconf.yaml")
def run(cfg: DictConfig):
    spaces = {
        "observation_space": Box(
            low=cfg.env.min_price,
            high=cfg.env.max_price,
            shape=(cfg.env.memory_size * cfg.env.num_sellers,),
        ),
        "action_space": Box(low=cfg.env.min_price, high=cfg.env.max_price, shape=(1,)),
    }
    config = {
        "env": "duopoly_env",
        "train_batch_size": 16,
        "env_config": spaces | OmegaConf.to_container(cfg.env),
        "num_workers": 1,
        "framework": "torch",
        # "callbacks": ActionLogger,
        "multiagent": {
            "policies": {
                "agent0": PolicySpec(
                    observation_space=env_creator(cfg.env).observation_space,
                    action_space=env_creator(cfg.env).action_space,
                    config=A2CConfig.overrides(framework_str="torch"),
                ),
                "agent1": PolicySpec(
                    observation_space=env_creator(cfg.env).observation_space,
                    action_space=env_creator(cfg.env).action_space,
                    config=A2CConfig.overrides(framework_str="torch"),
                ),
            },
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: agent_id,
        },
    }
    env_creator(env_config=OmegaConf.to_container(cfg.env))
    register_env("duopoly_env", env_creator)

    # Set up trainer
    trainer = RLTrainer(
        run_config=RunConfig(
            # THIS WILL BE SET FROM CONFIG
            stop={"training_iteration": 2},
            # callbacks=[WandbLoggerCallback(project="RLAC_CUSTOM_METRICS")],
        ),
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
        # THIS WILL BE SET FROM CONFIG
        algorithm="A2C",
        config=config,
    )
    result = trainer.fit()


if __name__ == "__main__":
    run()
