import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from ray.air.config import RunConfig, ScalingConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train.rl import RLTrainer
from ray.tune.registry import register_env

import wandb
from loggers.action_logger import SharedMetrics
from utils.algo_helpers import env_creator, experiment_config_builder


@hydra.main(version_base=None, config_path="config/", config_name="simpleconf.yaml")
def run(cfg: DictConfig):
    config = experiment_config_builder(cfg)
    env_creator(env_config=OmegaConf.to_container(cfg.env))
    register_env(cfg.env.name, env_creator)

    shared_metrics_actor = SharedMetrics.remote()

    # Set up trainer
    trainer = RLTrainer(
        run_config=RunConfig(
            # THIS WILL BE SET FROM CONFIG
            stop={"training_iteration": 2},
            callbacks=[WandbLoggerCallback(project="BRUH")],
        ),
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
        # THIS WILL BE SET FROM CONFIG
        algorithm=cfg.training.algo,
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
    wandb.init(project="BRUH", group="D_DEBUG")
    run()
