import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from ray import tune
from ray.air.config import RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.registry import register_env

import wandb
from loggers.action_logger import SharedMetrics
from utils.algo_helpers import env_creator, experiment_config_builder
from utils.custom_trainers import AsymmetricDuopoly


@hydra.main(version_base=None, config_path="config/", config_name="asymmconf.yaml")
def run(cfg: DictConfig):
    # Generated config dictionary for the experiment
    config = experiment_config_builder(cfg)

    # Environment registration for RLLib
    env_creator(env_config=OmegaConf.to_container(cfg.env))
    register_env(cfg.env.name, env_creator)

    # Dummy actor to collect agent actions
    shared_metrics_actor = SharedMetrics.remote()

    # Training loop
    tune.Tuner(
        AsymmetricDuopoly,
        param_space=config,
        run_config=RunConfig(
            stop={"training_iteration": cfg.training.iterations},
            callbacks=[WandbLoggerCallback(project="BRUH")],
        ),
    ).fit()

    # MOVE THE COLLECTION AND LOGGING TO CORRESPONDING MODULE
    # Collect data from custom callbacks
    shared_metrics = ray.get(shared_metrics_actor.get_result.remote())

    # Iterate through the list of custom metrics and send to wandb
    for ep_metric in shared_metrics:
        prices = ep_metric.get("step_last_price")
        for idx, agent_id in enumerate(cfg.env.agent_ids):
            wandb.log({f"{agent_id}_prices": prices[idx]})


if __name__ == "__main__":
    ray.init()
    wandb.init(project="BRUH", group="D_DEBUG")
    run()
