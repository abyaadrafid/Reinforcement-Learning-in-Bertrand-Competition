import hydra
import ray
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete
from omegaconf import DictConfig, OmegaConf
from ray import tune
from ray.air.config import RunConfig, ScalingConfig
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.train.rl import RLTrainer
from ray.tune.registry import register_env

import wandb
from loggers.action_logger import SharedMetrics
from utils.algo_helpers import env_creator, experiment_config_builder
from utils.custom_trainers import AsymmetricDuopoly


@hydra.main(version_base=None, config_path="config/", config_name="asymmconf.yaml")
def run(cfg: DictConfig):
    # Generated config dictionary for the experiment
    policies = {
        "dqn": (
            DQNTorchPolicy,
            None,
            None,
            # Provide entire AlgorithmConfig object, not just an override.
            DQNConfig()
            .training(target_network_update_freq=500)
            .framework("torch")
            .training(_enable_learner_api=False)
            .rl_module(_enable_rl_module_api=False),
        ),
        "ppo": (
            PPOTorchPolicy,
            None,
            None,
            # Provide entire AlgorithmConfig object, not just an override.
            PPOConfig()
            .training(num_sgd_iter=10, sgd_minibatch_size=128)
            .framework("torch")
            .training(_enable_learner_api=False)
            .rl_module(_enable_rl_module_api=False),
        ),
    }
    # Environment registration for RLLib
    env_creator(env_config=OmegaConf.to_container(cfg.env))
    register_env("duopoly_env", env_creator)
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
    config = (
        AlgorithmConfig()
        .environment("duopoly_env", env_config=spaces | OmegaConf.to_container(cfg.env))
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
        .rollouts(num_rollout_workers=0, rollout_fragment_length=50)
        .reporting(metrics_num_episodes_for_smoothing=30)
        .training(_enable_learner_api=False)
        .rl_module(_enable_rl_module_api=False)
    )
    # Dummy actor to collect agent actions
    shared_metrics_actor = SharedMetrics.remote()

    # Set up trainer
    # trainer = RLTrainer(
    #     run_config=RunConfig(
    #         stop={"training_iteration": cfg.training.iterations},
    #         callbacks=[WandbLoggerCallback(project="BRUH")],
    #     ),
    #     scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
    #     algorithm=AsymmetricDuopoly,
    #     config=config,
    # )
    stop = {"training_iteration": 20}
    # result = trainer.fit()
    results = tune.Tuner(
        AsymmetricDuopoly,
        param_space=config.to_dict(),
        run_config=RunConfig(
            stop=stop, callbacks=[WandbLoggerCallback(project="BRUH")]
        ),
    ).fit()

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
