from typing import Any, Dict, Optional, Union

import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

import wandb


# Create a Ray actor to store shared metrics data
@ray.remote(name="shared_metrics")
class SharedMetrics:
    def __init__(self):
        self.all_prices = []

    def append_price(self, metric_data):
        self.all_prices.append(metric_data)

    def get_result(self):
        return self.all_prices


class ActionLogger(DefaultCallbacks):
    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy] | None = None,
        episode: Episode | EpisodeV2,
        env_index: int | None = None,
        **kwargs
    ) -> None:
        shared_metrics_actor = ray.get_actor("shared_metrics")
        envs = base_env.get_sub_environments()
        last_prices = envs[env_index].get_last_prices()
        shared_metrics_actor.append_price.remote({"step_last_price": last_prices})

        return super().on_episode_step(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs
        )
