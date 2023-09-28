from typing import Any, Dict

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
        self.metrics = []

    def append(self, metric_data):
        self.metrics.append(metric_data)

    def get_result(self):
        return self.metrics


class ActionLogger(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode | EpisodeV2 | Exception,
        env_index: int | None = None,
        **kwargs
    ) -> None:
        shared_metrics_actor = ray.get_actor("shared_metrics")
        envs = base_env.get_sub_environments()
        mean_prices = envs[env_index].get_mean_prices()
        shared_metrics_actor.append.remote({"ep_mean_price": mean_prices})
        return super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs
        )
