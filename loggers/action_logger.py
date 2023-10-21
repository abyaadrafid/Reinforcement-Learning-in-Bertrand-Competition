from typing import Dict

import numpy as np
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID


# Create a Ray actor to store shared metrics data
@ray.remote(name="shared_metrics")
class SharedMetrics:
    def __init__(self):
        self.all_prices = []
        self.mean_prices = []

    def append_price(self, metric_data):
        self.all_prices.append(metric_data)

    def store_mean_price(self):
        self.mean_prices.append(
            {
                "ep_mean_price": np.mean(
                    [
                        step_prices["step_last_price"]
                        for step_prices in self.all_prices[-500:]
                    ],
                    axis=0,
                )
            }
        )

    def get_result(self):
        return self.all_prices, self.mean_prices


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
        shared_metrics_actor.store_mean_price.remote()
        return super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs
        )
