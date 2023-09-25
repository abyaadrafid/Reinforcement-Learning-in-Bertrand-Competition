from typing import Any, Dict, List, Optional, Union

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

import wandb


class ActionLogger(DefaultCallbacks):
    def __init__(
        self, price_table: wandb.Table, legacy_callbacks_dict: Dict[str, Any] = None
    ):
        self.price_table = price_table
        super().__init__(legacy_callbacks_dict)

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
        envs = base_env.get_sub_environments()
        self.price_table.add_data(*envs[env_index].get_mean_prices())
        return super().on_episode_end(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs
        )
