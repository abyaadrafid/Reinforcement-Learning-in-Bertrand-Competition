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
    def __init__(self, legacy_callbacks_dict: Dict[str, Any] = None):
        wandb.init()
        self.price_table = wandb.Table(columns=["price_value_0", "price_value_1"])
        super().__init__(legacy_callbacks_dict)

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
        envs = base_env.get_sub_environments()
        last_actions = envs[env_index].last()
        print(last_actions)
        # self.price_table.add_data([last_actions[i].astype(float) for i in range(2)])

        return super().on_episode_step(
            worker=worker,
            base_env=base_env,
            policies=policies,
            episode=episode,
            env_index=env_index,
            **kwargs
        )

    def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs) -> None:
        wandb.log({"action_list": self.price_table})

        return super().on_train_result(algorithm=algorithm, result=result, **kwargs)
