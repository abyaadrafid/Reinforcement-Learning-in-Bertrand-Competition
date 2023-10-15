"""
To be used exclusively for asymmetric agents
"""
from typing import Callable, Optional

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import (
    MultiAgentReplayBuffer,
)
from ray.rllib.utils.typing import ResultDict
from ray.tune.logger import Logger


class AsymmetricDuopoly(Algorithm):
    def __init__(
        self,
        cfg,
        config: AlgorithmConfig | None = None,
        env=None,
        logger_creator: Callable[[], Logger] | None = None,
        **kwargs
    ):
        super().__init__(config, env, logger_creator, **kwargs)
        self.cfg = cfg

    @override(Algorithm)
    def setup(self, config):
        super().setup(config)
        if "DQN" in self.cfg.values():
            self.local_replay_buffer = MultiAgentReplayBuffer(
                num_shards=1, capacity=50000
            )
        elif "PPO" in self.cfg.values():
            self.on_policy_batch = []

    @override(Algorithm)
    def training_step(self) -> ResultDict:

        return super().training_step()
