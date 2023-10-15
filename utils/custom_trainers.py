"""
To be used exclusively for asymmetric agents
"""
from typing import Callable, Optional

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import NUM_AGENT_STEPS_SAMPLED, NUM_ENV_STEPS_SAMPLED
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
        env_steps = 0
        if hasattr(self, "on_policy_batch"):
            self.on_policy_batch.clear()
        while env_steps < self.cfg.get("on_policy_bs"):
            ma_batches = synchronous_parallel_sample(
                worker_set=self.workers, concat=False
            )
            for batch in ma_batches:
                self._counters[NUM_ENV_STEPS_SAMPLED] += batch.count
                self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
                if hasattr(self, "on_policy_batch"):
                    pass
                if hasattr(self, "local_replay_buffer"):
                    pass
        return super().training_step()
