"""
For now support limited to discrete actions
"""
import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from typing import Callable

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

from agents.custom_dqn import CustomDQN
from agents.custom_ppo import CustomPPO
from agents.random_agent import RandomAgent


class AsymmetricDuopoly(Algorithm):
    def __init__(
        self,
        config: AlgorithmConfig | None = None,
        env=None,
        logger_creator: Callable[[], Logger] | None = None,
        **kwargs
    ):
        self._parse_configs(config)
        super().__init__(config, env, logger_creator, **kwargs)

    def _parse_configs(self, config):
        self.config = config
        self.algos = [
            self._make_custom_algo(algo, config["agent_ids"][idx])
            for idx, algo in enumerate(config["algo_classes"])
        ]
        self.env_steps_per_training_step = config["env_steps_per_training_step"]
        self.agent_ids = config["agent_ids"]

    def _make_custom_algo(self, algo_name: str, algo_id: str):
        match algo_name:
            case "DQN":
                return CustomDQN(algo_id, self.config)
            case "PPO":
                return CustomPPO(algo_id, self.config)
            case "Random":
                return RandomAgent(algo_id)

    @override(Algorithm)
    def setup(self, config):
        # Call super's `setup` to create rollout workers.
        super().setup(config)
        # Create local replay buffer.
        self.local_replay_buffer = MultiAgentReplayBuffer(num_shards=1, capacity=50000)

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        ma_batches = synchronous_parallel_sample(
            worker_set=self.workers,
            concat=False,
            max_env_steps=self.env_steps_per_training_step,
        )
        # Loop through ma-batches (which were collected in parallel).
        for batch in ma_batches:
            self._counters[NUM_ENV_STEPS_SAMPLED] += batch.count
            self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
            for algo in self.algos:
                algo.process_batch(batch)

        train_results = {}

        for algo in self.algos:
            train_results = algo.train(self)

        for algo in self.algos:
            algo.postprocess(self)
        return train_results
