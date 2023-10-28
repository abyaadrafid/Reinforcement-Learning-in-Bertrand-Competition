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
from ray.rllib.utils.typing import ResultDict
from ray.tune.logger import Logger

from agents.custom_dqn import CustomDQN
from agents.custom_ppo import CustomPPO
from agents.random_agent import RandomAgent


class AsymmetricDuopoly(Algorithm):
    """Custom trainer with multiple workflows.
    Parses configs, initializes custom workflow for each agent and trains them
    """

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
        # create seperate objects for workflow
        self.algos = [
            self._make_custom_algo(algo, config["agent_ids"][idx])
            for idx, algo in enumerate(config["algo_classes"])
            if not algo == "Random"
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

    # Probably not needed, will removed in another patch
    @override(Algorithm)
    def setup(self, config):
        # Call super's `setup` to create rollout workers.
        super().setup(config)

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        # Collect multiple multi-agent batches
        ma_batches = synchronous_parallel_sample(
            worker_set=self.workers,
            concat=False,
            max_env_steps=self.env_steps_per_training_step,
        )
        # Loop through multi agent batches batches
        for batch in ma_batches:
            # Update counters
            self._counters[NUM_ENV_STEPS_SAMPLED] += batch.count
            self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
            # Send to custom workflow objects to process samples
            for algo in self.algos:
                algo.process_batch(batch)

        train_results = {}

        # train every algo
        for algo in self.algos:
            train_results = algo.train(self)
        # Do some post training operations
        for algo in self.algos:
            algo.postprocess(self)
        return train_results
