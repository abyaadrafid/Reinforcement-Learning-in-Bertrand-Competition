import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.execution.train_ops import train_one_step
from ray.rllib.utils.metrics import LAST_TARGET_UPDATE_TS, NUM_TARGET_UPDATES
from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer


class CustomDQN:
    def __init__(self, algo_id, config):
        self.id = algo_id
        self.config = config
        self.replay_buffer = ReplayBuffer(capacity=self.config["replay_buffer_size"])

    def train(self, algorithm):
        result = {}
        if self.replay_buffer.__len__() >= self.config["min_replay_samples"]:
            for _ in range(self.config["policy_update_per_training_step"]):
                train_batches = self.replay_buffer.sample(
                    num_items=self.config["dqn_batch_size"]
                )
                result = train_one_step(algorithm, train_batches, [self.id])
                algorithm._counters[
                    f"agent_steps_trained_{self.id}"
                ] += train_batches.agent_steps()
        return result

    def postprocess(self, algorithm_instance):
        if (
            algorithm_instance._counters[f"agent_steps_trained_{self.id}"]
            - algorithm_instance._counters[LAST_TARGET_UPDATE_TS]
            >= algorithm_instance.get_policy(self.id).config[
                "target_network_update_freq"
            ]
        ):
            algorithm_instance.workers.local_worker().get_policy(
                self.id
            ).update_target()
            algorithm_instance._counters[NUM_TARGET_UPDATES] += 1
            algorithm_instance._counters[
                LAST_TARGET_UPDATE_TS
            ] = algorithm_instance._counters[f"agent_steps_trained_{self.id}"]

    def process_batch(self, batch):
        self.replay_buffer.add(batch.policy_batches.pop(self.id))
