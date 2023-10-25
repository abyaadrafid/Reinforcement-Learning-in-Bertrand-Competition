from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.execution.train_ops import train_one_step
from ray.rllib.policy.sample_batch import MultiAgentBatch, concat_samples
from ray.rllib.utils.sgd import standardized


class CustomPPO:
    def __init__(self, algo_id, config):
        self.id = algo_id
        self.config = config
        # Batch of temporally related samples
        self.batch = []

    def process_batch(self, batch):
        self.batch.append(batch.policy_batches.pop(self.id))

    def train(self, algorithm_instance):
        # Create a batch by concating them
        train_batch = concat_samples(self.batch)
        # Update counter
        algorithm_instance._counters[
            f"agent_steps_trained_{self.id}"
        ] += train_batch.agent_steps()
        # Standardize advantages.
        train_batch[Postprocessing.ADVANTAGES] = standardized(
            train_batch[Postprocessing.ADVANTAGES]
        )
        train_batch = MultiAgentBatch({self.id: train_batch}, train_batch.count)
        result = train_one_step(algorithm_instance, train_batch, [self.id])

        # Clear sample buffer
        self.batch.clear()
        return result

    def postprocess(self, algorithm):
        pass
