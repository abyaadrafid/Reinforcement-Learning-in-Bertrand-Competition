from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.execution.train_ops import train_one_step
from ray.rllib.policy.sample_batch import MultiAgentBatch, concat_samples
from ray.rllib.utils.sgd import standardized


class CustomPPO:
    def __init__(self, algo_id, config):
        self.id = algo_id
        self.config = config
        self.batch = []

    def process_batch(self, batch):
        self.batch.append(batch.policy_batches.pop(self.id))

    def train(self, algorithm):
        train_batch = concat_samples(self.batch)
        # Standardize advantages.
        train_batch[Postprocessing.ADVANTAGES] = standardized(
            train_batch[Postprocessing.ADVANTAGES]
        )
        train_batch = MultiAgentBatch({self.id: train_batch}, train_batch.count)
        result = train_one_step(algorithm, train_batch, [self.id])
        return result

    def postprocess(self, algorithm):
        pass
