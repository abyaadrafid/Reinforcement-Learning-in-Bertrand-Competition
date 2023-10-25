from ray.rllib.execution.train_ops import train_one_step
from ray.rllib.policy.sample_batch import SampleBatch


class RandomAgent:
    """
    Random Agent compatible with RLLib RandomPolicy
    """

    def __init__(self, id: str) -> None:
        self.id = id

    def process_batch(self, batch):
        pass

    def train(self, algorithm_instance):
        result = train_one_step(algorithm_instance, SampleBatch(), [self.id])
        return result

    def postprocess(self, algorithm):
        pass
