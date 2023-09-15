import random

import gymnasium.spaces.box as box
import pandas as pd


class RandomAgent:
    """
    Random Agent for stock trading env
    output range : [-10000, 10000]
    """

    def __init__(self, action_space: box.Box):
        self.low = -10000
        self.high = 10000
        # Move high and low to config or read from somewhere else

        self.action_space = action_space.shape[0]

    def set_seed(self, seed: int):
        self.seed = seed

    def get_actions(self):
        return pd.DataFrame(
            [random.randint(self.low, self.high) for i in range(self.action_space)]
        )
