import sys
from pathlib import Path

from base_agent import BaseAgent
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))


class RandomAgent(BaseAgent):
    """
    Random Agent compatible with RLLib RandomPolicy
    """

    def __init__(
        self, id, obs_space, fc1_size, fc2_size, action_space: Discrete | Box, seed=0
    ) -> None:
        super().__init__(id)
        self.id = id
        self.obs_space = obs_space
        self.action_space = action_space
        self.timestep = 0

    def act(self, state):
        return self.action_space.sample()
