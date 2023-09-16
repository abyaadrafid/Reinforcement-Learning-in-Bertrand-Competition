from typing import Dict, Optional

import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import override

from utils.market import Market


class DuopolyEnv(MultiAgentEnv, gym.Env):
    def __init__(self, config: Optional[Dict]) -> None:
        if not config:
            config = {}

        self.parse_config(config)
        self._init_states()

    def _init_states(self):
        prices = np.random.uniform(
            self.min_price, self.max_price, (self.memory_size, self.num_seller)
        )
        sold = np.random.uniform(0, self.max_capacity, (self.memory_size, 1))
        self.states = np.hstack((prices, sold))
        self.market = Market(self.num_seller, self.num_customer, self.max_capacity)

    def parse_config(self, config):
        self.num_customer = config.get("num_customer", 3000)
        self.max_price = config.get("max_price", 1200)
        self.min_price = config.get("min_price", 50)
        self.num_seller = config.get("num_seller", 2)
        self.max_capacity = config.get("max_capacity", 2000)
        self.memory_size = config.get("memory_size", 5)

    @override(gym.Env)
    def step(self, actions: list[int]):
        self.market.allocate_items(actions)

    @override(gym.Env)
    def reset(self):
        self._init_states()
        return self.states

    @override(gym.Env)
    def close(self):
        pass

    @override(gym.Env)
    def render(self):
        pass
