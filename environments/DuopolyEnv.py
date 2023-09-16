from typing import Dict, Optional

import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import override

from utils.market import Market, Seller


class DuopolyEnv(MultiAgentEnv, gym.Env):
    def __init__(self, config: Optional[Dict]) -> None:
        if not config:
            config = {}

        self.parse_config()
        self.init_states()

    def init_states(self):
        prices = np.random.uniform(
            self.min_price, self.max_price, (self.num_seller, self.memory_size)
        )
        sold = np.random.uniform(0, self.max_capacity, (self.memory_size, 1))
        self.states = np.hstack(prices, sold)

    def parse_config(self, config):
        self.num_customer = config.get("num_customer", 3000)
        self.max_price = config.get("max_price", 1200)
        self.min_price = config.get("min_price", 50)
        self.num_seller = config.get("num_seller", 2)
        self.max_capacity = config.get("max_cap", 2000)
        self.price_preference = np.random.uniform(
            self.max_price, self.min_price, self.num_customer
        )
        self.memory_size = config.get("memory_size", 5)

    def setup_entities(self):
        self.market = Market(self.num_seller)
        self.sellers = Seller(
            name=["agent" + i for i in range(self.num_seller)],
            capacity=self.max_capacity,
        )

    @override(gym.Env)
    def step(self, actions: list[int]):
        self.market.demand.generate_linear(self.min_price, self.max_price)
        demand = self.market.get_demand()
        total_items = self.max_capacity * self.num_customer

    @override(gym.Env)
    def reset(self):
        pass

    @override(gym.Env)
    def close(self):
        pass

    @override(gym.Env)
    def render(self):
        pass
