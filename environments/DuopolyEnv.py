from typing import Dict, Optional

import gym
import numpy as np
from ray.rllib.utils import override

from utils.market import Market, Seller


class DuopolyEnv(gym.Env):
    def __init__(self, config: Optional[Dict]) -> None:
        if not config:
            config = {}

        self.parse_config()

    def parse_config(self, config):
        self.num_customer = config.get("num_customer", 200)
        self.max_price = config.get("max_price", 1200)
        self.min_price = config.get("min_price", 50)
        self.num_seller = config.get("num_seller", 2)
        self.max_capacity = config.get("max_cap", 2000)
        self.price_preference = np.random.uniform(
            self.max_price, self.min_price, self.num_customer
        )

    def setup_entities(self):
        self.market = Market()
        self.sellers = Seller(
            name=["agent" + i for i in range(self.num_seller)],
            capacity=self.max_capacity,
        )

    @override(gym.Env)
    def step(self):
        pass

    @override(gym.Env)
    def reset(self):
        pass

    @override(gym.Env)
    def close(self):
        pass

    @override(gym.Env)
    def render(self):
        pass
