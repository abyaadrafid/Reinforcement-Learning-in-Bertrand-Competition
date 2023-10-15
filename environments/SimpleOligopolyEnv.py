from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import override

from utils.demand_functions import SimpleMarket


class SimpleOligopolyEnv(MultiAgentEnv, gym.Env):
    def __init__(self, config, seed: Optional[int]) -> None:
        super().__init__()
        self._parse_config(config)

        self.action_space = config.get("action_space")
        self.observation_space = config.get("observation_space")

        self.market = SimpleMarket(config=config.get("market"))

    def _parse_config(self, config):
        self.max_price = config.get("max_price")
        self.min_price = config.get("min_price")
        self.num_sellers = config.get("num_sellers")
        self.num_customers = config.get("num_customers")
        self.memory_size = config.get("memory_size")
        self.action_type = config.get("action_type")
        self.max_steps = config.get("max_steps")
        self.seller_ids = config.get("agent_ids")

    def _init_internal_vars(self):
        self.n_features = self.memory_size * self.num_sellers
        self._agent_ids = self.seller_ids
        self.action_memory = np.zeros(shape=(self.num_sellers, self.max_steps))
        self.curstep = 0

    def _init_states(self):
        self.states = np.random.uniform(
            low=self.min_price,
            high=self.max_price,
            size=(self.memory_size * self.num_sellers),
        )
        self.rewards = np.zeros(shape=(self.num_sellers,))

        if self.action_type == "disc":
            self.possible_actions = [-2, -0.1414, -0.01, 0, 0.01, 0.1414, 2]

    def _create_states(self, actions: list):
        self.states = np.roll(self.states, -self.num_sellers)
        for i, action in enumerate(actions):
            if self.action_type == "disc":
                action = self.possible_actions[action]
            self.states[self.memory_size * self.num_sellers - i - 1] = action

    @override(gym.Env)
    def reset(self, *, seed=None, options=None):
        self._init_states()
        self._init_internal_vars()
        (
            states,
            _,
            _,
            _,
            infos,
        ) = self._build_dictionary()  # next_states, rewards, dones, truncated, infos
        return states, infos

    @override(gym.Env)
    def step(self, actions: Dict):
        if actions:
            actions = self._from_RLLib_API_to_list(actions)
            for idx in range(self.num_sellers):
                self.action_memory[idx][self.curstep] = actions[idx]

            self._create_states(actions)
            self.rewards = self.market.compute_profit(actions)
            self.curstep += 1
        return self._build_dictionary()
        # next_state, rewards, dones, truncated, infos

    def get_mean_prices(self):
        return np.mean(self.action_memory, axis=1)

    def _from_RLLib_API_to_list(self, actions):
        """
        Parse RLLib dictonary into lists
        """
        actions = [actions[player_id] for player_id in self.seller_ids]
        return actions

    def _build_dictionary(self):
        """
        Create dictonary of PlayerID:Observation for RLLib support
        """

        states = {}
        rewards = {}
        dones = {}
        truncateds = {}
        infos = {}
        for i in range(self.num_sellers):
            states[self.seller_ids[i]] = self.states
            rewards[self.seller_ids[i]] = self.rewards[i]
            dones[self.seller_ids[i]] = False if self.curstep < self.max_steps else True
            truncateds[self.seller_ids[i]] = (
                False if self.curstep < self.max_steps else True
            )
            infos[self.seller_ids[i]] = {}
        dones["__all__"] = False if self.curstep < self.max_steps else True
        truncateds["__all__"] = False if self.curstep < self.max_steps else True

        return states, rewards, dones, truncateds, infos