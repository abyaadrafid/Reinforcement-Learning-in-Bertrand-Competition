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

        # Market object to encapsulate demand and revenue calculation
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

    def _discretize_action_space(self):
        competitive_price = self.market.compute_competitive_prices(self.num_sellers)
        monopoly_price = self.market.compute_monopoly_prices(self.num_sellers)
        # Get evenly spaced actions between competitive and monopoly prices
        # Following scheme from Calvano 2020 , setting ξ=1
        self.possible_actions = np.linspace(
            2 * competitive_price - monopoly_price,
            2 * monopoly_price - competitive_price,
            self.action_space.n,
        )

    def _init_internal_vars(self):
        # n_features is the input size for the policy network
        self.n_features = self.memory_size * self.num_sellers
        # for rllib compatibility
        self._agent_ids = self.seller_ids
        # storing actions for the whole episode
        self.action_memory = np.zeros(shape=(self.num_sellers, 1))
        # termination condition
        self.curstep = 0

    def _init_states(self):
        # randomly initialize states within boundaries
        # states are 1D (flattened for ease, nD possible)
        if self.action_type == "disc":
            self._discretize_action_space()
            self.states = np.random.uniform(
                low=np.min(self.possible_actions),
                high=np.max(self.possible_actions),
                size=(self.memory_size * self.num_sellers),
            )
        else:
            self.states = np.random.uniform(
                low=-self.max_price,
                high=self.max_price,
                size=(self.memory_size * self.num_sellers),
            )

        self.rewards = np.zeros(shape=(self.num_sellers,))

    def _create_states(self, actions: list):
        # shift and replace older states
        self.states = np.roll(self.states, -self.num_sellers)
        for i, action in enumerate(actions):
            # update states accordingly
            self.states[self.memory_size * self.num_sellers - i - 1] = action

    @override(gym.Env)
    def reset(self, *, seed=None, options=None):
        # reinitialize every internal variable
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
            if self.action_type == "disc":
                actions = [self.possible_actions[action] for action in actions]

            # generate new states based on actions
            self._create_states(actions)
            # call the market for revenue/rewards
            self.rewards = self.market.compute_profit(actions)
            # move episode step
            self.curstep += 1
        return self._build_dictionary()
        # next_state, rewards, dones, truncated, infos

    def get_last_prices(self):
        # for logging
        return self.states[-self.num_sellers :]

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
            # update dones and truncateds if step limit is reached
            dones[self.seller_ids[i]] = False if self.curstep < self.max_steps else True
            truncateds[self.seller_ids[i]] = (
                False if self.curstep < self.max_steps else True
            )
            infos[self.seller_ids[i]] = {}
        dones["__all__"] = False if self.curstep < self.max_steps else True
        truncateds["__all__"] = False if self.curstep < self.max_steps else True

        return states, rewards, dones, truncateds, infos
