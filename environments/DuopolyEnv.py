from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gym.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import override

from utils.market import Market


class DuopolyEnv(MultiAgentEnv, gym.Env):
    def __init__(self, seed: Optional[int], config: Optional[Dict] = None) -> None:
        super().__init__()
        if not config:
            config = {}

        self.action_space = Box(low=20, high=1500, shape=(1,), dtype=np.float32)

        self.observation_space = Box(
            low=20,
            high=1500,
            shape=(10,),
            dtype=np.float32,
        )
        # RLLib Compatibility
        self.parse_config(config)
        self._init_spaces()
        self._init_states()

    def seed(self, seed: int):
        self.seed = seed

    def _init_spaces(self):

        """
        Initialize spaces for RLLib compatibility
        """
        # if self.action_type is "Discrete" :
        # Use when we decide to allow continuous action spaces
        self.n_features = self.memory_size * self.num_seller
        self._agent_ids = ["agent" + str(i) for i in range(self.num_seller)]

    def _init_states(self):

        # Generate random state
        prices = np.random.uniform(
            low=self.min_price,
            high=self.max_price,
            size=(self.memory_size * self.num_seller),
        )

        """
        Maybe use number of sold/unsold items as additional inputs too
        """
        # sold = np.random.uniform(0, self.max_capacity, (self.memory_size, self.num_seller))
        # self.states = np.hstack((prices, sold))
        self.states = prices
        self.rewards = np.zeros(shape=(self.num_seller))

        self.market = Market(
            self.num_seller,
            self.num_customer,
            self.max_capacity,
            self.max_price,
            self.min_price,
        )

    def _create_states(self, actions: list[int]):
        np.roll(self.states, -self.num_customer)
        for i, action in enumerate(actions):
            self.states[self.memory_size * self.num_seller - i - 1] = action[i]

    def parse_config(self, config):
        self.num_customer = config.get("num_customer", 3000)
        self.max_price = config.get("max_price", 1200)
        self.min_price = config.get("min_price", 500)
        self.num_seller = config.get("num_seller", 2)
        self.max_capacity = config.get("max_capacity", 1000)
        self.memory_size = config.get("memory_size", 5)
        self.seller_ids = config.get("agent_ids", ["agent0", "agent1"])
        self.action_type = config.get(
            "action_type", "Discrete"
        )  # can be continous (not super good for DQN)

    @override(gym.Env)
    def step(self, actions: Dict):
        print(actions)
        if actions:
            actions = self._from_RLLib_API_to_list(actions)

            self._create_states(actions)
            self.rewards = np.array(
                self.market.allocate_items(actions), dtype=np.float32
            )

        return self._build_dictionary()
        # next_state, rewards, dones, truncated, infos

    @override(gym.Env)
    def reset(self, *, seed=None, options=None):
        self._init_states()
        (
            states,
            _,
            _,
            _,
            infos,
        ) = self._build_dictionary()  # next_states, rewards, dones, truncated, infos
        return states, infos

    @override(gym.Env)
    def close(self):
        pass

    @override(gym.Env)
    def render(self):
        pass

    def _build_dictionary(self, reward=None):
        """
        Create dictonary of PlayerID:Observation for RLLib support
        """

        states = {}
        rewards = {}
        dones = {}
        truncateds = {}
        infos = {}
        for i in range(self.num_seller):
            states[self.seller_ids[i]] = self.states
            rewards[self.seller_ids[i]] = self.rewards[i]
            dones[self.seller_ids[i]] = False
            truncateds[self.seller_ids[i]] = False
            infos[self.seller_ids[i]] = {}

        return states, rewards, dones, truncateds, infos

    def _from_RLLib_API_to_list(self, actions):
        """
        Parse RLLib dictonary into lists
        """
        actions = [actions[player_id] for player_id in self.seller_ids]
        return actions

    def _to_RLLib_API(self, observations, rewards, ep_is_done, info):
        """
        Turn our list of observations and rewards into dictonaries
        """

        states_list = {}
        rewards_list = {}
        dones_list = {}

        for i in range(self.NUM_AGENTS):
            states_list[self.player_ids[i]] = observations[i]
            rewards_list[self.player_ids[i]] = rewards[i]
            dones_list[self.player_ids[i]] = ep_is_done

        dones_list["__all__"] = ep_is_done

        return states_list, rewards_list, dones_list, info
