from typing import Dict, Optional

import gym
import numpy as np
from gym.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import override
from sklearn.preprocessing import normalize

from utils.market import Market


class DuopolyEnv(MultiAgentEnv, gym.Env):
    def __init__(self, seed: Optional[int], config: Optional[Dict] = None) -> None:
        super().__init__()
        if not config:
            config = {}

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

        self.action_space = Discrete(5)

        self.observation_space = Box(
            low=self.min_price,
            high=self.max_price,
            shape=(self.memory_size, self.num_seller),
            dtype="int16",
        )
        self.n_features = self.memory_size * self.num_seller
        self._agent_ids = ["agent" + str(i) for i in range(self.num_seller)]

    def _init_states(self):

        # Generate random state
        prices = np.random.randint(
            low=self.min_price,
            high=self.max_price,
            size=(self.memory_size, self.num_seller),
        )

        """
        Maybe use number of sold/unsold items as additional inputs too
        """
        # sold = np.random.uniform(0, self.max_capacity, (self.memory_size, self.num_seller))
        # self.states = np.hstack((prices, sold))
        self.states = prices
        self.market = Market(
            self.num_seller,
            self.num_customer,
            self.max_capacity,
            self.max_price,
            self.min_price,
        )

    def _create_states(self, actions: list[int]):
        np.roll(self.states, -self.states.shape[1])
        self.states[-1] = actions

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
            print(actions)

            self._create_states(actions)
            revenue = np.array(self.market.allocate_items(actions))

            # My understanding : large reward signals arent usually the best thing
            rewards = normalize(revenue)
            return (
                self._build_observation_dictionary(),
                rewards,
                False,
                False,
                {},
            )  # next_state, rewards, dones, info
        else:
            return self._build_observation_dictionary(), 0, False, False, {}

    @override(gym.Env)
    def reset(self, *, seed=None, options=None):
        self._init_states()
        return self._build_observation_dictionary(), {}

    @override(gym.Env)
    def close(self):
        pass

    @override(gym.Env)
    def render(self):
        pass

    def _build_observation_dictionary(self):
        """
        Create dictonary of PlayerID:Observation for RLLib support
        """

        observation = {}
        for i in range(self.num_seller):
            observation[self.seller_ids[i]] = self.states

        return observation

    def _from_RLLib_API_to_list(self, actions):
        """
        Parse RLLib dictonary into lists
        """
        actions = [actions[player_id] for player_id in self.seller_ids]
        return actions

    def _to_RLLib_API(self, observations, rewards, ep_is_done, info):
        """
        Turn our list of observations and rewards into dictoraries
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
