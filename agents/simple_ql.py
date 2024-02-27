import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import math
import random

import numpy as np

from agents.base_agent import BaseAgent

LR = 0.15
GAMMA = 0.95
FIXED_FOR = 50000
FIXED_ACTION = 0
EPS_BETA = 1e-4


class QLearner(BaseAgent):
    def __init__(self, id, observation_space, action_space, fixed: bool, seed=0):
        super().__init__(id)
        self.id = id
        self.observation_space = observation_space
        self.action_size = action_space.n
        self.action_space = action_space
        self.timestep = 0
        self.eps_t = 0

        self.fixed = True if fixed == "True" else False
        self._init_q_table()

    def _init_q_table(self):
        # Q table size = discrete actions * discrete observations
        # Convert observation space into chunks of our choosing
        # MOVE DISCRETE OBSERVATION SPACE TO CONFIG
        DISCRETE_OS_SIZE = [32] * len(self.observation_space.high)
        self.discrete_os_win_size = (
            self.observation_space.high - self.observation_space.low
        ) / DISCRETE_OS_SIZE
        self.q_table = np.random.uniform(
            low=100, high=100, size=(DISCRETE_OS_SIZE + [self.action_size])
        )

    def _discretize_state(self, state):
        # Discretizing state (e.g. 1.908 => 5)
        discrete_state = (
            state - self.observation_space.low
        ) / self.discrete_os_win_size
        return tuple(discrete_state.astype(np.int32))

    def act(self, state):
        # Act in the environment
        if self.fixed and self.timestep <= FIXED_FOR:
            self.eps = 0
            return self.action_space.sample()
        elif self.fixed and self.timestep == FIXED_FOR:
            self.fixed = False
        if not self.fixed:
            self.eps = math.exp(-EPS_BETA * self.eps_t)
            self.eps_t += 1
        rnd = random.random()
        if rnd < self.eps:
            return np.random.randint(self.action_size)
        else:
            state = self._discretize_state(state)
            return np.argmax(self.q_table[state])

    def learn(self, experience):
        state, action, reward, next_state, done = experience
        # Find maximum possible q value in the next state
        max_future_q = np.max(self.q_table[next_state])
        # Current Q values all actions in current state
        current_q = self.q_table[state + (action,)]

        # Q update
        new_q = (1 - LR) * current_q + LR * (reward + GAMMA * max_future_q)
        self.q_table[state + (action,)] = new_q

    def step(self, state, action, reward, next_state, done):
        # Agent step after environment step

        state = self._discretize_state(state)
        next_state = self._discretize_state(next_state)
        self.timestep += 1
        self.metrics["eps"] = self.eps
        self.metrics["reward"] = reward

        # learning
        self.learn((state, action, reward, next_state, done))
