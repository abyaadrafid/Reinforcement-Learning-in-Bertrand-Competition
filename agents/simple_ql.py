import random

import numpy as np

LR = 0.15
GAMMA = 0.95


class QLearner:
    def __init__(self, id, observation_space, action_space, seed=0):
        self.id = id
        self.observation_space = observation_space
        self.action_size = action_space.n
        self.timestep = 0
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

    def act(self, state, eps=0.0):
        # Act in the environment
        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_size)
        else:
            state = self._discretize_state(state)
            return np.argmax(self.q_table[state])

    def learn(self, experience):
        state, action, reward, next_state, done = experience

        # update q table if episode ends
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state + (action,)]
        # Q update
        new_q = (1 - LR) * current_q + LR * (reward + GAMMA * max_future_q)
        self.q_table[state + (action,)] = new_q

    def step(self, state, action, reward, next_state, done):
        # Agent step after environment step
        state = self._discretize_state(state)
        next_state = self._discretize_state(next_state)
        self.timestep += 1
        # learning
        self.learn((state, action, reward, next_state, done))
