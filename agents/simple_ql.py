import random

import numpy as np

LR = 5e-7
GAMMA = 0.95


class QLearner:
    def __init__(self, id, observation_space, action_space, seed=0):
        self.id = id
        self.observation_space = observation_space
        self.action_size = action_space
        self.timestep = 0
        self._init_q_table()

    def _init_q_table(self):
        DISCRETE_OS_SIZE = [6] * len(self.observation_space.high)
        self.discrete_os_win_size = (
            self.observation_space.high - self.observation_space.low
        ) / DISCRETE_OS_SIZE
        self.q_table = np.random.uniform(
            low=-1, high=0, size=(DISCRETE_OS_SIZE + [self.action_size])
        )

    def _discretize_state(self, state):
        discrete_state = (
            state - self.observation_space.low
        ) / self.discrete_os_win_size
        return tuple(discrete_state.astype(np.int32))

    def act(self, state, eps=0.0):
        rnd = random.random()
        if rnd < eps:
            return np.random.randint(self.action_size)
        else:
            state = self._discretize_state(state)
            return np.argmax(self.q_table[state])

    def learn(self, experience):
        state, action, reward, next_state, done = experience
        if done:
            max_future_q = np.max(self.q_table[next_state])
            current_q = self.q_table[state + (action,)]
            new_q = (1 - LR) * current_q + LR * (reward + GAMMA * max_future_q)
            self.q_table[state + (action,)] = new_q
        else:
            self.q_table[state + (action,)] = 0

    def step(self, state, action, reward, next_state, done):
        state = self._discretize_state(state)
        next_state = self._discretize_state(next_state)
        self.timestep += 1
        self.learn((state, action, reward, next_state, done))
