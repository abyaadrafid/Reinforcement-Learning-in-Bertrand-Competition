import random
from typing import Dict, Optional

import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import override
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv


class MultiAgentStockTradingEnv(MultiAgentEnv, gym.Env):

    NAME = "MultiAgentStockTradingEnv"
    NUM_ACTIONS = None
    NUM_AGENTS = 2

    def __init__(self, env: DummyVecEnv):
        self.base_env = env

    @override(gym.Env)
    def step(self, actions: Dict, agent_id: list):
        state, rewards, dones = self.execute_actions(actions, agent_id)
        return state, rewards, dones, {}

    def execute_actions(self, actions: Dict, agent_id: list):
        random.shuffle(agent_id)

        dones = []
        rewards = []
        for agent in agent_id:
            state, reward, done, _ = self.base_env.step(actions[agent])
            rewards.append(reward)
            dones.append(done)
        return state, rewards, dones
