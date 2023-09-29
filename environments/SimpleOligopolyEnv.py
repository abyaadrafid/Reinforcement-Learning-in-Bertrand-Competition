from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from utils.demand_functions import LinearDemand, LogitDemand


class SimpleOligopolyEnv(MultiAgentEnv, gym.Env):
    def __init__(self, seed: Optional[int], config: Optional[Dict] = None) -> None:
        super().__init__()
