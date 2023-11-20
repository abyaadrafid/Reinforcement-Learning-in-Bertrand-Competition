import numpy as np
from scipy.optimize import minimize


class RandomAgent:
    def __init__(self) -> None:
        pass


class FixedAgent:
    def __init__(self) -> None:
        pass


class BestResponseAgent:
    def __init__(self, id, cfg) -> None:
        self.id = id
        self.a0 = cfg.get("a0")
        self.cost = cfg.get("cost")
        self.mu = cfg.get("mu")
        self.a = cfg.get("a")

    def br_profits(self, p_index):
        self.prices = np.append(self.prices, p_index)
        utility = np.exp((self.a - self.prices) / self.mu)
        demand = utility / (np.sum(utility) + np.exp(self.a0 / self.mu))
        profits = (self.prices - self.cost) * demand
        return -profits[1]

    def act(self, states, actions):
        self.prices = actions
        pj = minimize(self.br_profits, [1.5], method="L-BFGS-B")
        return pj.x[0]

    def step(self, **args):
        pass
