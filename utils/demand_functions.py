import numpy as np
from scipy.optimize import fsolve


class SimpleMarket:
    def __init__(self, config) -> None:
        self._parse_config(config=config)

    def _parse_config(self, config):
        self.type = config.get("demand_type")
        match self.type:
            case "Logit":
                self.demand_func = LogitDemand(config=config.get("logit"))
            case "Linear":
                self.demand_func = LinearDemand(config=config.get("linear"))
            case _:
                raise NotImplementedError("Use a defined demand type")

    def _foc(self, prices):
        """Compute first order condition"""
        d = self.demand_func.demand(prices)
        zero = 1 - (prices - self.demand_func.cost) * (1 - d) / self.demand_func.mu
        return np.squeeze(zero)

    def _foc_monoply(self, prices):
        """Compute first order condition of a monopolist"""
        d = self.demand_func.demand(prices)
        d1 = np.flip(d)
        p1 = np.flip(prices)
        zero = (
            1
            - (prices - self.demand_func.cost) * (1 - d) / self.demand_func.mu
            + (p1 - self.demand_func.cost) * d1 / self.demand_func.mu
        )
        return np.squeeze(zero)

    def compute_competitive_prices(self, num_sellers: int):
        p0 = np.ones((1, num_sellers)) * 1 * self.demand_func.cost
        return fsolve(self._foc, p0)[0]

    def compute_monopoly_prices(self, num_sellers: int):
        p0 = np.ones((1, num_sellers)) * 1 * self.demand_func.cost
        return fsolve(self._foc_monoply, p0)[0]

    def compute_profit(self, prices):
        """
        Return the profits based on the prices
        """
        prices = np.array(prices, dtype=np.float32).reshape(
            -1,
        )
        demand = self.demand_func.demand(prices)
        profits = (prices - self.demand_func.cost) * demand
        return profits


class LogitDemand:
    def __init__(self, config):
        self._parse_config(config=config)

    def _parse_config(self, config):
        # a0 = Inverse index of aggregate demand
        self.a0 = config.get("a0", 0)
        # cost = Marginal cost
        self.cost = config.get("cost", 1)
        # mu = Index of product differentiation
        self.mu = config.get("mu", 0.25)
        # a = Product Quality index
        self.a = config.get("a", 2)

    def demand(self, prices):
        """
        Demand function
        """
        utility = np.exp((self.a - prices) / self.mu)
        demand = utility / (np.sum(utility) + np.exp(self.a0 / self.mu))
        return demand


class LinearDemand:
    def __init__(self, config) -> None:
        self._parse_config(config=config)

    def _parse_config(self, config):
        self.intercept = config.get("intercept", 10)
        self.slope = config.get("slope", -2)
        self.cost = config.get("cost", 1)

    def demand(self, prices):
        demand = self.intercept + self.slope * prices
        return demand
