import numpy as np


class SimpleMarket:
    def __init__(self, config) -> None:
        self._parse_config(config=config)

    def _parse_config(self, config):
        self.type = config.get("demand_type")
        match self.type:
            case "Logit":
                self.demand_func = LogitDemand(config=config)
            case "Linear":
                self.demand_func = LinearDemand(config=config)
            case default:
                raise NotImplementedError("Use a defined demand type")

    def get_demand(self, prices):
        return self.demand_func.demand(prices=prices)

    def compute_profit(self, prices):
        """
        Return the profits based on the prices
        """
        demand = self.get_demand(prices)
        profits = (prices - self.cost) * demand
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
