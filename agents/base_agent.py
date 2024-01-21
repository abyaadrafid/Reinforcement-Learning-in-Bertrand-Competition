class BaseAgent:
    def __init__(self, id) -> None:
        self.id = id
        self.losses = {}
        self.metrics = {}

    def act(self, state, eps=0.0):
        pass

    def step(self, state, action, reward, next_state, done):
        pass

    def get_metrics(self):
        return self.metrics

    def get_losses(self):
        return self.losses
