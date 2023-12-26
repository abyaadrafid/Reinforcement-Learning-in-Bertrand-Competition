class BaseAgent:
    def __init__(self, id) -> None:
        self.id = id
        self.losses = None

    def act(self, state, eps=0.0):
        pass

    def step(self, state, action, reward, next_state, done):
        pass
