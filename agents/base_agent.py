class BaseAgent:
    def __init__(self) -> None:
        pass

    def act(self, state, eps=0.0):
        pass

    def step(self, state, action, reward, next_state, done):
        pass
