class BaseAgent:
    def __init__(self, id, obs_space, fc1, fc2, action_type, action_space) -> None:
        pass

    def act(self, state, eps=0.0):
        pass

    def step(self, state, action, reward, next_state, done):
        pass
