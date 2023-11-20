import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

GAMMA = 0.95

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyGradientAgent:
    def __init__(
        self,
        lr,
        state_size,
        action_size,
        fc1_size=128,
        fc2_size=256,
    ):
        self.policy_network = PolicyNetwork(
            state_size, fc1_size, fc2_size, action_size
        ).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr)
        self.action_memory = []
        self.reward_memory = []

    def act(self, state):
        action_probs = self.policy_network(state)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.optimizer.zero_grad()
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= GAMMA
            G[t] = G_sum

        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1

        G = (G - mean) / std
        G = torch.tensor(G, dtype=torch.float).to(device)

        loss = 0
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob

        loss.backward()
        self.optimizer.step()

        self.action_memory = []
        self.reward_memory = []


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, fc1_size, fc2_size, action_size):
        super(PolicyNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_size, fc1_size),
            nn.ReLU(),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
            nn.Linear(fc2_size, action_size),
            nn.Softmax(dim=1),
        )

    def forward(self, state):
        x = self.layers(state)
        dist = Categorical(x)

        return dist
