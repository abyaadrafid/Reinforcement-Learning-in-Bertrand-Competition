import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agents.base_agent import BaseAgent

GAMMA = 0.99
LR = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PG(BaseAgent):
    def __init__(self, id, state_size, fc1_size, fc2_size, action_size):
        self.id = id
        self.policy_network = PolicyNetwork(
            state_size, fc1_size, fc2_size, action_size
        ).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), LR)
        self.action_memory = []
        self.reward_memory = []

    def act(self, state, eps=0.0):

        action_probs = self.policy_network(state)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def step(self, state, action, reward, next_state, done):
        self.store_rewards(reward)
        if done:
            self._learn()

    def _learn(self):
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
            nn.Softmax(dim=0),
        )

    def forward(self, state):
        state = torch.Tensor(state).to(device)
        x = self.layers(state)
        dist = Categorical(x)

        return dist
