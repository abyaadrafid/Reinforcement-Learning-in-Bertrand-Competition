import random
import sys
from pathlib import Path

import numpy as np

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 1e-5
GAMMA = 0.99


class A2C:
    def __init__(self, id, state_size, FC1_SIZE, FC2_SIZE, action_size):
        self.id = id
        self.gamma = GAMMA
        self.action_size = action_size
        self.actor_critic = ActorCritic_Network(
            state_size, action_size, FC1_SIZE, FC2_SIZE
        ).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), LR)
        self.log_probs = None

    def act(self, state, eps=0.0):
        action_probs, _ = self.actor_critic.forward(torch.tensor(state).to(device))

        rnd = random.random()
        if rnd < eps:
            action = torch.randint(low=0, high=self.action_size, size=(1,))
        else:
            action = action_probs.sample()

        self.log_probs = action_probs.log_prob(action)

        return action.item()

    def step(self, state, action, reward, next_state, done):
        self._learn(state, next_state, reward, done)

    def _learn(self, state, next_state, reward, done):
        _, value = self.actor_critic.forward(torch.tensor(state).to(device))
        _, next_value = self.actor_critic.forward(torch.tensor(next_state).to(device))

        advantage = reward + self.gamma * next_value * (1 - int(done)) - value

        actor_loss = -self.log_probs * advantage
        critic_loss = 0.5 * (advantage**2)

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ActorCritic_Network(nn.Module):
    def __init__(self, state_size, action_size, fc1_size=128, fc2_size=256):
        super(ActorCritic_Network, self).__init__()

        self.stem = nn.Sequential(
            nn.Linear(state_size, fc1_size),
            nn.ReLU(),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            self.stem, nn.Linear(fc2_size, action_size), nn.Softmax()
        )

        self.critic = nn.Sequential(
            self.stem,
            nn.Linear(fc2_size, 1),
        )

    def forward(self, x):
        x = x.float()
        value = self.critic(x)
        probabilities = self.actor(x)
        dist = Categorical(probabilities)

        return dist, value
