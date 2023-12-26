import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.base_agent import BaseAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 1e-4
GAMMA = 0.99


class A2C(BaseAgent):
    def __init__(self, id, state_size, FC1_SIZE, FC2_SIZE, action_size, action_type):
        self.id = id
        self.gamma = GAMMA
        self.action_size = action_size
        self.state_size = state_size
        self.actor_critic = ActorCritic_Network(
            state_size.shape[0],
            action_size.shape[0] if action_type == "cont" else action_size.n,
            FC1_SIZE,
            FC2_SIZE,
        ).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), LR)
        self.log_probs = None

    def act(self, state, eps=0.0):
        action_logits, _ = self.actor_critic.forward(torch.tensor(state).to(device))
        action_probs = F.softmax(action_logits)
        action = action_probs.multinomial(num_samples=1)
        return action

    def step(self, state, action, reward, next_state, done):
        action_logits, _ = self.actor_critic.forward(torch.tensor(state).to(device))
        log_probs = F.log_softmax(action_logits)
        self.log_probs = log_probs[int(action)]

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

        self.actor = nn.Sequential(self.stem, nn.Linear(fc2_size, action_size))

        self.critic = nn.Sequential(
            self.stem,
            nn.Linear(fc2_size, 1),
        )

    def forward(self, x):
        x = x.float()
        value = self.critic(x)
        probabilities = self.actor(x)

        return probabilities, value
