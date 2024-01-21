import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.spaces.box import Box
from gymnasium.spaces.discrete import Discrete
from torch.distributions import Categorical, Normal

from agents.base_agent import BaseAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
GAMMA = 0.99
ENTROPY_COEFF = 0.005


class A2C(BaseAgent):
    """
    Entropy regularized A2C implementation with discrete and continuous action space support.
    """

    def __init__(
        self, id, state_size, FC1_SIZE, FC2_SIZE, action_space: Discrete | Box
    ):
        super().__init__(id)
        self.gamma = GAMMA
        self.action_type = "disc" if isinstance(action_space, Discrete) else "cont"
        self.action_space = action_space
        self.state_size = state_size

        # ActorCritic network that has shared weights
        self.actor_critic = ActorCritic_Network(
            state_size.shape[0],
            self.action_space.shape[0] + 1
            if self.action_type == "cont"
            else self.action_space.n,
            FC1_SIZE,
            FC2_SIZE,
        ).to(device)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), LR)

    def _get_distribution(self, input):
        # Turn action into a distribution to get entropy and to sample from
        # In case of continuous action it gives more stochasticity
        if self.action_type == "disc":
            # Softmax to scale down, not super important to do
            probs = F.softmax(input, -1)
            return Categorical(probs)
        else:
            # For continous action, use actor network logits to generate a Normal distribution
            # Actor network logit used as mean and log_std
            return Normal(input[0], torch.exp(input[1]))

    def act(self, state, eps=0.0):
        # Get action logits without accumulating gradients
        with torch.no_grad():
            action_logits, _ = self.actor_critic.forward(torch.tensor(state).to(device))

        # Turn logits into distribution
        action_distribution = self._get_distribution(action_logits)
        # Sample action from distribution
        action = action_distribution.sample()

        # if self.action_type == "cont" : action = action.sigmoid() * (self.action_space.high-self.action_space.low) + self.action_space.low
        # Gotta denormalize continuous action because they might go outside state space
        # Actor network acts weird when I denormalize, needs more tweaks.
        return action.cpu()

    def step(self, state, action, reward, next_state, done):
        # Get action logits
        action_logits, values = self.actor_critic.forward(
            torch.tensor(state).to(device)
        )
        # Get state value for next state
        _, next_values = self.actor_critic.forward(torch.tensor(next_state).to(device))
        action_distribution = self._get_distribution(action_logits)

        # Calculate log probs for loss calculation
        self.log_probs = action_distribution.log_prob(action)
        # Calculate Entropy for regularization
        self.entropy = action_distribution.entropy()

        # Advantage function estimation
        advantage = reward + self.gamma * next_values * (1 - int(done)) - values

        # Losses
        actor_loss = -self.log_probs * advantage
        critic_loss = 0.5 * (advantage**2)

        loss = actor_loss + critic_loss - ENTROPY_COEFF * self.entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.metrics["reward"] = reward
        self.metrics["entropy"] = self.entropy
        self.losses["total"] = loss.item()
        self.losses["critic"] = critic_loss.item()
        self.losses["actor"] = actor_loss.item()


class ActorCritic_Network(nn.Module):
    def __init__(self, state_size, action_space, fc1_size=128, fc2_size=256):
        super(ActorCritic_Network, self).__init__()

        self.stem = nn.Sequential(
            nn.Linear(state_size, fc1_size),
            nn.ReLU(),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(self.stem, nn.Linear(fc2_size, action_space))

        self.critic = nn.Sequential(
            self.stem,
            nn.Linear(fc2_size, 1),
        )

    def forward(self, x):
        x = x.float()
        probabilities = self.actor(x)
        value = self.critic(x)

        return probabilities, value
