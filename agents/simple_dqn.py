import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.base_agent import BaseAgent

TAU = 1e-3
LR = 1e-5
GAMMA = 0.99
UPDATE_EVERY = 4
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, buffer_size, batch_size, seed):
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add_experience(self, state, action, reward, next_state, done):
        # Wrap as experience
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        # Sample from ReplayMemory
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(
                np.vstack(
                    [
                        experience.state
                        for experience in experiences
                        if experience is not None
                    ]
                )
            )
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack(
                    [
                        experience.action
                        for experience in experiences
                        if experience is not None
                    ]
                )
            )
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack(
                    [
                        experience.reward
                        for experience in experiences
                        if experience is not None
                    ]
                )
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack(
                    [
                        experience.next_state
                        for experience in experiences
                        if experience is not None
                    ]
                )
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack(
                    [
                        experience.done
                        for experience in experiences
                        if experience is not None
                    ]
                ).astype(np.uint8)
            )
            .float()
            .to(device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class DQN(BaseAgent):
    # Deep Q Network algorithm
    def __init__(self, id, state_size, fc1_size, fc2_size, action_size, type, seed=0):
        # fc1_size and fc2_size are linear layer sizes
        self.id = id
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.q_network = DQN_Network(
            self.state_size, fc1_size, fc2_size, self.action_size, seed
        ).to(device)
        self.target_network = DQN_Network(
            self.state_size, fc1_size, fc2_size, self.action_size, seed
        ).to(device)

        # Init optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LR)

        # Replay Memory
        self.memory = ReplayMemory(BUFFER_SIZE, BATCH_SIZE, seed)
        self.timestep = 0
        self.type = type

        # Support for Average reward setting
        if self.type == "avg_reward":
            self.avg_rewards = torch.randn(
                size=(BATCH_SIZE, 1), requires_grad=True, device="cuda"
            )
            self.reward_optim = optim.Adam([self.avg_rewards])

    def step(self, state, action, reward, next_state, done):
        # Agent step after environment step
        self.memory.add_experience(state, action, reward, next_state, done)
        self.timestep += 1
        if self.timestep % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                # Sample from memory
                sampled_experience = self.memory.sample()
                # Learn from samples
                self._learn(sampled_experience)

    def _learn(self, experiences):
        # Unpack experiences
        states, actions, rewards, next_states, dones = experiences

        # Q output from network
        Q_output = self.q_network(states).gather(1, actions)

        # Q output from target network
        action_values = self.target_network(next_states).detach()
        # Choose actions for max
        max_action_values = action_values.max(1)[0].unsqueeze(1)

        # update for average setting
        if self.type == "avg_reward":
            Q_target = rewards - self.avg_rewards + max_action_values
        # update for discounted setting
        else:
            Q_target = rewards + GAMMA * max_action_values * (1 - dones)

        # Calculate loss
        loss = F.mse_loss(Q_output, Q_target)

        # Set zero grad
        if self.type == "avg_reward":
            self.reward_optim.zero_grad()
        self.optimizer.zero_grad()

        # backprop
        loss.backward()

        # optimizer step
        if self.type == "avg_reward":
            self.reward_optim.step()
        self.optimizer.step()

        # Update fixed target network
        self._update_target_network(self.q_network, self.target_network)

    def _update_target_network(self, source_network, target_network):
        for source_parameters, target_parameters in zip(
            source_network.parameters(), target_network.parameters()
        ):
            target_parameters.data.copy_(
                TAU * source_parameters.data + (1 - TAU) * target_parameters.data
            )

    def act(self, state, eps=0.0):
        # Acting on the environment
        rnd = random.random()

        # epsilon action
        if rnd < eps:
            return np.random.randint(self.action_size)
        else:
            # Greedy action
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

            with torch.no_grad():
                action_values = self.q_network(state)

            # Get action from agent
            action = np.argmax(action_values.cpu().data.numpy())
            return action


class DQN_Network(nn.Module):
    def __init__(self, state_size, fc1_size, fc2_size, action_size, seed):
        super(DQN_Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.Sequential(
            nn.Linear(state_size, fc1_size),
            nn.ReLU(),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
            nn.Linear(fc2_size, action_size),
        )

    def forward(self, x):
        return self.layers(x)
