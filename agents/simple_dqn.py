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

# Fixed network update parameter
TAU = 1  # 1 ==> Hard update
LR = 0.001
GAMMA = 0.99
UPDATE_EVERY = 100
BUFFER_SIZE = int(5000)
BATCH_SIZE = 32
# Update step size for average reward estimation
REWARD_LR = 0.001

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
    """
    Classical Double DQN with a fixed target network for discounted reward setting
    """

    # Deep Q Network algorithm
    def __init__(self, id, obs_space, fc1_size, fc2_size, action_space, seed=0):
        # fc1_size and fc2_size are linear layer sizes
        self.id = id
        self.state_size = obs_space.shape[0]
        self.action_size = action_space.n
        self.seed = random.seed(seed)

        self.q_network = DQN_Network(
            self.state_size, fc1_size, fc2_size, self.action_size, seed
        ).to(device)
        self.target_network = DQN_Network(
            self.state_size, fc1_size, fc2_size, self.action_size, seed
        ).to(device)

        # Init optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=LR, betas=[0.9, 0.999], eps=1e-7
        )

        # Replay Memory
        self.memory = ReplayMemory(BUFFER_SIZE, BATCH_SIZE, seed)
        self.timestep = 0

    def step(self, state, action, reward, next_state, done):
        # Agent step after environment step
        self.memory.add_experience(state, action, reward, next_state, done)
        self.timestep += 1
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

        Q_target = rewards + GAMMA * max_action_values * (1 - dones)

        # Calculate loss
        loss = F.mse_loss(Q_output, Q_target)

        # Set zero grad
        self.optimizer.zero_grad()

        # backprop
        loss.backward()

        # optimizer step
        self.optimizer.step()

        # Update fixed target network
        if self.timestep % UPDATE_EVERY == 0:
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


class AvgDQN(DQN):
    """
    DQN implementation for average reward setting, applicable for continuous environments
    This version does not use discounted future rewards, rather uses the expectation of the future rewards
    """

    def __init__(self, id, obs_space, fc1_size, fc2_size, action_space, seed=0) -> None:
        super().__init__(id, obs_space, fc1_size, fc2_size, action_space, seed)

        # Randomly initialize average rewards (expected rewards)
        self.avg_rewards = torch.randn(
            size=(BATCH_SIZE, 1), requires_grad=True, device=device
        )

    def step(self, state, action, reward, next_state, done):
        # Keep the current experience for updating average rewards
        current_experience = (state, action, reward, next_state)
        # Add to replay buffer
        self.memory.add_experience(state, action, reward, next_state, done)
        self.timestep += 1
        if len(self.memory) > BATCH_SIZE:
            # Sample from memory
            sampled_experience = self.memory.sample()
            # Learn from samples
            self._learn(sampled_experience, current_experience)

    def _learn(self, sampled_experience, current_experience):
        # Update average reward estimation
        self._update_avg_reward(current_experience)
        # Unpack
        states, actions, rewards, next_states, _ = sampled_experience

        # Q output from network
        Q_output = self.q_network(states).gather(1, actions)

        # Q output from target network
        action_values = self.target_network(next_states).detach()
        # Choose actions for max
        max_action_values = action_values.max(1)[0].unsqueeze(1)

        # update for average setting

        Q_target = rewards - self.avg_rewards + max_action_values

        # Calculate loss
        loss = F.mse_loss(Q_output, Q_target)

        # Set zero grad
        self.optimizer.zero_grad()

        # backprop
        loss.backward()

        # optimizer step
        self.optimizer.step()

        # Update fixed target network
        if self.timestep % UPDATE_EVERY == 0:
            self._update_target_network(self.q_network, self.target_network)

    def _update_avg_reward(self, current_experience):

        # Unpacking and converting to tensor
        state, action, reward, next_state = current_experience
        state = torch.tensor(state).float()
        action = torch.tensor(action)
        reward = torch.tensor(reward).float()
        next_state = torch.tensor(next_state).float()

        with torch.no_grad():
            # Q output from Q network for the current sample
            Q_output = self.q_network(state).gather(0, action)

            # Q output from target network
            action_values = self.target_network(next_state).detach()
            # Choose actions for max
            max_action_values = action_values.max()
            # Make bootstrap update on average rewards
            self.avg_rewards = self.avg_rewards + REWARD_LR * (
                reward - self.avg_rewards + max_action_values - Q_output
            )


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
