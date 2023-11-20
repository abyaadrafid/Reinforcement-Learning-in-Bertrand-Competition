import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

TAU = 5e-4
ACTOR_LR = 1e-3
CRITIC_LR = 1e-4
GAMMA = 0.99
UPDATE_EVERY = 4
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
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
            .float()
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


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_size, fc1_dims, fc2_dims, action_size) -> None:
        super(CriticNetwork, self).__init__()
        self.state_size = state_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_size = action_size
        self.layers = nn.Sequential(
            nn.Linear(state_size, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
        )
        self.action_value = nn.Linear(action_size, fc2_dims)
        self.q_value = nn.Linear(self.fc2_dims, 1)
        self._init_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def _init_weights(self):
        f1 = 1 / np.sqrt(self.layers[0].weight.data.size()[0])
        f2 = 1 / np.sqrt(self.layers[3].weight.data.size()[0])
        f3 = 0.003

        torch.nn.init.uniform_(self.layers[0].weight.data, -f1, f1)
        torch.nn.init.uniform_(self.layers[0].bias.data, -f1, f1)
        torch.nn.init.uniform_(self.layers[3].weight.data, -f2, f2)
        torch.nn.init.uniform_(self.layers[3].bias.data, -f2, f2)
        torch.nn.init.uniform_(self.q_value.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.q_value.bias.data, -f3, f3)

    def forward(self, state, action):
        state_value = self.layers(state)
        action_value = self.action_value(action)
        q = self.q_value(F.relu(torch.add(state_value, action_value)))

        return q


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_size, fc1_dims, fc2_dims, action_size) -> None:
        super(ActorNetwork, self).__init__()
        self.state_size = state_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_size = action_size
        self.layers = nn.Sequential(
            nn.Linear(state_size, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU(),
        )
        self.mu_value = nn.Linear(self.fc2_dims, action_size)
        self._init_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def _init_weights(self):
        f1 = 1 / np.sqrt(self.layers[0].weight.data.size()[0])
        f2 = 1 / np.sqrt(self.layers[3].weight.data.size()[0])
        f3 = 0.003

        torch.nn.init.uniform_(self.layers[0].weight.data, -f1, f1)
        torch.nn.init.uniform_(self.layers[0].bias.data, -f1, f1)
        torch.nn.init.uniform_(self.layers[3].weight.data, -f2, f2)
        torch.nn.init.uniform_(self.layers[3].bias.data, -f2, f2)
        torch.nn.init.uniform_(self.mu_value.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.mu_value.bias.data, -f3, f3)

    def forward(self, state):
        state_value = self.layers(state)
        a = torch.tanh(self.mu_value(state_value))

        return a


class DDPG:
    def __init__(self, id, state_size, fc1_size, fc2_size, action_size, seed=0) -> None:
        self.id = id
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.memory = ReplayMemory(BUFFER_SIZE, BATCH_SIZE, seed)
        self.actor = ActorNetwork(ACTOR_LR, state_size, fc1_size, fc2_size, action_size)
        self.target_actor = ActorNetwork(
            ACTOR_LR, state_size, fc1_size, fc2_size, action_size
        )
        self.critic = CriticNetwork(
            CRITIC_LR, state_size, fc1_size, fc2_size, action_size
        )
        self.target_critic = CriticNetwork(
            CRITIC_LR, state_size, fc1_size, fc2_size, action_size
        )

        self.timestep = 0
        self._update_target_network(self.actor, self.target_actor, tau=1)
        self._update_target_network(self.critic, self.target_critic, tau=1)

    def step(self, state, action, reward, next_state, done):
        self.memory.add_experience(state, action, reward, next_state, done)
        self.timestep += 1
        if self.timestep % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                sampled_experiences = self.memory.sample()
                self.learn(sampled_experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(next_states)
        target_state_actions = self.target_critic.forward(next_states, target_actions)
        state_actions = self.critic.forward(states, actions)

        targets = []
        for i in range(BATCH_SIZE):
            targets.append(rewards[i] + GAMMA * target_state_actions[i] * dones[i])
        targets = torch.tensor(targets).to(device)
        targets = targets.view(BATCH_SIZE, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(targets, state_actions)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(states)
        self.actor.train()
        actor_loss = -self.critic.forward(states, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self._update_target_network(self.actor, self.target_actor)
        self._update_target_network(self.critic, self.target_critic)

    def _update_target_network(self, source_network, target_network, tau=None):
        if not tau:
            tau = TAU
        for source_parameters, target_parameters in zip(
            source_network.parameters(), target_network.parameters()
        ):
            target_parameters.data.copy_(
                tau * source_parameters.data + (1 - tau) * target_parameters.data
            )

    def act(self, state, eps=0.0):
        rnd = random.random()
        if rnd < eps:
            return np.random.uniform(0, 2)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

            self.actor.eval()
            with torch.no_grad():
                action_values = self.actor(state)

            self.actor.train()
            action = action_values.cpu().data.numpy()
            return action.item()
