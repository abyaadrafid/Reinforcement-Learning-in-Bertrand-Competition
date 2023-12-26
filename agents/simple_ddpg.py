import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

TAU = 5e-4
ACTOR_LR = 1e-5
CRITIC_LR = 1e-6
GAMMA = 0.99
UPDATE_EVERY = 4
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActionNormalizer:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def _agent_to_env(self, action):
        action = self.low + (action + 1.0) * 0.5 * (self.high - self.low)
        action = np.clip(action, self.low, self.high)
        return action

    def _env_to_agent(self, action):
        action = 2 * (action - self.low) / (self.high - self.low) - 1
        action = np.clip(action, self.low, self.high)
        return action


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


class OUNoise(object):
    # OU Noise for action, has exploration built in
    def __init__(
        self,
        action_space,
        mu=0.0,
        theta=0.15,
        max_sigma=0.3,
        min_sigma=0.3,
        decay_period=5000,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.clip(action.cpu() + ou_state, self.low, self.high)


class CriticNetwork(nn.Module):
    # Value network
    def __init__(
        self, beta, state_space, fc1_dims, fc2_dims, action_space, init_w=3e-3
    ):
        super(CriticNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_space + action_space, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )
        self.layers[4].weight.data.uniform_(-init_w, init_w)
        self.layers[4].bias.data.uniform_(-init_w, init_w)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state, action):
        input = torch.cat([state, action], 1)
        return self.layers(input)


class ActorNetwork(nn.Module):
    # Policy network
    def __init__(
        self, alpha, state_space, fc1_dims, fc2_dims, action_space, init_w=3e-3
    ):
        super(ActorNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_space, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, action_space),
        )

        self.layers[4].weight.data.uniform_(-init_w, init_w)
        self.layers[4].bias.data.uniform_(-init_w, init_w)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        return F.tanh(self.layers(state))

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0, 0]


class DDPG:
    def __init__(
        self, id, state_space, fc1_size, fc2_size, action_space, seed=0
    ) -> None:
        self.id = id
        self.state_space = state_space
        self.action_space = action_space
        self.seed = random.seed(seed)
        self.memory = ReplayMemory(BUFFER_SIZE, BATCH_SIZE, seed)
        self.action_normalizer = ActionNormalizer(state_space.low, state_space.high)
        self.actor = ActorNetwork(
            ACTOR_LR,
            state_space.shape[0],
            fc1_size,
            fc2_size,
            self.action_space.shape[0],
        ).to(device)
        self.target_actor = ActorNetwork(
            ACTOR_LR,
            state_space.shape[0],
            fc1_size,
            fc2_size,
            self.action_space.shape[0],
        ).to(device)
        self.critic = CriticNetwork(
            CRITIC_LR,
            state_space.shape[0],
            fc1_size,
            fc2_size,
            self.action_space.shape[0],
        ).to(device)
        self.target_critic = CriticNetwork(
            CRITIC_LR,
            state_space.shape[0],
            fc1_size,
            fc2_size,
            self.action_space.shape[0],
        ).to(device)
        self.ou_noise = OUNoise(action_space)
        self.ou_noise.reset()

        self.timestep = 0

        # intialize target network with current network params
        self._update_target_network(self.actor, self.target_actor, tau=1)
        self._update_target_network(self.critic, self.target_critic, tau=1)

    def step(self, state, action, reward, next_state, done):
        # Agent training step after environment step
        action = self.action_normalizer._env_to_agent(action)
        self.memory.add_experience(state, action, reward, next_state, done)
        self.timestep += 1
        if len(self.memory) > BATCH_SIZE:
            sampled_experiences = self.memory.sample()
            self.learn(sampled_experiences)

    def learn(self, experiences):
        # Learn from experiences (DDPG update)
        states, actions, rewards, next_states, dones = experiences

        actor_loss = self.critic(states, self.actor(states))
        actor_loss = -actor_loss.mean()

        target_action = self.target_actor(next_states)
        target_value = self.target_critic(next_states, target_action.detach())

        expected_value = rewards + (1.0 - dones) * GAMMA * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(states, actions)
        critic_loss = F.mse_loss(value, expected_value.detach())

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        self._update_target_network(self.actor, self.target_actor)
        self._update_target_network(self.critic, self.target_critic)

    def _update_target_network(self, source_network, target_network, tau=TAU):
        for source_parameters, target_parameters in zip(
            source_network.parameters(), target_network.parameters()
        ):
            target_parameters.data.copy_(
                tau * source_parameters.data + (1 - tau) * target_parameters.data
            )

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = self.actor(state)
            # Introduce OU noise to actions
            action_values = self.ou_noise.get_action(action_values, self.timestep)
        action = action_values.cpu().data.numpy()
        return self.action_normalizer._agent_to_env(action.item()).item()
