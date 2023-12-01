import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 3e-4


class ActorCritic(nn.Module):
    def __init__(self, state_size, fc1_dims, fc2_dims, action_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_size, fc1_dims), nn.ReLU(), nn.Linear(fc1_dims, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(state_size, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, action_size),
        )
        self.log_std = nn.Parameter(torch.ones(1, action_size) * std)
        self.apply(self.init_weights)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.1)
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


class PPO:
    def __init__(self, id, state_size, fc1_size, fc2_size, action_size, seed=0) -> None:
        self.id = id
        self.state_size = state_size
        self.action_size = action_size.shape[0]
        self.network = ActorCritic(state_size, fc1_size, fc2_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)

    def _compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            )
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def _ppo_iter(
        self, mini_batch_size, states, actions, log_probs, returns, advantage
    ):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[
                rand_ids, :
            ], returns[rand_ids, :], advantage[rand_ids, :]

    def _ppo_update(
        self,
        ppo_epochs,
        mini_batch_size,
        states,
        actions,
        log_probs,
        returns,
        advantages,
        clip_param=0.2,
    ):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self._ppo_iter(
                mini_batch_size, states, actions, log_probs, returns, advantages
            ):
                dist, value = self.network(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
