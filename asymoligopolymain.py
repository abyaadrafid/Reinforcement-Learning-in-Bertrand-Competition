import random

import hydra
import numpy as np
import torch.multiprocessing as mp
from gymnasium.spaces import Box, Discrete
from omegaconf import DictConfig

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

import wandb
from agents.simple_a2c import A2C
from agents.simple_ddpg import DDPG
from agents.simple_dqn import DQN
from agents.simple_pg import PG
from agents.simple_ql import QLearner
from environments.SimpleOligopolyEnv import SimpleOligopolyEnv

EPS_START = 1.0
EPS_DECAY = 0.9995
EPS_MIN = 0.01
FC1_SIZE = 16
FC2_SIZE = 32
MAX_EPISODES = 100
MAX_STEPS = 500
PROCESSES = 5


def make_agents(id, type, obs_space, fc1, fc2, action_space, seed):
    match type:
        case "A2C":
            return A2C(id, obs_space, fc1, fc2, action_space)
        case "DQN":
            return DQN(id, obs_space, fc1, fc2, action_space, "avg_reward", seed=seed)
        case "PG":
            return PG(id, obs_space, fc1, fc2, action_space)
        case "DDPG":
            return DDPG(id, obs_space, fc1, fc2, action_space)
        case "QL":
            return QLearner(id, obs_space, action_space)


@hydra.main(version_base=None, config_path="config/", config_name="asymmconf.yaml")
def train(cfg: DictConfig):
    processes = []
    for process_num in range(PROCESSES):
        p = mp.Process(
            target=run, args=(cfg, process_num), name=f"Process_{process_num}"
        )
        p.start()
        processes.append(p)

    for process in processes:
        process.join()


def run(cfg: DictConfig, process_name):
    wandb.init(
        project="QLearning", group="Monopoly", name="6act_1e-6_50k" + str(process_name)
    )
    # init env
    env = SimpleOligopolyEnv(seed=random.randint(0, 255), config=cfg.env)
    env.action_space = (
        Discrete(cfg.env.disc_action_size)
        if env.action_type == "disc"
        else Box(low=cfg.env.min_price, high=cfg.env.max_price, shape=(1,))
    )
    env.observation_space = Box(
        low=cfg.env.min_price,
        high=cfg.env.max_price,
        shape=(cfg.env.memory_size * cfg.env.num_sellers,),
    )

    # make agents
    agents = [
        make_agents(
            id,
            type,
            env.observation_space,
            FC1_SIZE,
            FC2_SIZE,
            env.action_space.n if env.action_type == "disc" else 1,
            seed=random.randint(0, 255),
        )
        for id, type in zip(cfg.env.agent_ids, cfg.training.algo)
    ]
    eps = EPS_START
    all_actions = [] * cfg.env.num_sellers

    for episode in range(1, MAX_EPISODES + 1):
        # Every episode
        states, _ = env.reset()
        for _ in range(MAX_STEPS):
            # Every step

            # Collect actions from each agent and turn them into a dict
            actions = {}
            for agent in agents:
                actions[agent.id] = agent.act(states.get(agent.id), eps)

            # Take multi-agent step in the env
            next_states, rewards, dones, _, infos = env.step(actions)

            # Loop through every agent
            for agent in agents:
                # Collect values for each
                state = states.get(agent.id)
                action = actions.get(agent.id)
                next_state = next_states.get(agent.id)
                reward = rewards.get(agent.id)
                done = dones.get(agent.id)
                # Take training step
                agent.step(state, action, reward, next_state, done)
            states = next_states

            # log each step
            prices = [
                env.possible_actions[action] if env.action_type == "disc" else action
                for action in actions.values()
            ]
            all_actions.append(prices)

            prices_dict = {}
            for idx, agent_id in enumerate(cfg.env.agent_ids):
                prices_dict[f"{agent_id}_prices"] = prices[idx]
            wandb.log(prices_dict)

            if done:
                break
            # Epsilon decay
            eps = max(eps * EPS_DECAY, EPS_MIN)

        # log each episode
        print(f"Progress {episode} / {MAX_EPISODES} :")
        print(f"epsilon : {eps}")
        wandb.log({"epsilon": eps})
        mean_prices_dict = {}

        mean_prices = np.mean(all_actions, axis=0)
        for idx, agent in enumerate(agents):
            mean_prices_dict[f"{agent.id}_mean_prices"] = mean_prices[idx]
            print(f"Mean price by {agent.id} : {mean_prices[idx]}")

        wandb.log(mean_prices_dict)


if __name__ == "__main__":
    train()
